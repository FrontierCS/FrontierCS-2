"""Reconstruct task answer files from a submitted unified-diff patch.

Every Preview task accepts exactly one archive member, `payload/solution.patch`.
Tasks whose answer is a file or project (JSON, Python source, a C# project)
publish that answer's pristine starter. The Verifier applies the submitted patch
to its own copy of the starter and scores the reconstructed files exactly as it
scored a directly submitted answer.

This applier is deliberately strict and dependency-free (no `git` in Verifier
images): text-only unified diffs as produced by `git diff` or `diff -u`, exact
context at the stated line numbers, no fuzz or offset search, no binary,
rename, copy, mode or deletion records, and only explicitly allowed paths.
An empty patch means the starter is submitted unchanged.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path, PurePosixPath

PATCH_MEMBER = "payload/solution.patch"
HUNK_RE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
NO_NEWLINE = b"\\ No newline at end of file"
FORBIDDEN_HEADERS = (
    b"GIT binary patch",
    b"Binary files ",
    b"rename from ",
    b"rename to ",
    b"copy from ",
    b"copy to ",
    b"old mode ",
    b"new mode ",
    b"deleted file mode ",
    b"similarity index ",
    b"dissimilarity index ",
)


class PatchRejected(ValueError):
    """The submission is not an acceptable patch for this task."""


def read_patch_member(archive_path: Path, max_archive_bytes: int, max_patch_bytes: int) -> bytes:
    """Return the bytes of the archive's only member, `payload/solution.patch`."""
    if archive_path.is_symlink() or not archive_path.is_file():
        raise PatchRejected("payload is not a regular file")
    if not 0 < archive_path.stat().st_size <= max_archive_bytes:
        raise PatchRejected("payload archive exceeds its byte limit")
    patch: bytes | None = None
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                name = member.name
                if name in seen:
                    raise PatchRejected("payload contains duplicate member paths")
                seen.add(name)
                if name == "payload" and member.isdir():
                    continue
                if name != PATCH_MEMBER or not member.isfile() or member.pax_headers:
                    raise PatchRejected("payload must contain only payload/solution.patch")
                if member.size > max_patch_bytes:
                    raise PatchRejected("solution.patch exceeds its byte limit")
                stream = archive.extractfile(member)
                if stream is None:
                    raise PatchRejected("solution.patch could not be read")
                patch = stream.read(max_patch_bytes + 1)
                if len(patch) != member.size:
                    raise PatchRejected("solution.patch size is inconsistent")
    except PatchRejected:
        raise
    except (tarfile.TarError, EOFError, OSError) as error:
        raise PatchRejected("payload is not an uncompressed POSIX tar") from error
    if patch is None:
        raise PatchRejected("payload entrypoint payload/solution.patch is missing")
    return patch


def _diff_path(raw: bytes, prefix: bytes) -> str | None:
    """Path named by a `---`/`+++` header, without its a/ or b/ prefix."""
    value = raw.split(b"\t", 1)[0]
    if value == b"/dev/null":
        return None
    if not value.startswith(prefix):
        raise PatchRejected("patch file headers must use a/ and b/ prefixes")
    try:
        text = value[len(prefix):].decode("utf-8")
    except UnicodeDecodeError as error:
        raise PatchRejected("patch path is not UTF-8") from error
    if not text or "\\" in text or "\x00" in text or text.startswith("/"):
        raise PatchRejected("patch path is unsafe")
    path = PurePosixPath(text)
    if any(part in {"", ".", ".."} for part in path.parts) or str(path) != text:
        raise PatchRejected("patch path is unsafe")
    return text


def _allowed(path: str, allowed: tuple[str, ...]) -> bool:
    return any(path == item or (item.endswith("/") and path.startswith(item)) for item in allowed)


def _parse(patch: bytes) -> list[tuple[str | None, str, list[tuple[int, int, int, int, list[bytes]]]]]:
    """Parse into (old path, new path, hunks); hunk lines keep their marker."""
    if b"\x00" in patch:
        raise PatchRejected("patch must be text")
    try:
        patch.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PatchRejected("patch must be UTF-8 text") from error
    lines = patch.splitlines(keepends=True)
    files: list[tuple[str | None, str, list]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        bare = line.rstrip(b"\r\n")
        if any(bare.startswith(header) for header in FORBIDDEN_HEADERS):
            raise PatchRejected("binary, rename, copy, mode and deletion records are not accepted")
        if not bare.startswith(b"--- "):
            # Accept only known unified-diff metadata, never arbitrary text.
            if bare.startswith(b"new file mode ") and bare != b"new file mode 100644":
                raise PatchRejected("new files must be regular non-executable files")
            if bare.startswith((b"@@", b"+", b"-", b" ")) and not bare.startswith(b"diff "):
                raise PatchRejected("hunk without file headers")
            if bare and not bare.startswith((b"diff --git ", b"index ", b"new file mode ")):
                raise PatchRejected("unexpected text outside patch records")
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith(b"+++ "):
            raise PatchRejected("file header is incomplete")
        old = _diff_path(bare[4:], b"a/")
        new = _diff_path(lines[index + 1].rstrip(b"\r\n")[4:], b"b/")
        if new is None:
            raise PatchRejected("file deletion is not accepted")
        if old is not None and old != new:
            raise PatchRejected("renames are not accepted")
        index += 2
        hunks = []
        while index < len(lines) and lines[index].startswith(b"@@"):
            match = HUNK_RE.match(lines[index])
            if match is None:
                raise PatchRejected("malformed hunk header")
            old_start, old_count, new_start, new_count = (
                int(match.group(1)),
                int(match.group(2) or 1),
                int(match.group(3)),
                int(match.group(4) or 1),
            )
            index += 1
            body: list[bytes] = []
            seen_old = seen_new = 0
            while index < len(lines) and (seen_old < old_count or seen_new < new_count):
                entry = lines[index]
                marker = entry[:1]
                if marker == b" ":
                    seen_old += 1
                    seen_new += 1
                elif marker == b"-":
                    seen_old += 1
                elif marker == b"+":
                    seen_new += 1
                elif entry.rstrip(b"\r\n") == NO_NEWLINE:
                    if not body:
                        raise PatchRejected("misplaced no-newline marker")
                    body.append(NO_NEWLINE)
                    index += 1
                    continue
                else:
                    raise PatchRejected("hunk is shorter than its header")
                body.append(entry)
                index += 1
            if seen_old != old_count or seen_new != new_count:
                raise PatchRejected("hunk line counts do not match its header")
            if index < len(lines) and lines[index].rstrip(b"\r\n") == NO_NEWLINE:
                body.append(NO_NEWLINE)
                index += 1
            hunks.append((old_start, old_count, new_start, new_count, body))
        if not hunks:
            raise PatchRejected("file header without hunks")
        files.append((old, new, hunks))
    if patch.strip() and not files:
        raise PatchRejected("non-empty patch has no file records")
    return files


def _split(entry: bytes) -> tuple[bytes, bytes]:
    """Marker and line content including its newline."""
    return entry[:1], entry[1:]


def _apply_file(original: list[bytes], hunks: list) -> bytes:
    result: list[bytes] = []
    cursor = 0  # next unconsumed original line (0-based)
    for old_start, old_count, _new_start, _new_count, body in hunks:
        start = old_start - 1 if old_count else old_start
        if start < cursor or start > len(original):
            raise PatchRejected("hunks overlap or start outside the starter file")
        result.extend(original[cursor:start])
        cursor = start
        entries: list[tuple[bytes, bytes]] = []
        for entry in body:
            if entry == NO_NEWLINE:
                marker, content = entries[-1]
                if not content.endswith(b"\n"):
                    raise PatchRejected("duplicate no-newline marker")
                entries[-1] = (marker, content[:-1])
                continue
            entries.append(_split(entry))
        for marker, content in entries:
            if marker in {b" ", b"-"}:
                if cursor >= len(original) or original[cursor] != content:
                    raise PatchRejected("patch context does not match the starter file")
                cursor += 1
            if marker in {b" ", b"+"}:
                result.append(content)
    result.extend(original[cursor:])
    return b"".join(result)


def apply_patch(
    patch: bytes,
    starter: Path,
    allowed: tuple[str, ...],
    *,
    max_files: int = 50,
    max_file_bytes: int,
) -> dict[str, bytes]:
    """Return {path: reconstructed bytes} for every allowed file.

    `allowed` entries are exact relative file paths, or directory prefixes that
    end with `/` (new files may only be created there). Files the patch does
    not touch keep their starter bytes.
    """
    files = _parse(patch)
    if len(files) > max_files:
        raise PatchRejected("patch touches too many files")
    result: dict[str, bytes] = {}
    for item in allowed:
        if not item.endswith("/"):
            result[item] = (starter / item).read_bytes()
    touched: set[str] = set()
    for old, new, hunks in files:
        if not _allowed(new, allowed):
            raise PatchRejected(f"patch changes a path outside the answer: {new}")
        if new in touched:
            raise PatchRejected("patch changes the same file twice")
        touched.add(new)
        source = starter / new
        if old is None:
            if source.exists() or new in result:
                raise PatchRejected("patch creates a file that already exists")
            original: list[bytes] = []
        else:
            if source.is_symlink() or not source.is_file():
                raise PatchRejected(f"patch modifies a file absent from the starter: {new}")
            original = source.read_bytes().splitlines(keepends=True)
        body = _apply_file(original, hunks)
        if len(body) > max_file_bytes:
            raise PatchRejected(f"reconstructed {new} exceeds its byte limit")
        result[new] = body
    return result
