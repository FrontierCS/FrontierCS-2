"""Mirror a Candidate harness build recipe from the platform source repository.

Every directory under ``candidates/`` is a self-contained Docker build context.
The platform source keeps all harnesses under ``candidate-overlays/`` and shares
the submission and GPU clients from ``candidate-overlays/codex``. Mirroring copies
the exact bytes each published image was built from: the official Dockerfile's
COPY sources are placed beside it (shared clients under ``submit/`` and
``gpu-helper/``) and only those COPY source paths are rewritten. ``source.json``
records both Dockerfile hashes and every mirrored file's source path and hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = "https://github.com/FrontierCS/FrontierCS-2"
SOURCE_PREFIX = "candidate-overlays/"
CANDIDATES = "candidates"
ROSTER = "config/candidates.json"

# Shared platform clients and the local name each harness context uses.
SHARED = {
    "codex/frontier_submit.py": "submit/frontier_submit.py",
    "codex/skills/frontier-submit/SKILL.md": "submit/SKILL.md",
    "codex/frontier_gpu.py": "gpu-helper/frontier_gpu.py",
    "codex/frontier_gpu_connect_proxy.py": "gpu-helper/frontier_gpu_connect_proxy.py",
    "codex/skills/frontier-gpu/SKILL.md": "gpu-helper/SKILL.md",
}
# Harness-specific skill documents use the same local names.
HARNESS_SKILLS = {
    "skills/frontier-submit/SKILL.md": "submit/SKILL.md",
    "skills/frontier-gpu/SKILL.md": "gpu-helper/SKILL.md",
}
# The Codex Dockerfile already uses its own directory as the build context.
SELF_CONTEXT = {"codex"}

# Kimi Code is the attested frontier.gpu-helper/v3 Candidate. These names are
# bound by config/kimi-candidate-image.json and scripts/task_selection.py.
KIMI = "kimi-code"
FILES = {
    "launcher.py": "candidate-overlays/kimi-code/launcher.py",
    "submit/frontier_submit.py": "candidate-overlays/codex/frontier_submit.py",
    "submit/SKILL.md": "candidate-overlays/kimi-code/skills/frontier-submit/SKILL.md",
    "gpu-helper/frontier_gpu.py": "candidate-overlays/codex/frontier_gpu.py",
    "gpu-helper/frontier_gpu_connect_proxy.py": "candidate-overlays/codex/frontier_gpu_connect_proxy.py",
    "gpu-helper/SKILL.md": "candidate-overlays/kimi-code/skills/frontier-gpu/SKILL.md",
}
DOCKERFILE = "candidate-overlays/kimi-code/Dockerfile"
INSTALLED = {
    "launcher.py": ("/opt/frontier/candidate/launch", 0o555),
    "submit/frontier_submit.py": ("/opt/frontier/candidate/bin/frontier-submit", 0o555),
    "submit/SKILL.md": ("/opt/frontier/candidate/skills/frontier-submit/SKILL.md", 0o444),
    "gpu-helper/frontier_gpu.py": ("/opt/frontier/candidate/bin/frontier-gpu", 0o555),
    "gpu-helper/frontier_gpu_connect_proxy.py": ("/opt/frontier/candidate/bin/frontier-gpu-connect-proxy", 0o555),
    "gpu-helper/SKILL.md": ("/opt/frontier/candidate/skills/frontier-gpu/SKILL.md", 0o444),
}


def sha256(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def local_name(harness: str, source: str) -> str:
    """Map a COPY source (relative to candidate-overlays/) to its local path."""
    if harness in SELF_CONTEXT:
        return source.removeprefix(harness + "/")
    if source in SHARED:
        return SHARED[source]
    if not source.startswith(harness + "/"):
        raise ValueError(f"{harness} Dockerfile copies a foreign source: {source}")
    rest = source.removeprefix(harness + "/")
    return HARNESS_SKILLS.get(rest, rest)


def copy_sources(harness: str, recipe: bytes) -> list[str]:
    """Return COPY sources (relative to candidate-overlays/) in recipe order."""
    sources: list[str] = []
    for line in recipe.decode().splitlines():
        words = line.split()
        if not words or words[0] != "COPY" or line.rstrip().endswith("\\"):
            continue
        if any(word.startswith("--from") for word in words[1:]):
            continue
        for word in [w for w in words[1:] if not w.startswith("--")][:-1]:
            source = f"{harness}/{word}" if harness in SELF_CONTEXT else word
            if source not in sources:
                sources.append(source)
    if not sources:
        raise ValueError(f"{harness} Dockerfile has no COPY sources")
    return sources


def adapted_dockerfile(body: bytes, harness: str = KIMI) -> bytes:
    """Rewrite only COPY source paths so the harness directory is the context."""
    if harness in SELF_CONTEXT:
        return body
    lines = body.decode().split("\n")
    for index, line in enumerate(lines):
        words = line.split()
        if not words or words[0] != "COPY" or any(w.startswith("--from") for w in words):
            continue
        for word in [w for w in words[1:] if not w.startswith("--")][:-1]:
            old = f" {word} "
            if line.count(old) != 1:
                raise ValueError(f"Platform Dockerfile COPY changed: {word}")
            line = line.replace(old, f" {local_name(harness, word)} ")
        lines[index] = line
    return "\n".join(lines).encode()


def git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(source), *args])


def require_commit(source: Path, commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Platform source must be an explicit full commit")
    subprocess.run(
        ["git", "-C", str(source), "merge-base", "--is-ancestor", commit, "origin/main"],
        check=True,
    )


def roster_commit(harness: str, root: Path = ROOT) -> str:
    roster = json.loads((root / ROSTER).read_text())
    return roster["harnesses"][harness]["recipe"]["source_commit"]


# Wording-only edits applied to the public copy (env deny-list names and comments that
# name internal services). The receipt keeps the source hash and adds the local one.
PUBLIC_EDITS: dict[tuple[str, str], list[tuple[bytes, bytes]]] = {
    ("zcode", "launcher.py"): [(b'"META_API_KEY", "CRS_API_KEY", "LD_LIBRARY_PATH",',
                                b'"META_API_KEY", "LD_LIBRARY_PATH",')],
    # Public copy authenticates with a standard Anthropic API key.
    ("claude-code", "launcher.py"): [
        (b"Official Claude Code, subscription auth, bounded",
         b"Official Claude Code, API-key auth, bounded"),
        (b"oauth = os.environ.get('CLAUDE_CODE_OAUTH_TOKEN', '').strip()",
         b"api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()"),
        (b"if not oauth.startswith('sk-ant-oat01-') or len(oauth) > 16384 or",
         b"if not api_key.startswith('sk-ant-') or len(api_key) > 16384 or"),
        (b"raise LaunchError('missing or invalid subscription credential/model')",
         b"raise LaunchError('missing or invalid API key/model')"),
        (b"child.update({'CLAUDE_CODE_OAUTH_TOKEN': oauth,",
         b"child.update({'ANTHROPIC_API_KEY': api_key,"),
        (b"return child, upload, [oauth.encode(), bearer.encode()], model, effort",
         b"return child, upload, [api_key.encode(), bearer.encode()], model, effort"),
    ],
    ("codex", "launcher.py"): [(b"# Explicit Preview mode preserves its operator-supplied provider key.",
                                b"# Explicit direct mode keeps its operator-supplied provider key.")],
}


def mirror(source: Path, harness: str, commit: str) -> tuple[dict[str, bytes], dict]:
    """Read one harness at ``commit``; return local files and the source receipt."""
    require_commit(source, commit)
    dockerfile = f"{SOURCE_PREFIX}{harness}/Dockerfile"
    official = git(source, "show", f"{commit}:{dockerfile}")
    contents: dict[str, bytes] = {}
    files: dict[str, dict[str, str]] = {}
    for item in copy_sources(harness, official):
        name = local_name(harness, item)
        if name in contents:
            raise ValueError(f"{harness}: two sources map to {name}")
        body = git(source, "show", f"{commit}:{SOURCE_PREFIX}{item}")
        files[name] = {"source": SOURCE_PREFIX + item, "sha256": sha256(body)}
        for before, after in PUBLIC_EDITS.get((harness, name), []):
            if body.count(before) != 1:
                raise ValueError(f"{harness}/{name}: public edit does not apply")
            body = body.replace(before, after)
        if (harness, name) in PUBLIC_EDITS:
            files[name]["local_sha256"] = sha256(body)
        contents[name] = body
    local = adapted_dockerfile(official, harness)
    contents["Dockerfile"] = local
    receipt = {
        "repository": SOURCE_REPOSITORY,
        "platform_source_commit": commit,
        "dockerfile": {
            "source": dockerfile,
            "sha256": sha256(official),
            "local_sha256": sha256(local),
        },
        "files": files,
    }
    return contents, receipt


def receipts_for(receipt: dict) -> dict[str, dict]:
    receipts = {"source.json": receipt}
    for subdirectory in ("submit", "gpu-helper"):
        nested = {name.removeprefix(subdirectory + "/"): value
                  for name, value in receipt["files"].items()
                  if name.startswith(subdirectory + "/")}
        if nested:
            receipts[f"{subdirectory}/source.json"] = {
                "platform_source_commit": receipt["platform_source_commit"],
                "files": nested,
            }
    return receipts


def synchronize(source: Path, commit: str, *, check: bool, root: Path = ROOT,
                harness: str = KIMI) -> None:
    contents, receipt = mirror(source, harness, commit)
    destination = root / CANDIDATES / harness
    receipts = receipts_for(receipt)
    if check:
        for name, body in contents.items():
            if (destination / name).read_bytes() != body:
                raise ValueError(f"{harness} recipe mirror differs from source: {name}")
        for name, value in receipts.items():
            if json.loads((destination / name).read_text()) != value:
                raise ValueError(f"{harness} source receipt differs from source: {name}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    for name, body in contents.items():
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(body)
    for name, value in receipts.items():
        (destination / name).write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True,
                        help="clean clone of the platform source repository")
    parser.add_argument("--harness", required=True,
                        help="directory name under candidates/, e.g. kimi-code")
    parser.add_argument("--commit", help="full source commit (default: config/candidates.json)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    commit = args.commit or roster_commit(args.harness)
    synchronize(args.source.resolve(), commit, check=args.check, harness=args.harness)


if __name__ == "__main__":
    main()
