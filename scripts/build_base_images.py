#!/usr/bin/env python3
"""Build, and optionally publish, the shared task images (#757).

  ./python scripts/build_base_images.py                              # build agent + verifier
  ./python scripts/build_base_images.py --roles gpu-tools --push
  ./python scripts/build_base_images.py --print-tags                 # role -> derived tag

tasks/shared/bases.json holds the one current set: agent and verifier are bases a task builds
FROM (frontier-<role>-base, recipe tasks/shared/images/<role>-base.Dockerfile); gpu-tools is a
files-only image GPU task images COPY --from (frontier-gpu-tools, recipe
tasks/shared/images/gpu-tools.Dockerfile). Context is tasks/shared. With --push, each built role
is pushed as docker.io/<namespace>/<name>:<tag> and becomes the current entry for that role in
bases.json; task Dockerfiles take the bases by build argument (ARG AGENT_BASE / VERIFIER_BASE /
GPU_TOOLS) and build_task_images injects the bases.json references, so a base change moves all
tasks at once.

Tags are derived, never picked: a role's tag is `src-<12 hex>`, the git tree hash of exactly the
sources its image is built from (SOURCES, paths under tasks/shared, hashed from the committed
work tree; uncommitted changes to them refuse the build). Two branches therefore cannot mint the
same tag for different content, and the same sources always name the same tag. The digest stays
the pin; pushing a tag that already names a different digest fails (a non-reproducible rebuild).
Hand-numbered tags (v1..v10) remain in the registry as history and are never minted again.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import source_tree
from task_manifest import SHARED

ROLES = ("agent", "verifier", "gpu-tools")
IMAGE_NAMES = {"agent": "frontier-agent-base", "verifier": "frontier-verifier-base",
               "gpu-tools": "frontier-gpu-tools"}
RECIPES = {"agent": "agent-base.Dockerfile", "verifier": "verifier-base.Dockerfile",
           "gpu-tools": "gpu-tools.Dockerfile"}
BASES = SHARED / "bases.json"
# Everything each role's image is built from, relative to tasks/shared (the build context). The
# recipe's COPY sources must lie inside these (tests/test_build_base_images.py checks it).
SOURCES = {
    "agent": ("images/agent-base.Dockerfile", "agent/bin", "lib/edit_scope.py"),
    "verifier": ("images/verifier-base.Dockerfile", "verifier", "lib/edit_scope.py",
                 "lib/isolate.py", "lib/install_frontier_pth.py"),
    "gpu-tools": ("images/gpu-tools.Dockerfile", "gpu-tools", "verifier-gpu",
                  "lib/isolate.py", "lib/install_frontier_pth.py"),
}
TAG_HEX = 12


def source_files(role: str) -> list[str]:
    """The role's source files, relative to tasks/shared: git's tracked files under its SOURCES in
    a checkout (ignored files such as a __pycache__ are not sources), every file under them in an
    exported tree."""
    listed = subprocess.run(["git", "-C", str(SHARED), "ls-files", "-z", "--", *SOURCES[role]],
                            capture_output=True)
    if source_tree.in_work_tree(SHARED):
        if listed.returncode != 0:
            raise SystemExit(f"git ls-files failed: {listed.stderr.decode().strip()}")
        return sorted(os.fsdecode(path) for path in listed.stdout.split(b"\0") if path)
    files = []
    for source in SOURCES[role]:
        path = SHARED / source
        found = [path] if not path.is_dir() else [p for p in path.rglob("*") if not p.is_dir()]
        files += [p.relative_to(SHARED).as_posix() for p in found]
    return sorted(files)


def _git_tree(entries: dict) -> bytes:
    """The git object id of a tree: entries maps a name to bytes (a blob) or a nested dict."""
    body = b""
    def order(item):  # git orders a tree entry by its name, with "/" after a tree's name
        name, value = item
        return name + ("/" if isinstance(value, dict) else "")
    for name, value in sorted(entries.items(), key=order):
        if isinstance(value, dict):
            mode, oid = b"40000", _git_tree(value)
        else:
            mode, data = value
            oid = hashlib.sha1(b"blob %d\0" % len(data) + data).digest()
        body += mode + b" " + name.encode() + b"\0" + oid
    return hashlib.sha1(b"tree %d\0" % len(body) + body).digest()


def source_tag(role: str) -> str:
    """`src-<12 hex>`: the git tree hash of a tree holding only the role's source files at their
    repository paths (tasks/shared/...), as they are in the work tree, the id `git write-tree`
    gives after adding exactly those files. Computed without a git index, so an exported tree
    names the same tag as its checkout."""
    root: dict = {}
    prefix = SHARED.relative_to(SHARED.parents[1]).as_posix()  # tasks/shared
    for relative in source_files(role):
        path = SHARED / relative
        if path.is_symlink():
            value = (b"120000", os.readlink(path).encode())
        else:
            mode = b"100755" if path.stat().st_mode & 0o100 else b"100644"
            value = (mode, path.read_bytes())
        *directories, name = f"{prefix}/{relative}".split("/")
        node = root
        for directory in directories:
            node = node.setdefault(directory, {})
        node[name] = value
    return f"src-{_git_tree(root).hex()[:TAG_HEX]}"


def uncommitted_sources(role: str) -> list[str]:
    """Source files that differ from HEAD (staged, unstaged or untracked)."""
    status = subprocess.run(["git", "-C", str(SHARED), "status", "--porcelain", "-z",
                             "--untracked-files=all", "--",
                             *[str(SHARED / path) for path in SOURCES[role]]],
                            check=True, capture_output=True, text=True).stdout
    return [entry[3:] for entry in status.split("\0") if entry]


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command), file=sys.stderr)
    return subprocess.run(command, check=True, **kwargs)


def remote_digest(reference: str) -> str | None:
    result = subprocess.run(["crane", "digest", reference], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_context(role: str, scratch: Path) -> Path:
    """The build context: only the role's git-tracked SOURCES (source_tree; an untracked or
    ignored file under tasks/shared, such as a __pycache__, fails instead of reaching the
    image)."""
    context = scratch / role
    source_tree.stage(SHARED, context, paths=list(SOURCES[role]))
    return context


def gpu_tools_context(scratch: Path) -> Path:
    """The gpu-tools sources plus .root/: the pinned bootstrap layer's one file (hash
    checked), the GPU adapter, the isolate launcher and its import-path installer."""
    pin = json.loads((SHARED / "gpu-tools/bootstrap.json").read_text())
    context = source_context("gpu-tools", scratch)
    image = pin["source_image"].split("@")[0]
    blob = run(["crane", "blob", f"{image}@{pin['source_layer']}"], capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as layer:
        body = layer.extractfile(pin["path"]).read()
    if hashlib.sha256(body).hexdigest() != pin["sha256"]:
        raise SystemExit("GPU SSH bootstrap hash differs from gpu-tools/bootstrap.json")
    files = {pin["path"]: (body, 0o555),
             "opt/frontier/shared-gpu/fcs2_verifier_gpu.py":
                 ((context / "verifier-gpu/fcs2_verifier_gpu.py").read_bytes(), 0o444),
             "opt/frontier/shared-gpu/isolate.py":
                 ((context / "lib/isolate.py").read_bytes(), 0o444),
             "opt/frontier/shared-gpu/install_frontier_pth.py":
                 ((context / "lib/install_frontier_pth.py").read_bytes(), 0o444)}
    for relative, (content, mode) in files.items():
        target = context / ".root" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(mode)
    for path in sorted((context / ".root").rglob("*"), reverse=True):  # files, then parents
        os.utime(path, (0, 0))  # fixed mtimes: the same inputs give the same image
        if path.is_dir():
            path.chmod(0o555)
    (context / ".root").chmod(0o755)
    return context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="joyemang33")
    parser.add_argument("--print-tags", action="store_true",
                        help="print each role's derived tag and exit")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--roles", default="agent,verifier",
                        help=f"comma-separated subset of {','.join(ROLES)}")
    options = parser.parse_args()
    roles = options.roles.split(",")
    if not roles or any(role not in ROLES for role in roles):
        parser.error(f"--roles must name some of {', '.join(ROLES)}")
    if options.print_tags:
        for role in roles:
            print(f"{role} {source_tag(role)}")
        return 0
    for role in roles:
        if uncommitted_sources(role):
            raise SystemExit(f"{role}: commit its sources first (the tag names committed "
                             f"content): {', '.join(uncommitted_sources(role))}")
    recorded = json.loads(BASES.read_text())
    references = {}
    scratch = Path(tempfile.mkdtemp(prefix="frontier-bases-"))
    for role in roles:
        repository = f"docker.io/{options.namespace}/{IMAGE_NAMES[role]}"
        tag = f"{repository}:{source_tag(role)}"
        context = (gpu_tools_context(scratch) if role == "gpu-tools"
                   else source_context(role, scratch))
        run(["docker", "build", "--provenance=false", "--platform", "linux/amd64",
             "-f", str(context / "images" / RECIPES[role]), "--tag", tag, str(context)])
        if not options.push:
            continue
        existing = remote_digest(tag)
        run(["docker", "push", tag])
        digest = remote_digest(tag)
        if digest is None:
            raise SystemExit(f"cannot read back {tag}")
        if existing and existing != digest:
            raise SystemExit(f"{tag} already named {existing}: the same sources built a different "
                             "image (a non-reproducible recipe); fix the recipe")
        reference = f"{tag}@{digest}"
        references[role] = reference
        print(f"{role}: {reference}")
    for directory in scratch.rglob("*"):  # the staged .root tree is read-only
        if directory.is_dir() and not directory.is_symlink():
            directory.chmod(0o755)
    shutil.rmtree(scratch, ignore_errors=True)
    if options.push:
        recorded.update(references)
        BASES.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"recorded {BASES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
