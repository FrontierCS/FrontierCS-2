#!/usr/bin/env python3
"""Create a source patch, then package that payload for the generic Candidate CLI."""
import io
from pathlib import Path
import subprocess
import sys
import tarfile


def package(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if repo == output or repo in output.parents:
        raise ValueError("archive must be outside the source repository")
    def git(*args):
        return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args])
    git("add", "-A")
    names = git("diff", "--cached", "--name-only", "-z", "HEAD").decode().split("\0")
    names = [n for n in names if n]
    if not names or len(names) > 20:
        raise ValueError("patch must change 1 through 20 files")
    for name in names:
        if name != "code/DBSCAN.C" and not name.startswith("code/frontier_solution/"):
            raise ValueError("change outside editable surface")
        if (repo / name).is_symlink():
            raise ValueError("source symlinks are forbidden")
    patch = git("diff", "--cached", "--binary", "HEAD", "--", "code/DBSCAN.C", "code/frontier_solution/")
    if not patch or len(patch) > 262144:
        raise ValueError("patch must contain 1 through 262144 bytes")
    if b"GIT binary patch" in patch or b"\0" in patch:
        raise ValueError("submit source changes only")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".patch").write_bytes(patch)
    with tarfile.open(output, "w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("payload/solution.patch")
        member.size, member.mode, member.mtime = len(patch), 0o644, 0
        archive.addfile(member, io.BytesIO(patch))
    if output.stat().st_size > 286720:
        raise ValueError("archive exceeds 286720 bytes")
    return output

if __name__ == "__main__":
    print(package(sys.argv[1], sys.argv[2]))
