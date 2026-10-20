#!/bin/sh
# Write /workspace/payload/solution.patch from the working tree's diff.
#
# Textual diffs only: the evaluator rejects a patch containing binary hunks, and
# a --binary diff that passes here and fails there costs a train call to learn.
set -eu

source_root=${1:-/workspace/repo}
case "$source_root" in
  /workspace/*) ;;
  *) echo "source root must be below /workspace" >&2; exit 2 ;;
esac
if [ ! -d "$source_root/.git" ]; then
  echo "run /frontier/task/bin/prepare_workspace.sh first" >&2
  exit 2
fi

install -d -m 0700 /workspace/payload
temporary=/workspace/payload/.solution.patch.tmp
# Stage new extension modules as well as edits to existing source files.
git -C "$source_root" add --all -- .
python3 - "$source_root" <<'CHECK_PATHS'
import os
import subprocess
import sys

paths = subprocess.check_output([
    "git", "-C", sys.argv[1], "diff", "--cached", "--name-only", "-z", "HEAD", "--",
]).split(b"\0")
changed = [os.fsdecode(path) for path in paths if path]
allowed_prefix = "compaction/"
for path in changed:
    if not path.startswith(allowed_prefix):
        raise SystemExit("change outside editable surface: " + path)
if len(changed) > 40:
    raise SystemExit("submission changes more than 40 files")
CHECK_PATHS
git -C "$source_root" diff --cached --no-ext-diff --no-textconv HEAD -- > "$temporary"
if [ ! -s "$temporary" ]; then
  rm -f "$temporary"
  echo "submission patch is empty" >&2
  exit 2
fi
size=$(wc -c < "$temporary")
if [ "$size" -gt 262144 ]; then
  rm -f "$temporary"
  echo "submission patch exceeds 262144 bytes" >&2
  exit 2
fi
mv "$temporary" /workspace/payload/solution.patch
chmod 0600 /workspace/payload/solution.patch
printf '%s\n' /workspace/payload/solution.patch
