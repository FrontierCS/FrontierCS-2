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
  echo "run /frontier/task/prepare_workspace.sh first" >&2
  exit 2
fi

install -d -m 0700 /workspace/payload
temporary=/workspace/payload/.solution.patch.tmp
git -C "$source_root" diff --no-ext-diff HEAD -- > "$temporary"
if [ ! -s "$temporary" ]; then
  rm -f "$temporary"
  echo "submission patch is empty" >&2
  exit 2
fi
size=$(wc -c < "$temporary")
if [ "$size" -gt 131072 ]; then
  rm -f "$temporary"
  echo "submission patch exceeds 131072 bytes" >&2
  exit 2
fi
mv "$temporary" /workspace/payload/solution.patch
chmod 0600 /workspace/payload/solution.patch
printf '%s\n' /workspace/payload/solution.patch
