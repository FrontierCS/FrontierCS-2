#!/bin/sh
# Write /workspace/payload/solution.patch from the working tree's changes inside
# the editable surface, including new files under frontier_solution/.
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
# Record new files without staging their content, so the diff includes them.
git -C "$source_root" add --intent-to-add -- frontier_solution
git -C "$source_root" diff --no-ext-diff HEAD -- \
  frontier_solution \
  mujoco_playground/_src/locomotion/go1/joystick.py \
  mujoco_playground/_src/locomotion/go1/randomize.py \
  mujoco_playground/config/locomotion_params.py > "$temporary"
outside=$(git -C "$source_root" status --porcelain -- . \
  ':(exclude)frontier_solution' \
  ':(exclude)mujoco_playground/_src/locomotion/go1/joystick.py' \
  ':(exclude)mujoco_playground/_src/locomotion/go1/randomize.py' \
  ':(exclude)mujoco_playground/config/locomotion_params.py')
if [ -n "$outside" ]; then
  echo "left out of the patch (outside the editable surface):" >&2
  printf '%s\n' "$outside" >&2
fi
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
