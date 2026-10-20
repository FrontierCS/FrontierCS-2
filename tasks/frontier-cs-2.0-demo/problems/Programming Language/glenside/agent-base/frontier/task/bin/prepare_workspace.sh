#!/bin/sh
# Copy the read-only source tree into the workspace and make it a git repository,
# so `git diff` can produce the submission patch.
set -eu

destination=${1:-/workspace/repo}
case "$destination" in
  /workspace/*) ;;
  *) echo "destination must be below /workspace" >&2; exit 2 ;;
esac
if [ -e "$destination" ]; then
  echo "destination already exists: $destination" >&2
  exit 2
fi

if [ ! -d /workspace/.cargo ]; then
  cp -a /opt/cargo /workspace/.cargo
  chmod -R u+rwX /workspace/.cargo
fi
cp -a /frontier/task/repo "$destination"
chmod -R u+rwX "$destination"
git -C "$destination" init --quiet
git -C "$destination" config user.name frontier-task-baseline
git -C "$destination" config user.email frontier-task-baseline@invalid
git -C "$destination" add --all
git -C "$destination" commit --quiet --message baseline
install -d -m 0700 /workspace/payload
printf '%s\n' "$destination"
