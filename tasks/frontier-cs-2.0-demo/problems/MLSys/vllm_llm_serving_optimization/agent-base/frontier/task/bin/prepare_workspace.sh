#!/bin/sh
set -eu

destination=${1:-/workspace/repo}
source_root=${2:-/frontier/task/repo/vllm}
if [ -e "$destination" ]; then
  echo "destination already exists: $destination" >&2
  exit 2
fi
if [ ! -d "$source_root/.git" ]; then
  echo "the packaged vLLM source must contain its pinned Git history" >&2
  exit 2
fi
base=$(git -C "$source_root" rev-parse --verify HEAD)
if [ -n "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]; then
  echo "packaged vLLM source is not clean" >&2
  exit 2
fi
cp -a "$source_root" "$destination"
chmod -R u+rwX "$destination"
# Keep the package's base even if the Agent makes local commits while working.
printf '%s\n' "$base" > "$destination/.git/frontier-submission-base"
printf '%s\n' "$destination"
