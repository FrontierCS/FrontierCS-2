#!/bin/sh
set -eu

source_root=${1:-/workspace/repo}
output=${2:-/workspace/payload/solution.patch}
if [ ! -f "$source_root/.git/frontier-submission-base" ]; then
  echo "run /frontier/task/bin/prepare_workspace.sh first" >&2
  exit 2
fi
base=$(cat "$source_root/.git/frontier-submission-base")
git -C "$source_root" rev-parse --verify "$base^{commit}" >/dev/null
install -d -m 0700 "$(dirname "$output")"
temporary=$(mktemp "${output}.tmp.XXXXXX")
trap 'rm -f "$temporary"' EXIT HUP INT TERM
git -C "$source_root" add --all
git -C "$source_root" diff --cached --binary --no-ext-diff "$base" -- > "$temporary"
size=$(wc -c < "$temporary")
if [ "$size" -gt 1500000 ]; then
  echo "submission patch exceeds 1500000 bytes" >&2
  exit 2
fi
# An empty patch intentionally submits unmodified vLLM, as in the original task.
chmod 0600 "$temporary"
mv "$temporary" "$output"
trap - EXIT HUP INT TERM
printf '%s\n' "$output"
