#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"
git add -A -- . ':!solution.patch'
changed=$(git diff --cached --name-only)
test -n "$changed" || { echo 'Empty patches are invalid.' >&2; exit 1; }
outside=$(printf '%s\n' "$changed" | grep -v '^quarter_pir/code/pir/' || true)
test -z "$outside" || {
  echo 'Only files under quarter_pir/code/pir/ may change:' >&2
  printf '%s\n' "$outside" >&2
  exit 1
}
git diff --cached --binary --no-ext-diff > "$root/solution.patch"
