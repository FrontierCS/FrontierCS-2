#!/bin/sh
set -eu
cd "$(dirname "$0")"
git add -A -- src/frontier_solution
# Stage new files as well as modifications; refuse changes outside the solution directory.
git diff --name-only HEAD -- . ':!src/frontier_solution/' | while IFS= read -r path; do
  [ -z "$path" ] || { echo "Change outside allowed surface: $path" >&2; exit 1; }
done
git diff --cached --binary > solution.patch
