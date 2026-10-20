#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
git add -A
git diff --cached --binary -- apps/hyper/HyperKCore-Efficient.C apps/hyper/frontier_solution/ > solution.patch
test -s solution.patch || { echo "No editable changes to submit" >&2; exit 1; }
echo "Wrote $PWD/solution.patch"
