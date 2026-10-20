#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
git add -A -- src/block_gemm/KAMI/1d/fp64
git diff --cached --binary > solution.patch
test -s solution.patch || { echo "No editable changes to submit" >&2; exit 1; }
echo "Wrote $(pwd)/solution.patch"
