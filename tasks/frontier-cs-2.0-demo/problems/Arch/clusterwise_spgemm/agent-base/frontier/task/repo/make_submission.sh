#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
paths=(CSR_FlengthCluster.h BIN_FlengthCluster.h hash_mult_flengthcluster.h hash_mult.h cluster_utility.h CSR_VlengthCluster.h BIN.h)
[[ -d frontier_solution ]] && paths+=(frontier_solution)
git add -A -- "${paths[@]}"
git diff --cached --binary -- "${paths[@]}" > solution.patch
test -s solution.patch || { echo 'empty patches are not valid' >&2; exit 1; }
echo "$PWD/solution.patch"
