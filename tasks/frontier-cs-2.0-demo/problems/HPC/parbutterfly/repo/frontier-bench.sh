#!/usr/bin/env bash
set -euo pipefail
repo="${1:-/workspace/repo}"
graph="${2:?usage: frontier-bench.sh REPO GRAPH}"
OMP_NUM_THREADS=8 "$repo/apps/hyper/BipartiteButterflies" -countType BATCHS -rankType ACOCORE -per TOTAL -r 0 -m 2000000 -a 2000000 "$graph"
