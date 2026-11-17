#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
export TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
jupyter nbconvert --to notebook --execute plotsforpolar.ipynb --output /dev/null --ExecutePreprocessor.timeout=120
