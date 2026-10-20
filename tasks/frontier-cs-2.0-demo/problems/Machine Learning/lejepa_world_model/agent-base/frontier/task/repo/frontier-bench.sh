#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo/experiments 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")/experiments"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python3 run.py "$@"
