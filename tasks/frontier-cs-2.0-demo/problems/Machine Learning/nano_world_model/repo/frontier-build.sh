#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/frontier_runtime:$PWD/src"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/nanowm-bytecode"
python3 -m compileall -q src
python3 -c 'from experiments import build_experiment; from diffusion.df_sample import dfot_sample; print("native entrypoint imports OK")'
