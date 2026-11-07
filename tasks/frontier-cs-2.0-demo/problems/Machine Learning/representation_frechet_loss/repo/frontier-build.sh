#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
runtime_python=${FRONTIER_RUNTIME_PYTHON:-/opt/frontier/runtime-python}
if [[ ! -x "$runtime_python" ]]; then runtime_python=python3; fi
exec "$runtime_python" -m compileall -q frechet_distance models utils main_fd.py frontier_prepare_assets.py
