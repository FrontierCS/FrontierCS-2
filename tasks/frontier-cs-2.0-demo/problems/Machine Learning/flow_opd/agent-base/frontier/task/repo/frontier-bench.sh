#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FRONTIER_MODEL_ROOT="${FRONTIER_MODEL_ROOT:-/opt/frontier/models}"
export PATH="/opt/conda/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
if [[ -d /usr/local/cuda/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -x /opt/frontier/runtime-python ]]; then
    FLOW_RUNTIME_PYTHON=/opt/frontier/runtime-python
else
    FLOW_RUNTIME_PYTHON=python3
fi
exec "$FLOW_RUNTIME_PYTHON" frontier_runtime.py "$@"
