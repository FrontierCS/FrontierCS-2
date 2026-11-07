#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="/opt/conda/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
if [[ -x /opt/frontier/runtime-python ]]; then
    FLOW_RUNTIME_PYTHON=/opt/frontier/runtime-python
else
    FLOW_RUNTIME_PYTHON=python3
fi
"$FLOW_RUNTIME_PYTHON" - <<'PY'
import ast, importlib
from pathlib import Path
for name in ['torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'peft', 'paddleocr', 'Levenshtein', 'ml_collections']:
    importlib.import_module(name)
for path in Path('.').rglob('*.py'):
    ast.parse(path.read_text())
print('Offline imports and syntax checked. Model execution uses frontier-bench.sh.')
PY
