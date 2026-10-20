#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m py_compile experiments/run.py experiments/lejepa_id/*.py
