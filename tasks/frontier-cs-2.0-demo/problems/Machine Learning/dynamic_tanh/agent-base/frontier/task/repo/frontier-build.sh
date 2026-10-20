#!/usr/bin/env bash
set -euo pipefail
cd "${FRONTIER_REPO_ROOT:-/workspace/repo}"
python3 -m py_compile main.py engine.py datasets.py optim_factory.py utils.py dynamic_tanh.py
python3 -c 'import dynamic_tanh, main'
