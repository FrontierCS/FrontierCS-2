#!/usr/bin/env bash
set -euo pipefail
cd "${FRONTIER_REPO_ROOT:-/workspace/repo}"
python -m pip install -e ./verl --use-pep517 --no-deps --no-build-isolation --disable-pip-version-check
python -m pip install -e . --use-pep517 --no-deps --no-build-isolation --disable-pip-version-check
python -m compileall -q ragen/llm_agent/ctx_manager.py
