#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
python3 -m py_compile scripts/inference.py scripts/parse_regex_scalpel.py scripts/scalpel_to_thinking_steps.py
