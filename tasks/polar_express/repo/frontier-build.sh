#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m py_compile polar_express.py
python -c 'import polar_express'
