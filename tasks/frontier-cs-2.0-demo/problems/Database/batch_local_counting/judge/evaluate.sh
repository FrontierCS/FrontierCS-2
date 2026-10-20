#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOLUTION=${1:-"$SCRIPT_DIR/reference.patch"}
exec python3 "$SCRIPT_DIR/evaluator.py" "$SOLUTION"
