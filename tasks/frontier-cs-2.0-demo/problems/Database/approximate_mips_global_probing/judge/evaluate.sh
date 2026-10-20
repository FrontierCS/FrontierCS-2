#!/usr/bin/env bash
set -euo pipefail
TASK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOLUTION="${1:-$TASK_DIR/reference.patch}"
exec python3 "$TASK_DIR/evaluator.py" "$SOLUTION"
