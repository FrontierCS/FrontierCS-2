#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
solution="${1:?usage: evaluate.sh SOLUTION.patch}"
exec python3 "$script_dir/evaluator.py" "$solution"
