#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
git add -A -- ragen/llm_agent/ctx_manager.py
if [[ -d ragen/llm_agent/solver_extensions ]]; then
  git add -A -- ragen/llm_agent/solver_extensions
fi
git diff --cached --binary -- ragen/llm_agent/ctx_manager.py ragen/llm_agent/solver_extensions > solution.patch
test -s solution.patch || { echo "No editable changes to submit" >&2; exit 1; }
echo "Wrote $PWD/solution.patch"
