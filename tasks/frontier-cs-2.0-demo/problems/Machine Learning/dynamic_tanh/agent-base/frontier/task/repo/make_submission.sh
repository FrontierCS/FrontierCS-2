#!/usr/bin/env bash
set -euo pipefail
cd "${FRONTIER_REPO_ROOT:-/app}"
git add -A -- dynamic_tanh.py solver_extensions 2>/dev/null || git add -A -- dynamic_tanh.py
changed="$(git diff --cached --name-only)"
while IFS= read -r path; do
  case "$path" in dynamic_tanh.py|solver_extensions/*) ;; *) echo "refusing out-of-scope change: $path" >&2; exit 2;; esac
done <<< "$changed"
git diff --cached --binary --output=/app/solution.patch
echo /app/solution.patch
