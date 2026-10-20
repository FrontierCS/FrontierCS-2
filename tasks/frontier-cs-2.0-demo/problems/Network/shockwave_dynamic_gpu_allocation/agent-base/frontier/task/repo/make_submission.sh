#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
git add -A -- . ':!solution.patch'
bad="$(git diff --cached --name-only | awk '
  $0 == "scheduler/shockwave.py" {next}
  $0 == "scheduler/shockwave_helper.py" {next}
  index($0, "scheduler/solver_extensions/") == 1 {next}
  {print}
')"
if [[ -n "$bad" ]]; then
  echo "Changes outside the editable surface:" >&2
  echo "$bad" >&2
  exit 1
fi
git diff --cached --binary > solution.patch
echo "Wrote /app/solution.patch"
