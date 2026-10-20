#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
git add -A
mapfile -t changed < <(git diff --cached --name-only)
for path in "${changed[@]}"; do
  case "$path" in
    src/egraph.rs|src/tutorials/_01_background.rs|src/tutorials/mod.rs|src/solver_extensions/*) ;;
    *) echo "Change outside editable surface: $path" >&2; exit 2 ;;
  esac
done
git diff --cached --binary > solution.patch
echo "Wrote /app/solution.patch"
