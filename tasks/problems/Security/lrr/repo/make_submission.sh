#!/usr/bin/env bash
set -euo pipefail
repo="${REPO_ROOT:-/app}"
out="${1:-/app/solution.patch}"
git -C "$repo" add -A
git -C "$repo" diff --cached --binary > "$out"
test -s "$out" || { echo "submission is empty" >&2; exit 2; }
echo "Wrote $out ($(wc -c < "$out") bytes)"
