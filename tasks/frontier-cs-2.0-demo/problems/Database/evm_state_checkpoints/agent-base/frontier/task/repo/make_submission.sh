#!/usr/bin/env bash
set -euo pipefail
out="${1:-/app/solution.patch}"
git -C /app diff --binary -- lib/evmcow/execution_state.hpp > "$out"
test -s "$out" || { echo "No editable changes found" >&2; exit 2; }
echo "Wrote $out ($(wc -c < "$out") bytes)"
