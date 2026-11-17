#!/usr/bin/env bash
set -euo pipefail
repo_root="${FRONTIER_REPO_ROOT:-/workspace/repo}"
if [[ ! -d "$repo_root/scheduler" ]]; then
  repo_root="$(cd "$(dirname "$0")" && pwd)"
fi
cd "$repo_root/scheduler"
python scripts/drivers/simulate_scheduler_with_trace.py "$@"
