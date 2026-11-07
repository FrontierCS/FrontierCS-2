#!/usr/bin/env bash
set -euo pipefail

# Usage: frontier-bench.sh [wan|hunyuan]   (default: wan)
model="${1:-wan}"
case "$model" in
  wan) entrypoint=scripts/wan/wan_t2v_720p_ear.sh ;;
  hunyuan) entrypoint=scripts/hyvideo/hyvideo_t2v_720p_ear.sh ;;
  *) echo "usage: frontier-bench.sh [wan|hunyuan]" >&2; exit 2 ;;
esac

repo="${FRONTIER_REPO_ROOT:-/workspace/repo}"
if [[ ! -f "$repo/pyproject.toml" ]]; then
  repo=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

cd "$repo"
export PATH="$repo/.frontier:$PATH"
export PYTHONPATH="$repo"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec bash "$entrypoint"
