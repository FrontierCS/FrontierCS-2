#!/usr/bin/env bash
set -euo pipefail

repo="${FRONTIER_REPO_ROOT:-/workspace/repo}"
if [[ ! -f "$repo/pyproject.toml" ]]; then
  repo=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

cache="${FRONTIER_BUILD_CACHE:-/workspace/.frontier-build-cache}"
mkdir -p "$cache/pycache"
export PYTHONPYCACHEPREFIX="$cache/pycache"

cd "$repo"
python3 -m py_compile wan_t2v_inference.py svg/kmeans_utils.py \
  svg/models/wan/attention.py svg/models/wan/inference.py \
  svg/models/wan/custom_models.py hyvideo_t2v_inference.py \
  svg/models/hyvideo/attention.py svg/models/hyvideo/inference.py \
  svg/models/hyvideo/custom_models.py flash_attn/flash_attn_interface.py
cd svg/kernels
bash setup.sh
test -n "$(find build -maxdepth 1 -name '_kernels*.so' -print -quit)"
