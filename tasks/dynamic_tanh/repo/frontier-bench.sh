#!/usr/bin/env bash
set -euo pipefail
cd "${FRONTIER_REPO_ROOT:-/workspace/repo}"
python3 -c 'import torch; assert torch.cuda.is_available(), "Verifier NVIDIA GPU is required"'
exec python3 main.py "$@"
