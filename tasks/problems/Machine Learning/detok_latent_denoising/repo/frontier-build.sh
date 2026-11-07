#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python -m compileall -q main_reconstruction.py models utils diffusion transport
