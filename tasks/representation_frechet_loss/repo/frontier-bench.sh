#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
runtime_python=${FRONTIER_RUNTIME_PYTHON:-/opt/frontier/runtime-python}
if [[ ! -x "$runtime_python" ]]; then runtime_python=python3; fi
if [[ "${1:-}" == "--help" ]]; then
    exec "$runtime_python" -c 'from main_fd import get_args_parser; get_args_parser().print_help()'
fi
asset_root=${FRONTIER_PUBLIC_ASSETS:-/workspace/fd-assets}
model_root=${FRONTIER_MODEL_ROOT:-/opt/frontier/models}
"$runtime_python" frontier_prepare_assets.py --model-root "$model_root" --output-root "$asset_root"
export HF_HOME="$asset_root/cache" HF_HUB_CACHE="$asset_root/cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$model_root/torch"
exec "$runtime_python" -m torch.distributed.run --standalone --local-addr=127.0.0.1 \
    --nproc-per-node=1 main_fd.py \
    --fid_stats_path "$asset_root/stats/guided_diffusion_stats.npz" \
    --fd_repr_stats_paths "$asset_root/stats/guided_diffusion_stats.npz" "$@"
