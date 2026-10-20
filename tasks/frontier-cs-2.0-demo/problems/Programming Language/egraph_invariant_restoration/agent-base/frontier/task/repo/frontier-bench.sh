#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
export CARGO_NET_OFFLINE=true RAYON_NUM_THREADS=1
export RUSTUP_TOOLCHAIN=1.94.1-x86_64-unknown-linux-gnu
cargo test --release --locked --offline --test simple -- --nocapture
