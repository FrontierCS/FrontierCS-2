#!/usr/bin/env bash
set -euo pipefail

repo="${FRONTIER_REPO:-/workspace/repo}"
if [[ ! -d "$repo" ]]; then repo="$(cd "$(dirname "$0")" && pwd)"; fi
cd "$repo"
cargo build --release --locked --offline --bin benchmark \
  --manifest-path "$repo/frontier-build/Cargo.toml"
