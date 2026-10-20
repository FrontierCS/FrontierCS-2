#!/usr/bin/env bash
set -euo pipefail

repo="${FRONTIER_REPO:-/workspace/repo}"
if [[ ! -d "$repo" ]]; then repo="$(cd "$(dirname "$0")" && pwd)"; fi
out="${1:-$repo/sample-results.csv}"
cd "$repo"
RAYON_NUM_THREADS=1 cargo run --release --locked --bin benchmark \
  --manifest-path "$repo/frontier-build/Cargo.toml" -- \
  harness/data/dreamcoder-benchmarks/benches \
  --domain=physics --beam-size=64 --lps=1 --mode=babble --rounds=2 \
  --max-arity=2 --use-all=1 --lib-iter-limit=1 --output="$out"
