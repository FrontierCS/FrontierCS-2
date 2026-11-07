#!/usr/bin/env bash
set -euo pipefail
cd /workspace/repo
exec python3 -m frontier_harness.run_debug \
  --repo . --m 16 --n 16 --k 24 \
  --active-ranks 1 --allocated-ranks 1 --compile-only
