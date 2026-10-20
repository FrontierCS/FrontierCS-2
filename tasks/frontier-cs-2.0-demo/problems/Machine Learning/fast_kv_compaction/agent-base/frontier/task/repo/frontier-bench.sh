#!/usr/bin/env bash
# Public development check: compact and answer public QuALITY articles with
# compaction/solver_entry.py. Usage: frontier-bench.sh [article_id ...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec /opt/frontier/runtime-python frontier_bench.py "$@"
