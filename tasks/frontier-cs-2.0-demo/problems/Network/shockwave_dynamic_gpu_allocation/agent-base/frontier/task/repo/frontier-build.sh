#!/usr/bin/env bash
set -euo pipefail
repo_root="${FRONTIER_REPO_ROOT:-/workspace/repo}"
if [[ ! -d "$repo_root/scheduler" ]]; then
  repo_root="$(cd "$(dirname "$0")" && pwd)"
fi
cd "$repo_root/scheduler"
python -m grpc_tools.protoc -Iruntime/protobuf \
  --python_out=runtime/rpc_stubs \
  --grpc_python_out=runtime/rpc_stubs \
  runtime/protobuf/iterator_to_scheduler.proto \
  runtime/protobuf/scheduler_to_worker.proto \
  runtime/protobuf/worker_to_scheduler.proto \
  runtime/protobuf/common.proto \
  runtime/protobuf/enums.proto
