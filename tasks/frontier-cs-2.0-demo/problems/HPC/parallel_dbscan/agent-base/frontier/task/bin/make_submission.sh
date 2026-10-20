#!/bin/sh
set -eu
exec python3 /frontier/task/bin/package_submission.py "${1:-/workspace/parallel_dbscan}" "${2:-/workspace/submission.tar}"
