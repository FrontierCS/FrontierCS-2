#!/bin/sh
# Compatibility entrypoint; the native helper creates the patch payload.
set -eu
exec /frontier/task/bin/make_submission.sh "${1:-/workspace/repo}"
