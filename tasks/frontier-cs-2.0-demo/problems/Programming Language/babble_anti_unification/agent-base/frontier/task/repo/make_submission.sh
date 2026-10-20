#!/bin/sh
# Compatibility entrypoint; the native helper prepares a payload, not a tar.
set -eu
exec /frontier/task/bin/make_submission.sh "${BABBLE_REPO:-/workspace/repo}"
