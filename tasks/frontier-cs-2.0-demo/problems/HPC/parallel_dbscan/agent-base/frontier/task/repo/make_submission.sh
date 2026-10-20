#!/bin/sh
set -eu
exec /frontier/task/bin/make_submission.sh "$(dirname "$0")" "$@"
