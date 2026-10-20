#!/bin/sh
set -eu
exec python3 /frontier/task/bin/fcs2_submission.py "$@"
