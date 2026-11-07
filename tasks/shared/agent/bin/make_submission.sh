#!/bin/sh
set -eu
# The shared base's own Python, whatever Python the task image adds.
exec /usr/bin/python3 /frontier/task/bin/fcs2_submission.py "$@"
