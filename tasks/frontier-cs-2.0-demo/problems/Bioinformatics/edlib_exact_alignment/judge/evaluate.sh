#!/bin/sh
set -eu
exec python3 "$(dirname "$0")/evaluator.py" "$1"
