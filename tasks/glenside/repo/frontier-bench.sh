#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD=${1:-$SCRIPT_DIR/.frontier-build}
python3 "$SCRIPT_DIR/frontier-build.py" --output "$BUILD"
cd "$BUILD/code"
export CARGO_TARGET_DIR="$BUILD/target"
export CARGO_PROFILE_DEV_OPT_LEVEL=1 CARGO_PROFILE_DEV_DEBUG=0 CARGO_BUILD_JOBS=2
exec cargo test --locked --offline --no-default-features --lib language:: -- --test-threads=1
