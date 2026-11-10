#!/bin/sh
# Checks that the package imports and exposes the interface (run it on a GPU runner).
set -eu
python3 -m compileall -q kernel
python3 -c 'import kernel; assert callable(kernel.pairwise_sq_dists)'
