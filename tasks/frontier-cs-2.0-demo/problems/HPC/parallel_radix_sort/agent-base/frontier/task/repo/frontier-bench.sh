#!/bin/sh
set -eu
export CILK_NWORKERS=4
exec taskset -c 0-3 ./radixSort "$@"
