#!/usr/bin/env bash
set -euo pipefail
cd "${1:-/workspace/repo}/apps/hyper"
make -j8 PCC=g++ 'PCFLAGS=-std=c++14 -fopenmp -march=native -O3 -DOPENMP -DBYTERLE -include cstdint'
