#!/bin/sh
set -eu
./frontier-cxx -I RadixSort -std=c++14 RadixSort/example.cpp -DCILKP -DBITS_HACK -O3 -flto -march=native -fcilkplus -lcilkrts -o radixSort
