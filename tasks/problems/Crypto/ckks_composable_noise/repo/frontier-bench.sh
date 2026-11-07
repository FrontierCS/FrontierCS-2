#!/bin/sh
set -eu
CKKS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export GOMAXPROCS=1 GOMEMLIMIT=1GiB GOTOOLCHAIN=local GOENV=off GOWORK=off
export GOPATH="$CKKS_ROOT/.frontier-cache/path" GOMODCACHE="${CKKS_MODULE_CACHE:-/opt/frontier/go-mod}" GOCACHE="$CKKS_ROOT/.frontier-cache/build" GOTMPDIR="$CKKS_ROOT/.frontier-cache/tmp"
mkdir -p "$GOPATH" "$GOCACHE" "$GOTMPDIR"
export GOPROXY=off GOSUMDB=off
cd "$CKKS_ROOT/estimator/experiments/mul_relin_rescale"
exec go run -p=1 mul_relin_rescale.go
