#!/bin/sh
set -eu
CKKS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export GOMAXPROCS=2 GOMEMLIMIT=1GiB GOTOOLCHAIN=local GOENV=off GOWORK=off
export GOPATH="$CKKS_ROOT/.frontier-cache/path" GOMODCACHE="${CKKS_MODULE_CACHE:-/opt/frontier/go-mod}" GOCACHE="$CKKS_ROOT/.frontier-cache/build" GOTMPDIR="$CKKS_ROOT/.frontier-cache/tmp"
mkdir -p "$GOPATH" "$GOCACHE" "$GOTMPDIR"
export GOPROXY=off GOSUMDB=off
[ "$(go env GOVERSION)" = "go1.23.12" ]
cd "$CKKS_ROOT/estimator"
go mod verify
go build -p=1 ./...
