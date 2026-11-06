#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cache=${FRONTIER_BUILD_CACHE:-${TMPDIR:-/var/tmp}/quarter-pir-build}
mkdir -p "$cache"
export GOCACHE="$cache/go-build"
export GOMODCACHE="${GOMODCACHE:-$cache/go-mod}"
export CGO_ENABLED=0 GOMAXPROCS=4 GOGC=100
cd "$root/quarter_pir/code"
go mod download
for target in server ec24 reducer conductor policy; do
  go build -mod=readonly -buildvcs=false -trimpath -o "$cache/$target" "./$target"
done
# Source policy self-check against the pristine tree when it is available.
pristine=${FRONTIER_PRISTINE_CODE:-/frontier/task/repo/quarter_pir/code}
if [ -d "$pristine/pir" ]; then
  "$cache/policy" "$pristine" "$root/quarter_pir/code" >/dev/null
fi
printf 'Built server, ec24 (client driver), reducer, conductor and policy in %s\n' "$cache"
