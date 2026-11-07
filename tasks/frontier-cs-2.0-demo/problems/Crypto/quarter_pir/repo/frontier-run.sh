#!/bin/sh
# Run one complete local experiment exactly as the evaluator does, with fresh
# random seeds, and then the server-time replay.
#   frontier-run.sh [database entries, default 65536]
# Prints the conductor's result record and the median server time. After the
# 1000 measured queries the conductor issues its additional untimed queries;
# "probe_failure" in the record is empty when all of them were answered.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cache=${FRONTIER_BUILD_CACHE:-${TMPDIR:-/var/tmp}/quarter-pir-build}
size=${1:-65536}
port=${FRONTIER_PORT:-50651}
"$root/frontier-build.sh" >/dev/null
export GOMAXPROCS=4 GOGC=100
work=$(mktemp -d "$cache/run.XXXXXX")
cd "$work"
seed() { od -An -N7 -tu8 /dev/urandom | tr -d ' '; }
dbseed=$(seed)
printf '%s %s\n' "$size" "$dbseed" > config.txt
"$cache/server" -port "$port" -ignoreInit=false 2>server.log &
server=$!
# The conductor talks to the client and the reducer over their stdin and stdout,
# passed to it as descriptors 3-6.
mkfifo client.in client.out reducer.in reducer.out
"$cache/ec24" -size "$size" -thread 4 <client.in >client.out 2>client.log &
client=$!
"$cache/reducer" -size "$size" -thread 4 <reducer.in >reducer.out 2>reducer.log &
reducer=$!
trap 'kill "$server" "$client" "$reducer" 2>/dev/null || true' EXIT
cat > conductor.json <<JSON
{"upstream":"127.0.0.1:$port","client_stdin":3,"client_stdout":4,"reducer_stdin":5,"reducer_stdout":6,"size":$size,
 "db_seed":$dbseed,"coin_seed":$(seed),"query_seed":$(seed),"probe_seed":$(seed),"queries":1000,"threads":4,
 "transcript":"$work/transcript.json","result":"$work/result.json"}
JSON
"$cache/conductor" conductor.json 3>client.in 4<client.out 5>reducer.in 6<reducer.out 2>conductor.log
sed -e 's/,"upload_bytes":\[[^]]*\]//' -e 's/,"download_bytes":\[[^]]*\]//' result.json
printf '\n'
if grep -q '"valid":true' result.json && ! grep -q '"probe_failure":""' result.json; then
  printf 'additional untimed queries failed: this run is invalid\n'
fi
if grep -q '"valid":true' result.json; then
  FRONTIER_REPLAY=1 "$cache/server" "$size" "$dbseed" transcript.json replay.json measure 2>replay.log \
    && printf 'server replay passed: %s\n' "$work/replay.json" \
    || { printf 'server replay failed: ' ; tail -n 1 replay.log; }
else
  printf 'client log: %s\n' "$work/client.log"
fi
