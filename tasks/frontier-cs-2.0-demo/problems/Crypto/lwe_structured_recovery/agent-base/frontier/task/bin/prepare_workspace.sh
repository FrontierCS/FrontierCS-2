#!/bin/sh
set -eu
if [ ! -e /workspace/repo ]; then
    cp -R /frontier/task/repo /workspace/repo
    chmod -R u+rwX /workspace/repo
fi
test -d /workspace/repo
mkdir -p /workspace/payload
if [ ! -e /workspace/payload/solution.json ]; then
    cp /frontier/task/starter/payload/solution.json /workspace/payload/solution.json
fi
printf '%s\n' /workspace/repo
