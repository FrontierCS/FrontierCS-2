#!/bin/sh
# Run tests/test_polygon_search_sandbox.py inside a polygon Verifier image (MariaDB, Z3 and the
# pinned packages), at the Verifier's 2 CPU / 8 GiB. The parts of this checkout the test reads
# are copied into the container's /frontier first (the image's interpreter does not import from
# the read-only bind mount).
#   tests/run_polygon_search_sandbox.sh IMAGE [-k PATTERN ...]
# e.g. tests/run_polygon_search_sandbox.sh <registry>/polygon/verifier:<tag> -k malformed
set -eu
image=$1
shift
repo=$(cd "$(dirname "$0")/.." && pwd)
exec docker run --rm --network none --cpus 2 --memory 8g --user 65532:65532 \
  --tmpfs /frontier:rw,exec,mode=1777 -e TMPDIR=/frontier/tmp -e HOME=/frontier \
  -v "$repo:/source:ro" --entrypoint /bin/sh "$image" -c '
    set -eu
    mkdir -p /frontier/tmp /frontier/repo/tasks/problems
    cd /source
    cp -a tests scripts config /frontier/repo/
    cp -a tasks/shared tasks/template tasks/*.json /frontier/repo/tasks/
    cp -a "tasks/problems/Programming Language" /frontier/repo/tasks/problems/
    cd /frontier/repo
    exec python3 -m unittest discover -v -s tests -p test_polygon_search_sandbox.py "$@"
  ' run "$@"
