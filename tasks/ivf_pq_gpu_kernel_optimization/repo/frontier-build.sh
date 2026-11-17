#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 -m compileall -q ivfpqlib
python3 -c 'import ivfpqlib; assert callable(ivfpqlib.ivf_pq_search)'
