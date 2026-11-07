#!/bin/sh
set -eu
python3 -m compileall -q dbscanlib
python3 -c 'import dbscanlib; assert callable(dbscanlib.dbscan)'
