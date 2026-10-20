#!/bin/sh
set -eu
python3 -m compileall -q kmeanslib
python3 -c 'import kmeanslib; assert callable(kmeanslib.step)'
