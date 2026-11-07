#!/bin/sh
set -eu
python3 -m compileall -q knnlib
python3 -c 'import knnlib; assert callable(knnlib.knn)'
