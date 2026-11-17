#!/bin/sh
set -eu
cd "$(dirname "$0")"
git add -A
# Inspect every staged pathname, including newly created utility sources.
git diff --cached --name-only -z | python3 -c '
import sys
names=sys.stdin.buffer.read().split(b"\0")
for raw in names:
 if not raw: continue
 p=raw.decode("utf-8")
 fixed=p in ("estimator/stats.go","estimator/utils.go","estimator/go.mod","estimator/go.sum") or p.startswith("estimator/experiments/")
 if not p.startswith("estimator/") or fixed or not p.endswith(".go") or p.endswith("_test.go"):
  sys.exit("Staged change outside editable surface: "+p)
'
git diff --cached --binary --no-ext-diff > solution.patch
test -s solution.patch || { echo "Empty patches are not valid" >&2; exit 1; }
