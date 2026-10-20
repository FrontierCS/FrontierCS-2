#!/bin/sh
set -eu
cd /app
git add -A -- . ':!solution.patch' ':!.frontier-build' ':!target'
git diff --cached --name-only -z | python3 -c '
import sys
allowed = "code/src/language/"
paths = sys.stdin.buffer.read().decode().split("\0")
bad = [path for path in paths if path and path != allowed + "rewrites.rs" and not path.startswith(allowed + "solver_extensions/")]
if bad:
    sys.exit("Changes outside the editable surface: " + ", ".join(bad))
'
git diff --cached --binary > /app/solution.patch
