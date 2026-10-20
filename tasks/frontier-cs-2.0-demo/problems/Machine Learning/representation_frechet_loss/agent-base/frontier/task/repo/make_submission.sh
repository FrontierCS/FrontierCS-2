#!/usr/bin/env bash
set -euo pipefail
cd /app
git add -A -- frechet_distance/queue.py frechet_distance/frontier_solution
git diff --cached --binary -- frechet_distance/queue.py frechet_distance/frontier_solution > /app/solution.patch
test -s /app/solution.patch
echo /app/solution.patch

