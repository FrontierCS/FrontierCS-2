#!/usr/bin/env bash
set -euo pipefail

cd /app
allowed='^(lib/treap/scheduler\.hpp|lib/treap/search_insert\.hpp|lib/treap/search_delete\.hpp)$'
bad=$(git diff --name-only -- . ':!solution.patch' | grep -Ev "$allowed" || true)
if [[ -n "$bad" ]]; then
  echo "Changes outside the editable surface are not allowed:" >&2
  echo "$bad" >&2
  exit 1
fi
git diff -- lib/treap/scheduler.hpp lib/treap/search_insert.hpp \
  lib/treap/search_delete.hpp > /app/solution.patch
echo "Wrote /app/solution.patch"
