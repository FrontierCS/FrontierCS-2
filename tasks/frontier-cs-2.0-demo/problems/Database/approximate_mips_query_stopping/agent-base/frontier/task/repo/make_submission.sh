#!/usr/bin/env bash
set -euo pipefail

APP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT="${1:-$APP_DIR/solution.patch}"
BASE="$APP_DIR/.submission-base/src/alg.cpp"
CURRENT="$APP_DIR/src/alg.cpp"
BASE_SHA256="8b4a0077d6436935ad968179fdba624a6fb966759b87b1a64340544e8913b8bc"

actual_base=$(sha256sum "$BASE" | awk '{print $1}')
if [[ "$actual_base" != "$BASE_SHA256" ]]; then
  echo "The shipped submission baseline was modified; restore .submission-base/src/alg.cpp" >&2
  exit 2
fi

if [[ -f "$APP_DIR/.submission-manifest.sha256" ]] && \
   ! (cd "$APP_DIR" && sha256sum -c .submission-manifest.sha256 >/dev/null); then
  echo "A non-editable source or harness file was modified" >&2
  exit 2
fi

BODY="$OUT.body.$$"
trap 'rm -f "$BODY"' EXIT
if diff -u --label a/src/alg.cpp --label b/src/alg.cpp "$BASE" "$CURRENT" > "$BODY"; then
  echo "src/alg.cpp is unchanged; empty patches are not valid" >&2
  exit 2
fi

printf 'diff --git a/src/alg.cpp b/src/alg.cpp\n' > "$OUT"
cat "$BODY" >> "$OUT"
echo "Wrote $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)"
