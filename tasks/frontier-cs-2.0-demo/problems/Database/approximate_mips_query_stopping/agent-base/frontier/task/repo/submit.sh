#!/usr/bin/env bash
set -euo pipefail
APP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH="$APP_DIR/solution.patch"
bash "$APP_DIR/make_submission.sh" "$PATCH"
printf 'Payload prepared at %s. Package payload/solution.patch in a tar before frontier-submit train, store-final or final. See /frontier/task/bin/README.md.\n' "$PATCH"
