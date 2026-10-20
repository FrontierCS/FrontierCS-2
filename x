#!/bin/sh
# Repository commands use the same managed interpreter as arbitrary Python code.
set -eu
root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
if [ "${1:-}" = --offline ]; then
    export FRONTIER_BOOTSTRAP_OFFLINE=1
fi
if [ "${1:-}" = uv ] || { [ "${1:-}" = --offline ] && [ "${2:-}" = uv ]; }; then
    # uv lock must remain available when the locked Python environment is stale.
    [ "$1" != --offline ] || shift
    shift
    # shellcheck source=scripts/bootstrap-uv.sh
    . "$root/scripts/bootstrap-uv.sh"
    uv=$(bootstrap_uv "$root" x)
    exec "$uv" "$@"
fi
exec "$root/python" -B "$root/scripts/repo.py" "$@"
