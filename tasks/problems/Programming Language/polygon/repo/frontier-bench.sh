#!/usr/bin/env bash
set -euo pipefail
python_bin="$(command -v python3)"
export PATH=/usr/bin:/usr/local/bin:/bin
db_root="$(mktemp -d "${TMPDIR:-/tmp}/polygon-smoke.XXXXXX")"
db_pid=""
cleanup() {
  if [[ -n "$db_pid" ]]; then
    kill "$db_pid" 2>/dev/null || true
    wait "$db_pid" 2>/dev/null || true
  fi
  rm -rf -- "$db_root"
}
trap cleanup EXIT
mariadb-install-db --no-defaults --datadir="$db_root/data" \
  --auth-root-authentication-method=normal --lower-case-table-names=1 \
  >/dev/null 2>&1
server_user=()
if [[ "$(id -u)" -eq 0 ]]; then
  server_user=(--user=root)
fi
/usr/sbin/mariadbd --no-defaults "${server_user[@]}" --datadir="$db_root/data" \
  --socket="$db_root/mysql.sock" --pid-file="$db_root/mysql.pid" \
  --skip-grant-tables --skip-networking --lower-case-table-names=1 \
  >"$db_root/server.log" 2>&1 &
db_pid=$!
for _ in $(seq 1 100); do
  if mysqladmin --socket="$db_root/mysql.sock" ping --silent; then
    break
  fi
  sleep 0.05
done
mysqladmin --socket="$db_root/mysql.sock" ping --silent
FRONTIER_MYSQL_SOCKET="$db_root/mysql.sock" \
  "$python_bin" frontier_runner.py < public_smoke.json
FRONTIER_MYSQL_SOCKET="$db_root/mysql.sock" \
  "$python_bin" frontier_runner.py < public_disambiguation_smoke.json
