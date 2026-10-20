#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/bin:/usr/local/bin:/bin
python3 -m compileall -q sql_testing evaluation frontier_runner.py
z3 --version | grep -q 'Z3 version 4.13'
command -v mariadb-install-db >/dev/null
test -x /usr/sbin/mariadbd
