#!/bin/sh
set -eu
destination=${1:-/workspace/parallel_radix_sort}
case "$destination" in /workspace/*) ;; *) echo 'workspace must be under /workspace' >&2; exit 2;; esac
[ ! -e "$destination" ] || { echo 'destination exists' >&2; exit 2; }
cp -R /frontier/task/repo "$destination"
chmod -R u+w "$destination"
git -c core.hooksPath=/dev/null -C "$destination" init -q
git -C "$destination" config core.hooksPath /dev/null
git -C "$destination" add -A
git -c user.name=Frontier -c user.email=frontier@localhost -C "$destination" commit -qm 'Task baseline'
printf '%s\n' "$destination"
