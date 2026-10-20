#!/bin/sh
set -eu
destination=${1:-/workspace/parallel_dbscan}
if test -e "$destination"; then echo "destination already exists" >&2; exit 1; fi
mkdir -p "$destination"
cp -R /frontier/task/repo/. "$destination/"
chmod -R u+w "$destination"
git -c core.hooksPath=/dev/null -C "$destination" init -q
git -C "$destination" config core.hooksPath /dev/null
git -C "$destination" add -A
git -C "$destination" -c user.name=Frontier -c user.email=task@localhost commit -qm 'Retained task substrate'
echo "$destination"
