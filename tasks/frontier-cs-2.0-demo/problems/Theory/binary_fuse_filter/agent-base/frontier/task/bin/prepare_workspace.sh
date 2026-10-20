#!/bin/sh
set -eu
dest=${1:-/workspace/binary_fuse_filter}
test ! -e "$dest"
mkdir -p "$dest"
cp -R /frontier/task/repo/. "$dest/"
chmod -R u+rwX "$dest"
git -C "$dest" init -q
git -C "$dest" add .
git -C "$dest" -c user.name=solver -c user.email=solver@localhost commit -qm baseline
