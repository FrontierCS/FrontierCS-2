#!/bin/sh
set -eu
repo=${1:-/workspace/binary_fuse_filter}
out=${2:-/workspace/answer.tar}
stage=$(mktemp -d /workspace/binary-fuse-submission.XXXXXX)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/payload"
git -C "$repo" add -N include
git -C "$repo" diff --no-ext-diff --no-textconv --src-prefix=a/ --dst-prefix=b/ HEAD -- include > "$stage/payload/solution.patch"
test -s "$stage/payload/solution.patch"
tar --format=ustar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf "$out" -C "$stage" payload/solution.patch
test "$(wc -c < "$out")" -le 1048576
