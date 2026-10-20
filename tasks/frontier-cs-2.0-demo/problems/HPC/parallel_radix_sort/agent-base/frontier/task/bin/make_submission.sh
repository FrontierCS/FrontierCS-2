#!/bin/sh
set -eu
workspace=${1:-/workspace/parallel_radix_sort}
patch=${2:-/workspace/solution.patch}
case "$workspace" in /workspace/*) ;; *) exit 2;; esac
case "$patch" in /workspace/*) ;; *) exit 2;; esac
git -C "$workspace" add -A -- RadixSort/edgelistgraph.h
if [ -d "$workspace/RadixSort/frontier_solution" ]; then
  git -C "$workspace" add -A -- RadixSort/frontier_solution
fi
git -C "$workspace" diff --cached --binary > "$patch"
printf '%s\n' "$patch"
