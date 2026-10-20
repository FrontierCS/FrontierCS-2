#!/bin/sh
set -eu

source_root=${1:-/workspace/repo}
case "$source_root" in
  /workspace/*) ;;
  *) echo "source root must be below /workspace" >&2; exit 2 ;;
esac
if [ ! -d "$source_root/.git" ]; then
  echo "run /frontier/task/bin/prepare_workspace.sh first" >&2
  exit 2
fi
install -d -m 0700 /workspace/payload
temporary=/workspace/payload/.solution.patch.tmp
git -C "$source_root" add -N -- ':(glob)ivfpqlib/**/*.py'
git -C "$source_root" diff --name-only HEAD -- ':(glob)ivfpqlib/**/*.py' |
while IFS= read -r path; do
  case "$path" in
    */refivfpq.py|*/conftest.py)
      echo "reserved or trusted file may not change: $path" >&2
      exit 2 ;;
  esac
done
git -C "$source_root" diff --no-ext-diff HEAD -- ':(glob)ivfpqlib/**/*.py' > "$temporary"
size=$(wc -c < "$temporary")
if [ "$size" -gt 2000000 ]; then
  rm -f "$temporary"
  echo "submission patch exceeds 2000000 bytes" >&2
  exit 2
fi
mv "$temporary" /workspace/payload/solution.patch
chmod 0600 /workspace/payload/solution.patch
printf '%s\n' /workspace/payload/solution.patch
