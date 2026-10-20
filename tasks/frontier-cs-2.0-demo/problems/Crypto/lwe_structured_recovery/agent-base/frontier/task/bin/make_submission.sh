#!/bin/sh
# Write the ledger's patch payload: a unified diff from the read-only starter
# ledger to /workspace/payload/solution.json. Returns the patch path; the
# caller packages payload/solution.patch into the submission tar.
set -eu
if [ "$#" -ne 0 ]; then
    echo 'usage: make_submission.sh' >&2
    exit 2
fi
starter=/frontier/task/starter/payload/solution.json
answer=/workspace/payload/solution.json
patch=/workspace/payload/solution.patch
if [ ! -f "$answer" ] || [ -L "$answer" ] || [ ! -s "$answer" ]; then
    echo 'prepare a non-empty regular /workspace/payload/solution.json first' >&2
    exit 2
fi
if [ "$(wc -c < "$answer")" -gt 2000000 ]; then
    echo 'submission JSON exceeds 2000000 bytes' >&2
    exit 2
fi
status=0
diff -u --label a/payload/solution.json --label b/payload/solution.json \
    "$starter" "$answer" > "$patch.tmp" || status=$?
if [ "$status" -gt 1 ]; then
    rm -f "$patch.tmp"
    echo 'could not compare the ledger with its starter' >&2
    exit 2
fi
mv "$patch.tmp" "$patch"
chmod 0600 "$patch"
printf '%s\n' "$patch"
