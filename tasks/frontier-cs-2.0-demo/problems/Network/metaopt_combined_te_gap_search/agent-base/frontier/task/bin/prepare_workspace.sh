#!/usr/bin/env bash
# Copy the starter project into /workspace/project with a Git baseline that
# make_submission.sh diffs against.
set -euo pipefail
destination=/workspace/project
if [ ! -e "$destination" ]; then
    cp -R /frontier/task/repo/project "$destination"
    chmod -R u+w "$destination"
    git -c core.hooksPath=/dev/null -C "$destination" init -q
    git -C "$destination" config core.hooksPath /dev/null
    printf 'bin/\nobj/\n__pycache__/\n' >> "$destination/.git/info/exclude"
    git -C "$destination" add -A
    git -c user.name=Frontier -c user.email=frontier@localhost -C "$destination" \
        commit -qm 'Task baseline'
fi
printf '%s\n' "$destination"
