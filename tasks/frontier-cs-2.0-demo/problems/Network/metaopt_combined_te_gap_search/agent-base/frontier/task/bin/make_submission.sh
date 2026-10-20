#!/usr/bin/env bash
# Write the patch payload for the editable surface of /workspace/project: a text
# diff against the task baseline, with paths under project/. Only the search file
# and Candidate/ helpers are staged. Returns the patch path; the caller packages
# payload/solution.patch into the submission tar.
set -euo pipefail
workspace=/workspace/project
patch=/workspace/payload/solution.patch
editable=(MetaOptimize/TrafficEngineering/BudgetedDemandSearch.cs MetaOptimize/TrafficEngineering/Candidate)
if ! git -C "$workspace" rev-parse --verify -q HEAD >/dev/null; then
    echo 'run /frontier/task/bin/prepare_workspace.sh first' >&2
    exit 2
fi
git -C "$workspace" reset -q
git -C "$workspace" add -A -- "${editable[@]}"
if git -C "$workspace" diff --cached --name-status | grep -q '^[DR]'; then
    echo 'deleting or renaming files is not supported; edit or empty them instead' >&2
    exit 2
fi
if git -C "$workspace" diff --cached --numstat | grep -q '^-'; then
    echo 'binary files cannot be submitted; Candidate/ helpers must be C# source' >&2
    exit 2
fi
if [ -n "$(git -C "$workspace" status --porcelain --untracked-files=all -- . ":(exclude)${editable[0]}" ":(exclude)${editable[1]}")" ]; then
    echo 'note: changes outside the editable surface are not submitted' >&2
fi
mkdir -p /workspace/payload
git -C "$workspace" diff --cached --src-prefix=a/project/ --dst-prefix=b/project/ > "$patch"
printf '%s\n' "$patch"
