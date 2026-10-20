#!/usr/bin/env bash
set -euo pipefail

repo="${JOINBOOST_DIR:-/app}"
out="${1:-/app/solution.patch}"
allowed=(
  src/joinboost/app.py
  src/joinboost/semiring.py
  src/joinboost/aggregator.py
  src/joinboost/cjt.py
  src/joinboost/joingraph.py
)

if [[ ! -d "$repo/.git" ]]; then
  echo "JoinBoost repository not found at $repo" >&2
  exit 2
fi

declare -A accepted=()
for path in "${allowed[@]}"; do accepted["$path"]=1; done
outside=()
while IFS= read -r -d '' path; do
  [[ ${accepted[$path]+yes} ]] || outside+=("$path")
done < <(git -C "$repo" diff --name-only -z HEAD --)
if (( ${#outside[@]} )); then
  echo "ERROR: changes outside the editable surface:" >&2
  printf '  %s\n' "${outside[@]}" >&2
  exit 3
fi

git -C "$repo" diff --binary --no-color HEAD -- "${allowed[@]}" > "$out"
if [[ ! -s "$out" ]]; then
  echo "ERROR: an empty patch is not a valid submission" >&2
  exit 4
fi
echo "Wrote $out ($(wc -c < "$out" | tr -d ' ') bytes)."
