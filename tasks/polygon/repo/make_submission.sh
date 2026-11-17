#!/usr/bin/env bash
set -euo pipefail
cd /app
rm -f -- /app/solution.patch
git add -A
mapfile -t changed < <(git diff --cached --name-only)
if [[ ${#changed[@]} -eq 0 ]]; then
  echo "no changes to submit" >&2
  exit 1
fi
for path in "${changed[@]}"; do
  case "$path" in
    sql_testing/environment.py|sql_testing/smt/formula.py|sql_testing/smt/knowledgebase.py|sql_testing/smt/search_extensions/*) ;;
    *) echo "change outside editable surface: $path" >&2; exit 1 ;;
  esac
done
git diff --cached --no-ext-diff --src-prefix=a/ --dst-prefix=b/ > /app/solution.patch
test -s /app/solution.patch
echo /app/solution.patch
