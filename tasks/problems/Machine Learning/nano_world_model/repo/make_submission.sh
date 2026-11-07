#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
# The previous output is not source; support repeated submissions.
git add -A -- . ':!solution.patch'
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check
python3 - <<'PY'
import subprocess
import sys

allowed = {
    b'src/diffusion/df_sample.py',
    b'src/diffusion/gaussian_diffusion.py',
    b'src/diffusion/flow_matching.py',
}
changed = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z']).split(b'\0')
outside = [p for p in changed if p and p not in allowed
           and not p.startswith(b'src/diffusion/frontier_solution/')]
if outside:
    print('Changes outside the editable surface are rejected:', file=sys.stderr)
    for path in outside:
        print(path.decode('utf-8', errors='replace'), file=sys.stderr)
    sys.exit(2)
PY
git diff --cached --binary > solution.patch

echo "Wrote $PWD/solution.patch"
