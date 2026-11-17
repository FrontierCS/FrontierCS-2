#!/usr/bin/env bash
set -euo pipefail
cd /app
git add -A -- hardware/bbq/generator/bbq_level_lx.py hardware/bbq/generator/solver_extensions/ 2>/dev/null || git add -A -- hardware/bbq/generator/bbq_level_lx.py
# Also inspect all other edits; refuse rather than silently omit them.
python3 - <<'PY'
import subprocess
allowed='hardware/bbq/generator/bbq_level_lx.py'
prefix='hardware/bbq/generator/solver_extensions/'
raw=subprocess.check_output(['git','status','--porcelain','-z','--untracked-files=all'])
for item in raw.split(b'\0'):
    if not item: continue
    name=item[3:].decode()
    if name=='solution.patch': continue
    if name!=allowed and not name.startswith(prefix):
        raise SystemExit('Changes outside the editable surface are rejected')
subprocess.run(['git','add','-A','--','hardware/bbq/generator'],check=True)
with open('/app/solution.patch','wb') as f:
    subprocess.run(['git','diff','--cached','--binary'],stdout=f,check=True)
PY
