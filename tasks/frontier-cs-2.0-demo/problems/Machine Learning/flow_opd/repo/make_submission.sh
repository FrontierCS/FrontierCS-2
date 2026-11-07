#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
git add -A -- . ':!solution.patch'
python3 - <<'PY'
import subprocess
allowed = {'flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py', 'flow_grpo/diffusers_patch/sd3_sde_with_logprob.py'}
paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z']).decode().split('\0')
if any(p and p not in allowed and not p.startswith('flow_grpo/frontier_solution/') for p in paths):
    raise SystemExit('Changes outside the permitted source files and helper namespace are rejected.')
PY
git diff --cached --no-ext-diff --no-color --binary > solution.patch
