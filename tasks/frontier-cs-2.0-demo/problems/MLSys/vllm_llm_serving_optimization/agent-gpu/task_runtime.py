#!/opt/frontier/runtime-python
"""Restore the baked CUDA/Python environment after the trusted SSH bootstrap."""
import json
import os
from pathlib import Path
import sys

ENV_FILE = Path('/opt/frontier/task-runtime-env.json')
KEYS = ('PATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH', 'CUDA_HOME', 'CUDA_PATH',
        'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'DO_NOT_TRACK', 'VLLM_TARGET_DEVICE')
if sys.argv[1:] == ['--snapshot']:
    ENV_FILE.write_text(json.dumps({key: os.environ[key] for key in KEYS if key in os.environ}))
    ENV_FILE.chmod(0o444)
else:
    if len(sys.argv) < 2:
        raise SystemExit('usage: /opt/frontier/task-runtime COMMAND [ARGS...]')
    os.environ.update(json.loads(ENV_FILE.read_text()))
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
