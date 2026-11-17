#!/opt/frontier/venv/bin/python
"""Run public Task commands with the image's baked toolchain environment."""
import json
import os
from pathlib import Path
import sys

ENV_FILE = Path("/opt/frontier/task-runtime-env.json")
KEYS = ("PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CUDA_HOME",
        "CUDA_PATH", "CC", "CXX", "MUJOCO_GL", "XLA_PYTHON_CLIENT_PREALLOCATE",
        "PYTHONPATH")
if sys.argv[1:] == ["--snapshot"]:
    ENV_FILE.write_text(json.dumps({k: os.environ[k] for k in KEYS if k in os.environ}))
    ENV_FILE.chmod(0o444)
else:
    if len(sys.argv) < 2:
        raise SystemExit("usage: /opt/frontier/task-runtime COMMAND [ARGS...]")
    os.environ.update(json.loads(ENV_FILE.read_text()))
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
