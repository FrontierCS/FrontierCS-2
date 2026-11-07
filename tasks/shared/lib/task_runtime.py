#!/opt/frontier/runtime-python
"""/opt/frontier/task-runtime of a CPU runner image (tasks/shared/images/cpu-runner.Dockerfile):
run a command with the image's own environment, which sshd clears for every session.

  task-runtime --snapshot        (the build, once) record the image's ENV
  task-runtime PROGRAM [ARG...]  exec PROGRAM with that ENV over the session's

A command test sandbox step's argv is ["/opt/frontier/task-runtime", PROGRAM, ...], the same on a
GPU runner image (whose task-runtime replays its toolchain the same way)."""
import json
import os
import sys
from pathlib import Path

ENV_FILE = Path("/opt/frontier/task-runtime-env.json")
# The build's own variables, the build user's identity (the session's HOME/USER stay the runner
# user's) and per-process shell state are not the image's environment; nor is a builder's own.
NOT_IMAGE = {"DEBIAN_FRONTEND", "RUNNER_BASE", "GPU_TOOLS", "HOSTNAME", "PWD", "OLDPWD", "SHLVL",
             "_", "TERM", "HOME", "USER", "LOGNAME"}
NOT_IMAGE_PREFIXES = ("MODAL_",)

if sys.argv[1:] == ["--snapshot"]:
    ENV_FILE.write_text(json.dumps({k: v for k, v in sorted(os.environ.items())
                                    if k not in NOT_IMAGE
                                    and not k.startswith(NOT_IMAGE_PREFIXES)}, indent=1) + "\n")
    ENV_FILE.chmod(0o444)
else:
    if len(sys.argv) < 2:
        raise SystemExit("usage: /opt/frontier/task-runtime PROGRAM [ARGS...]")
    os.environ.update(json.loads(ENV_FILE.read_text()))
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)
