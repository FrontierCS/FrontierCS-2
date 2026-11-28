"""Fixed launcher of one model-requested interactive SWE command, as the tool user."""
from __future__ import annotations

import ctypes
import os
import sys

TOOL_UID = 10001
TOOL_GID = 10001
PR_SET_NO_NEW_PRIVS = 38
ALLOWED_ENVIRONMENT = frozenset({
    "PATH", "TZ", "LANG", "LC_ALL", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
})


def clean_environment(source):
    result = {key: value for key, value in source.items()
              if key in ALLOWED_ENVIRONMENT and isinstance(value, str)}
    result.update({
        "HOME": "/home/frontier-tool",
        "TMPDIR": "/var/tmp/frontier-tool",
        "XDG_CACHE_HOME": "/var/tmp/frontier-tool/cache",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    })
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if (len(argv) != 3 or argv[:2] != ["/bin/bash", "-lc"]
            or not isinstance(argv[2], str) or "\0" in argv[2]):
        raise ValueError("launcher accepts only one fixed bash command")
    # The container's own user is the tool user, in no group but its own: Pumpkin starts no
    # container as root.
    if os.getuid() != TOOL_UID or os.getgid() != TOOL_GID or set(os.getgroups()) - {TOOL_GID}:
        raise PermissionError("interactive command launcher must start as the tool user")
    environment = clean_environment(os.environ)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.umask(0o022)
    os.chdir("/testbed")
    os.execve(argv[0], argv, environment)


if __name__ == "__main__":
    main()
