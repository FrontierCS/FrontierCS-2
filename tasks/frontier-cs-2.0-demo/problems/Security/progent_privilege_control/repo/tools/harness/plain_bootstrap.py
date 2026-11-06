"""Unsandboxed policy-worker launch for local development only (never used by the judge).

Runs the same worker and protocol as sealed_bootstrap.py with the submitted tree on
sys.path, for machines without Landlock ABI 6. The judge always uses the sealed
bootstrap; code that relies on anything the sealed worker forbids (files, network,
processes, threads) fails there.
"""

import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 5:
        raise ValueError("expected public tree, worker, protocol and parent PID")
    tree, worker, protocol = map(Path, sys.argv[1:4])
    os.environ.clear()
    os.environ.update(PATH="/usr/bin:/bin", HOME=str(tree), PYTHONDONTWRITEBYTECODE="1", SECAGENT_GENERATE="False")
    sys.path[:0] = [str(protocol.parent), str(tree), str(tree / "agentdojo" / "src")]
    namespace = {"__name__": "__main__", "__file__": str(worker)}
    exec(compile(worker.read_bytes(), str(worker), "exec"), namespace)


if __name__ == "__main__":
    main()
