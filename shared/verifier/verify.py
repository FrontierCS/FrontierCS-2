#!/usr/bin/env python3
"""Verifier entrypoint of every task built on frontier-verifier-base.

The task's build writes /opt/frontier/verifier.json from its task.toml: the submission
entrypoint and, for a task with command test sandboxes, `sandboxes`.
"""

import json
from pathlib import Path

from fcs2_verifier import main  # /opt/frontier, on the import path by frontier.pth

if __name__ == "__main__":
    parameters = json.loads(Path("/opt/frontier/verifier.json").read_text(encoding="utf-8"))
    raise SystemExit(main(Path("/opt/frontier/task"), parameters["entrypoint"],
                          sandboxes="sandboxes" in parameters))
