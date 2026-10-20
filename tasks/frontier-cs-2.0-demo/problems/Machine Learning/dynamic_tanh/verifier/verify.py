#!/usr/bin/env python3
"""Verifier entrypoint for r_top01_2503-10622-cand_000-preflight-9740a08b6f00. Bundle path and entrypoint, then the shared adapter."""

import sys
from pathlib import Path

sys.path.insert(0, "/opt/frontier")

from fcs2_verifier import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(Path("/opt/frontier/task"), 'payload/solution.patch'))
