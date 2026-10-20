#!/usr/bin/env python3
"""Verifier entrypoint for r_top01_2604-28190-cand_000-preflight-8250ab0a4a09. Bundle path and entrypoint, then the shared adapter."""

import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/frontier")

if __name__ == "__main__":
    remaining = os.environ.get("FRONTIER_VERIFIER_REMAINING_SECONDS")
    if remaining is not None:
        seconds = float(remaining)
        if not math.isfinite(seconds) or not 0 < seconds <= 7 * 24 * 3600:
            raise ValueError("invalid remaining Verifier duration")
        # This timestamp stays on the GPU host; host monotonic clocks are not shared.
        os.environ["FRONTIER_VERIFIER_LOCAL_DEADLINE"] = str(time.monotonic() + seconds)
    from fcs2_verifier import main

    raise SystemExit(main(Path("/opt/frontier/task"), 'payload/solution.patch'))
