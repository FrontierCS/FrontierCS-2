#!/usr/bin/env python3
"""Native verifier entrypoint for egraph_invariant_restoration."""
import sys
from pathlib import Path
sys.path.insert(0, "/opt/frontier")
from fcs2_verifier import main
if __name__ == "__main__":
    raise SystemExit(main(Path("/opt/frontier/task"), "payload/solution.patch"))
