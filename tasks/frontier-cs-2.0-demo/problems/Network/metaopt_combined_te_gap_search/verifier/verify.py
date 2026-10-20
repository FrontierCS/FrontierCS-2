#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, "/opt/frontier")
from fcs2_verifier import main
if __name__ == "__main__":
    # Pre-create shm so .NET does not initialize it via mkdtemp in /tmp.
    Path("/frontier/tmp/.dotnet/shm").mkdir(parents=True, exist_ok=True)
    raise SystemExit(main(Path("/opt/frontier/task"), Path("/opt/frontier/starter"), "payload/solution.patch"))
