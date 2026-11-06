"""Put a directory on the running interpreter's default import path (#757).

    <python> install_frontier_pth.py /opt/frontier

writes frontier.pth (that one line) into the interpreter's site-packages, so every process of
that interpreter (the judge, a worker it starts with sys.executable, `python -I`) imports the
shared modules there (isolate, fcs2_verifier, ...) by name, with no sys.path edits. Run it
at image build time with each interpreter that runs judge code: frontier-verifier-base runs it
for its python3; a GPU task's Verifier image takes it from frontier-gpu-tools
(/opt/frontier/shared-gpu/install_frontier_pth.py) and runs it with its judge interpreter(s).
An existing frontier.pth is an error: the image installs it once.
"""

import sys
import sysconfig
from pathlib import Path

if len(sys.argv) != 2 or not Path(sys.argv[1]).is_absolute():
    raise SystemExit("usage: install_frontier_pth.py <absolute directory>")
site_packages = Path(sysconfig.get_paths()["purelib"])
site_packages.mkdir(parents=True, exist_ok=True)
with (site_packages / "frontier.pth").open("x", encoding="utf-8") as pth:
    pth.write(sys.argv[1] + "\n")
