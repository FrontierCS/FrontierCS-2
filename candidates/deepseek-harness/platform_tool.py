#!/usr/bin/env python3
"""Restore the Driver's fixed credential-file path after DSH subprocess scrubbing."""

import os
from pathlib import Path
import runpy
import sys

TOOLS = {
    'frontier-submit': 'frontier_submit.py',
    'frontier-gpu': 'frontier_gpu.py',
    'frontier-gpu-connect-proxy': 'frontier_gpu_connect_proxy.py',
}


def main() -> None:
    name = Path(sys.argv[0]).name
    if name not in TOOLS:
        raise SystemExit('unknown Frontier platform helper')
    # This is the closed Agent mount defined by driver-k8s, not a secret value.
    os.environ['FRONTIER_DRIVER_TOKEN_FILE'] = '/var/run/frontier/token'
    runpy.run_path(str(Path('/opt/frontier/candidate/deepseek-harness/tools') / TOOLS[name]),
                   run_name='__main__')


if __name__ == '__main__':
    main()
