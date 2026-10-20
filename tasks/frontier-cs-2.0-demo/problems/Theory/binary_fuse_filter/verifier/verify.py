#!/usr/bin/env python3
from pathlib import Path
from fcs2_verifier import main
if __name__ == '__main__':
    raise SystemExit(main(Path('/opt/frontier/task'), 'payload/solution.patch'))
