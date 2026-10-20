"""Private author entrypoint; production policy continues to use verify.py.

Only the exact private reference patch is accepted by evaluate_reference().
The shared adapter still performs extraction, import, scoring and projection.
This command measures runtime; it does not certify its own result.
"""
from pathlib import Path
from fcs2_verifier import main

if __name__ == '__main__':
    raise SystemExit(main(Path('/opt/frontier/task'), 'payload/solution.patch',
                          reference_only=True))
