"""Compile one KAMI device module inside its isolate instance (judge/evaluator.py runs it).

argv: SOURCE OUTPUT M N K ACTIVE_RANKS ALLOCATED_RANKS. Runs the public adapter's compile_cubin
(the judge's pristine frontier_harness/, copied beside this file) on the submitted source in the
instance's own tree and prints one JSON line: {"rendered_sha256", "returncode", "stdout",
"stderr"}, or {"timeout": true}, or {"error": message} when the adapter rejects the source.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontier_harness.adapter import compile_cubin  # noqa: E402


def main(argv: list[str]) -> int:
    source, output, m, n, k, active, allocated = argv[1:]
    try:
        rendered, proc = compile_cubin(Path(source), Path(output), m=int(m), n=int(n), k=int(k),
                                       active_ranks=int(active), allocated_ranks=int(allocated))
    except subprocess.TimeoutExpired:
        report = {"timeout": True}
    except ValueError as error:
        report = {"error": str(error)}
    else:
        report = {"rendered_sha256": rendered.rendered_sha256, "returncode": proc.returncode,
                  "stdout": proc.stdout, "stderr": proc.stderr}
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
