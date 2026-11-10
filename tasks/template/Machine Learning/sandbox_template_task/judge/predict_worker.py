"""One predict of one implementation, inside an isolate instance on the timing machine.

    predict_worker.py WORK PARAMS QUERIES

judge/timing_harness.py runs it with the implementation's tree as the work dir; the harness
times the whole process from outside (Result.command_seconds), so nothing here is trusted.
"""

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    work, params, queries = argv
    sys.path.insert(0, work)
    import model

    xs = json.loads(Path(queries).read_text())["x"]
    if len(model.predict(json.loads(Path(params).read_text()), xs)) != len(xs):
        raise ValueError("predict returned the wrong number of values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
