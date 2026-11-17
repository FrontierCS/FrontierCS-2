"""trim.py <judge directory>: leaves the final suite out of a copy of the judge's `workloads.json`,
in place, for the train judge's image.

The file holds a `development` suite (two workloads of 10 regexes) and a `final` suite (the same
two workloads and two held-out ones of 40). A train call reads `development` alone
(`evaluator.py`: `data["final"] if final else data["development"]`), so the key `final` is
removed; the file is otherwise written as the source writes it (two-space indent, one newline at
the end).
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "workloads.json"
data = json.loads(path.read_text(encoding="utf-8"))
assert list(data) == ["development", "final"], list(data)
assert data["final"][0] == data["development"][0] and data["final"][2] == data["development"][1], \
    "the final suite no longer repeats the development workloads"
del data["final"]
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
