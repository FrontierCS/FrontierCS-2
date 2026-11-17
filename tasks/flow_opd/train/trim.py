"""trim.py <judge directory>: leaves the final suite out of a copy of the judge, in place, for the
train judge's image.

The final suite's eight workloads (their prompts and seeds) are removed from the frozen suite
manifest, their entries from the reference cache, and `final` from the measured reference scores
and runner seconds; the source test corpus (`judge/corpus/test.txt`), which holds those prompts,
is deleted. The qualification record binds the manifest and the cache by SHA-256, so it is
rewritten with the trimmed files' digests. A train call selects the development and train-grading workloads alone,
and reads the corpus through the run's own copy of the training file, so nothing it reads changes.
"""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "judge"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def store(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


manifest_path = root / "suite_manifest.json"
manifest = load(manifest_path)
final = {row["id"] for row in manifest["workloads"] if row["suite"] == "final"}
assert len(final) == 8, final
manifest["workloads"] = [row for row in manifest["workloads"] if row["suite"] != "final"]
store(manifest_path, manifest)

cache_path = root / "reference_cache.json"
cache = load(cache_path)
assert final <= set(cache["workloads"])
cache["workloads"] = {key: value for key, value in cache["workloads"].items() if key not in final}
store(cache_path, cache)

(root / "corpus" / "test.txt").unlink()

record_path = root / "qualification.json"
record = load(record_path)
del record["measured_reference_scores"]["final"], record["measured_runner_seconds"]["final"]
for key, path in (("suite_manifest", manifest_path), ("reference_cache", cache_path)):
    record[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
store(record_path, record)
