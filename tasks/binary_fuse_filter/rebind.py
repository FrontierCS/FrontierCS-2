"""Re-binds the train judge's calibration to its measurement, in a copy of the judge tree.

    python3 rebind.py <judge directory>

The judge tree is the source's with `train/measurement.py` laid over it: a measurement that has
no final seed, so that `workload_ids(True)` is the train workloads. The calibration is certified
for the exact measurement sources (`evaluator.measurement_binding()`), so this rewrites, for
that measurement:

- the private profile `reference/statistical-calibration.json`, keeping only the train
  workloads in each table keyed by workload (the reference rounds, the anchors, the retained
  deltas) and binding the new measurement hash;
- the manifest `statistical_calibration.json`, binding the same hash and the new profile's.

Every kept value is the source's: the anchors and the rounds are per workload and unchanged.
It then checks the result with the judge's own `load_statistical_calibration()`.
"""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import evaluator
from measurement import workload_ids

ids = set(workload_ids(True))
binding = evaluator.measurement_binding()

profile_path = root / "reference/statistical-calibration.json"
profile = json.loads(profile_path.read_text())
profile["reference_repeat_summaries"] = [
    {key: value for key, value in rnd.items() if key in ids}
    for rnd in profile["reference_repeat_summaries"]
]
for table in ("anchors", "retained_reference_deltas"):
    profile[table] = {key: value for key, value in profile[table].items() if key in ids}
assert all(set(rnd) == ids for rnd in profile["reference_repeat_summaries"])
assert all(set(profile[table]) == ids for table in ("anchors", "retained_reference_deltas"))
profile["measurement_sha256"] = binding
profile_path.write_text(json.dumps(profile, indent=1) + "\n")

manifest_path = root / "statistical_calibration.json"
manifest = json.loads(manifest_path.read_text())
manifest["measurement_sha256"] = binding
manifest["statistical_profile_sha256"] = "sha256:" + hashlib.sha256(profile_path.read_bytes()).hexdigest()
manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")

evaluator.load_statistical_calibration()
