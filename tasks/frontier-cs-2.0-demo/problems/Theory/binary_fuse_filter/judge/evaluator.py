"""Judge of binary_fuse_filter for the shared verifier: evaluate(tree).

`tree` is the starter repo (`pristine/`) with only the submission's `include/` files overlaid;
the shared edit scope has already enforced the editable surface. The measurement itself is
measure() in measurement.py: a strict per-workload comparison against the exact upstream headers in
`reference/` (resident bytes, build time, lookup time: C < R) plus two certified
non-inferiority guards (effective FPR, space overhead) against preregistered, reference-only
anchors; the descriptive score is the mean of 100*R/(C+R) over the three performance metrics.
raw_score = score_unbounded; reference_score = 50 (the reference against itself); margin =
raw_score - 50. A construction, correctness or false-positive gate failure is a scored zero.

Beat-spec metrics (beat_spec_workload_passes, one row per workload, 72 train / 144 final):
- resident_bytes, build_ns, lookup_ns, lower is better: pass when C < R.
- effective_fpr, space_overhead_pct, lower is better: pass when the certified lower interval
  bound does not exceed the preregistered reference-only anchor (undefined points fail).
beats_reference = 1 iff all five pass on every workload. The full prior specification is
archived at docs/task-authoring/binary_fuse_filter/provenance/beat_spec.json.

Calibration binding. load_statistical_calibration() raises CalibrationRequired unless
`statistical_calibration.json` and the private profile `reference/statistical-calibration.json`
both bind measurement_binding(): the SHA-256 over the measurement code and data only
(measurement.py, accuracy_metrics.py, harness.c, bridge.c and the reference headers). This is
the calibration proof: the preregistered anchors are valid only for the exact measurement they
were derived with. Glue (this file), the generated config.yaml and the task layout are not part
of it, so a layout migration does not invalidate the calibration. After a change to a bound
file, recompute measurement_binding() and write it into the profile and the manifest (anchors
and retained rounds never change). The production-runtime qualification record of the r8
calibration is kept under docs/task-authoring/binary_fuse_filter/provenance/.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

from accuracy_metrics import CalibrationRequired, RULE, SAMPLE_ALPHA, reference_anchors
from measurement import ALL_METRICS, InvalidCandidate, measure, workload_ids

TASK = Path(__file__).resolve().parent
MANIFEST_SCHEMA = 'binary-fuse-calibration-manifest/v3'
PROFILE_SCHEMA = 'binary-fuse-reference-statistical-calibration/v3'
# The train metrics task.toml [feedback] shows the Agent; every valid result must carry them.
FEEDBACK_METRICS = ('mean_resident_bytes', 'mean_build_ns', 'mean_lookup_ns', 'mean_effective_fpr',
                    'space_overhead_defined_workloads')
# The files the reference-only calibration certifies.
MEASUREMENT_FILES = ('measurement.py', 'accuracy_metrics.py', 'harness.c', 'bridge.c',
                     'reference/include/binaryfusefilter.h', 'reference/include/xorfilter.h')

def measurement_binding():
    digest = hashlib.sha256()
    for name in MEASUREMENT_FILES:
        body = (TASK / name).read_bytes()
        digest.update(name.encode() + b'\0' + len(body).to_bytes(8, 'big') + body)
    return 'sha256:' + digest.hexdigest()

def _sha256_file(path):
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()

def _digest_string(value):
    return (isinstance(value, str) and len(value) == 71 and value.startswith('sha256:') and
            all(c in '0123456789abcdef' for c in value[7:]))

def load_statistical_calibration():
    """Validate retained reference-only statistics and their preregistered anchors.

    measurement_binding() hashes the exact measurement sources; the manifest and the private
    profile must bind that identity. Private point values stay in the verifier-only reference
    directory.
    """
    binding = measurement_binding()
    manifest = json.loads((TASK / 'statistical_calibration.json').read_text())
    private_profile = TASK / 'reference/statistical-calibration.json'
    if (manifest.get('schema') != MANIFEST_SCHEMA or
            manifest.get('reference_statistics_status') != 'CALIBRATED_REFERENCE_ONLY' or
            manifest.get('measurement_sha256') != binding or
            not private_profile.is_file() or
            manifest.get('statistical_profile_sha256') != _sha256_file(private_profile)):
        raise CalibrationRequired('reference statistics are missing or their binding differs')
    profile = json.loads(private_profile.read_text())
    if (profile.get('schema') != PROFILE_SCHEMA or
            profile.get('status') != 'CALIBRATED_REFERENCE_ONLY' or
            profile.get('rule') != RULE or
            profile.get('measurement_sha256') != binding or
            profile.get('successor_solver_results_seen') is not False or
            profile.get('sample_alpha') != SAMPLE_ALPHA):
        raise CalibrationRequired('successor requires exact-source reference-only calibration')
    rounds = profile.get('reference_repeat_summaries')
    anchors = reference_anchors(rounds, workload_ids(True))
    evidence = profile.get('reference_evidence_sha256', [])
    runtime_receipt = profile.get('reference_runtime_receipt_sha256')
    if (not _digest_string(runtime_receipt) or
            len(evidence) != len(rounds) or len(set(evidence)) != len(evidence) or
            any(not _digest_string(h) for h in evidence) or
            profile.get('anchors') != anchors):
        raise CalibrationRequired('reference evidence or recomputed anchors disagree')
    return anchors

def evaluate(tree):
    start = time.monotonic()
    calibration = load_statistical_calibration()
    final = os.environ.get('FRONTIER_SUBMISSION_ROLE') == 'final'
    try:
        result = measure(Path(tree), final, calibration, start)
    except InvalidCandidate as exc:
        return failure(str(exc), start, final)
    missing = [name for name in FEEDBACK_METRICS if name not in result[3]]
    if missing:
        raise RuntimeError(f'measurement lacks feedback metrics {missing}')
    return result

def failure(message, start, final):
    """A candidate failure is a scored zero in every phase: every workload of the suite fails
    every beat-spec metric, so train feedback and the final record carry the same rows a
    valid result does."""
    rows = [{'workload': workload, 'metrics': {name: False for name in ALL_METRICS}}
            for workload in workload_ids(final)]
    return 0.0, 0.0, 'invalid candidate: ' + message[:1000], {
        'beats_reference': 0, 'beat_spec_workload_passes': rows,
        'reference_score': 50.0, 'margin': -50.0,
        'all_final_metrics': {'candidate_failure': {'message': message[:2000]},
                              'runtime': time.monotonic() - start}}
