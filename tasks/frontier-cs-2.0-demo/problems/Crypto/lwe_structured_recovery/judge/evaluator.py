"""Judge of lwe_structured_recovery for the shared verifier (#757): evaluate(tree).

`tree` is the starter repo with the submission's one allowed file, `solution.json` (the
witness ledger), overlaid. The checker and the 200-instance catalog come from the trusted
pristine copy (`pristine/`), never from the submission's tree.

Score: raw_score = 100 * solved_count / instance_count and score_unbounded = solved_count, the
public checker's continuous score; solved_count counts distinct catalog instances whose unique
admissible ledger record passes every public predicate. Train and final check the same 200
public instances.

Beat-spec metric (beat_spec_workload_passes, one row per instance):
- valid_witness, higher is better: 1 when the ledger supplies a valid secret for the instance,
  else 0; passes when the candidate's value exceeds the frozen empty reference ledger's
  (`reference.json`, which solves nothing).
beats_reference = 1 iff every one of the 200 instances passes, i.e. all are solved.

The full prior specification is archived at
docs/task-authoring/lwe_structured_recovery/provenance/beat_spec.json.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRISTINE = ROOT / "pristine"
PUBLIC_METRICS = frozenset({
    "instance_count", "solved_count", "submitted_count", "invalid_count", "duplicate_count",
    "conflict_count", "unknown_count", "rejection_code_counts", "invalid_examples", "solved_ids",
})


def _checker():
    """The catalog and checker from the pristine repo."""
    sys.path.insert(0, str(PRISTINE))
    try:
        from lwe_challenge.evaluator_core import evaluate_path
        from lwe_challenge.schema import Catalog
    finally:
        sys.path.remove(str(PRISTINE))
    return Catalog.load(PRISTINE / "catalog.jsonl"), evaluate_path


def _plain(value):
    """The checker's metrics as plain JSON values (tuples become lists)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def workload_decisions(workload_ids, candidate_solved, reference_solved):
    ids = tuple(workload_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("workloads must be nonempty and unique")
    candidate = set(candidate_solved)
    reference = set(reference_solved)
    if not candidate.issubset(ids) or not reference.issubset(ids):
        raise ValueError("unknown solved workload")
    rows = [
        {"workload": identity,
         "metrics": {"valid_witness": int(identity in candidate) > int(identity in reference)}}
        for identity in ids
    ]
    return {
        "beat_spec_workload_passes": rows,
        "beats_reference": int(all(row["metrics"]["valid_witness"] for row in rows)),
    }


def evaluate(tree: str):
    catalog, evaluate_path = _checker()
    evaluated = evaluate_path(Path(tree) / "solution.json", catalog=catalog)
    score, unbounded = float(evaluated.score), float(evaluated.score_unbounded)
    if not (math.isfinite(score) and score >= 0 and math.isfinite(unbounded) and unbounded >= 0):
        raise RuntimeError("score is not finite and non-negative")
    metrics = _plain(evaluated.metrics)
    if set(metrics) != PUBLIC_METRICS:
        raise RuntimeError("public metric fields changed")
    reference = evaluate_path(ROOT / "reference.json", catalog=catalog)
    decisions = workload_decisions([i.instance_id for i in catalog.instances],
                                   metrics["solved_ids"], reference.solved_ids)
    return score, unbounded, str(evaluated.message), {
        **metrics, **decisions,
        "reference_score": float(reference.score), "margin": score - float(reference.score),
    }
