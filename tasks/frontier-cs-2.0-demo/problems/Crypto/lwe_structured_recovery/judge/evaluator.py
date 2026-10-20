"""Exact workload-level pass/fail, separate from public continuous scoring."""

import hashlib
from pathlib import Path


REFERENCE_SHA256 = hashlib.sha256(b'{"schema_version":1,"solutions":[]}\n').hexdigest()


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


def evaluate_reference_comparison(catalog, candidate_metrics, evaluate_path):
    reference_path = Path(__file__).with_name("reference.json")
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("reference ledger identity changed")
    reference = evaluate_path(reference_path, catalog=catalog)
    if reference.score != 0 or reference.solved_ids:
        raise ValueError("reference is not the approved empty ledger")
    if len(catalog.instances) != 200 or candidate_metrics["instance_count"] != 200:
        raise ValueError("production corpus must contain all 200 workloads")
    decisions = workload_decisions(
        [instance.instance_id for instance in catalog.instances],
        candidate_metrics["solved_ids"], reference.solved_ids,
    )
    decisions["reference_score"] = reference.score
    return decisions
