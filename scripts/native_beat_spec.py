"""Formal beat specs for tasks that keep a reviewed native verifier.

Beat spec v2. Some tasks decide PASS per workload inside their own
task-local verifier rather than a generic ``judge/evaluator.py``. Their native
contract modules source-bind the exact reviewed verifier and exercise its
decision on bounded fixtures. Historically those contracts forbade a
``judge/beat_spec.json`` so an out-of-date spec could not drift from the
verifier. This module lets a task carry a formal ``judge/beat_spec.json`` that
is required to transcribe exactly what its pinned verifier already does:

1. the task's own native contract still runs and still pins the exact verifier
   source (now including ``judge/beat_spec.json`` in its reviewed inventory), so
   the spec cannot drift from the code;
2. the spec passes the portable schema used by every other task;
3. the spec names exactly the per-workload metrics the verifier compares, lists
   exactly the train-feedback fields the verifier exposes, declares the
   all-workload/all-metric conjunction, and its public statement documents every
   metric and feedback field.

This is a structural and fixture-level check, not a workload measurement.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from vllm_native_contract import SOLVER_FEEDBACK as VLLM_SOLVER_FEEDBACK

# Per task: the exact per-workload metric names the pinned verifier compares,
# and the exact train-feedback field names it exposes to the solver. These are
# transcribed from the reviewed verifier and cross-checked here; the native
# contract guarantees the verifier bytes that emit them.
NATIVE_SPECS = {
    "join_tree_rewriting": {
        "contract": "join_tree_contract",
        "metrics": ["exact_result", "median_latency"],
        # Aggregates derived by the adapter from the native per-query records:
        # the count of train queries clearing both comparisons and the
        # candidate's own suite-median latency, beside beat_spec_progress.
        "solver_feedback": ["all_metrics_progress", "candidate_median_latency_us"],
    },
    "vllm_llm_serving_optimization": {
        "contract": "vllm_native_contract",
        # One decision per workload/load group: paired timing, three tail
        # latencies and median-minus-allowance correctness, identical for all groups.
        "metrics": ["paired_timing", "p90", "p95", "p99", "correctness"],
        # Exact public candidate aggregates and aggregate existing PASS checks.
        "solver_feedback": VLLM_SOLVER_FEEDBACK,
    },
}
PROGRESS_FIELDS = (
    "beat_spec_progress", "passed", "failed", "total", "pass_percent", "fail_percent",
)


class NativeBeatSpecFailure(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeBeatSpecFailure(message)


def is_native(root: Path) -> bool:
    return root.name in NATIVE_SPECS


def statement_text(root: Path) -> str:
    task = root / "agent-base/frontier/task"
    parts = []
    for path in (task / "README.md", task / "guides/statement.md", task / "repo/README.md"):
        if path.is_file() and not path.is_symlink():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def check(root: Path, evaluation: dict, schema_check) -> None:
    spec_info = NATIVE_SPECS.get(root.name)
    require(spec_info is not None, "task has no reviewed native beat-spec contract")
    contract = importlib.import_module(spec_info["contract"])
    # The native contract pins the exact verifier source, now including
    # judge/beat_spec.json in its reviewed inventory.
    try:
        contract.check_native_contract(root, evaluation)
    except contract.NativeContractFailure as error:
        raise NativeBeatSpecFailure(str(error)) from error

    path = root / "judge/beat_spec.json"
    require(path.is_file() and not path.is_symlink(), "beat_spec.json must be a regular file")
    spec = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(spec, dict), "beat_spec.json must contain a JSON object")

    expected_feedback = list(spec_info["solver_feedback"])
    # Native tasks carry no policy allowlist; the verifier's actual train
    # feedback is the allowlist, so validate the spec against it.
    names, visible = schema_check(
        spec, dict(evaluation, solver_visible_metrics=expected_feedback)
    )
    require(
        sorted(names) == sorted(spec_info["metrics"]),
        "beat_spec.json metrics differ from the native verifier's per-workload decisions",
    )
    require(
        sorted(visible) == sorted(expected_feedback),
        "beat_spec.json solver_feedback differs from the native verifier's train feedback",
    )
    require(
        spec.get("beats_reference_aggregation") == {"kind": "conjunction"},
        "native beat spec must declare the all-workload, all-metric conjunction",
    )
    # Tasks that expose per-metric progress to the solver must document those
    # aggregate fields, and every other train-feedback field, in the public
    # statement. A task whose train feedback is score-only (empty
    # solver_feedback) shows no beat_spec_progress, so the requirement is waived
    # for it.
    if expected_feedback:
        statement = statement_text(root)
        missing = [name for name in PROGRESS_FIELDS[1:] if name not in statement]
        require(not missing, f"public statement does not describe beat-spec fields {missing}")
        if root.name == "join_tree_rewriting":
            missing = [name for name in [PROGRESS_FIELDS[0], *expected_feedback]
                       if name not in statement]
            require(not missing, f"public statement does not describe train feedback {missing}")
