"""Bounded compatibility check for the reviewed native JoinTree protocol.

This source-bound check is not a measurement or acceptance receipt. It keeps
the original 20-train/39-final protocol and the complete private final verdict.
The task carries a formal judge/beat_spec.json that transcribes this verdict
(docs/task-authoring/join_tree_rewriting/beat-spec-v2.md). Train feedback is
aggregate only (docs/task-authoring/join_tree_rewriting/aggregate-train-feedback.md):
the adapter turns the native 'queries' records into per-metric pass counts, the
count of queries clearing both comparisons and the candidate's own suite-median
latency. The spec, the adapter and the feedback helper are hash-pinned here, so
none can drift. Other sources still require the ordinary beat spec.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any


class NativeContractFailure(ValueError):
    pass


# Reviewed source from Smith 227fefc46bfefe7743db3ccffae1a0057bb5f878,
# preserved in Preview PR36. Hashes guard execution of task-owned Python.
REVIEWED_FILES = {
    "judge/evaluator.py": "df102d47482da816254f310d6b248956c7f33f2575c04adb341a7583c586dc62",
    "judge/query_measurements.py": "c2ff58f06fbdefc7a8634e77b419dbe615bd9fe860830436c34081fa5de60665",
    "verifier/fcs2_verifier.py": "22e49ea8934ca1088751ba636ebe93782dc9d60017977e6be03b5d16f74cafb6",
    "verifier/beat_spec_feedback.py": "9839bc6d19c9400ecccf4be1c8e2fbe900a2f268cc0e814875a38286aa9e4805",
    "judge/beat_spec.json": "82e5cd299d24968459ea5658e85b440c56062194dbbe3d141095e2c9b74a7c41",
    "judge/judge/workloads/manifest.json": "9cd2ba0d9e68c9b5f945d90c1948c26ac15afc33fb2d7884d2152a26083175ca",
    "datasets/train.json": "99c99eee6760054f3e60245f7f188b321f4e5f3b5ded28080abed8c33338590b",
    "datasets/eval.json": "c0bf7e56e798755cbedda1511341d3bab74f403c7d44ec2eae2a0e29e6a94ae2",
    "judge/reference.patch": "a26bbd6620f39e2af5f497515fb2bea55b093d885304080107d52a7d0c2655db",
}
# The complete solver-visible train allowlist besides the reserved aggregate
# beat_spec_progress; policy.json and judge/beat_spec.json must both declare it.
TRAIN_FEEDBACK = ["all_metrics_progress", "candidate_median_latency_us"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeContractFailure(message)


def check_source(root: Path, evaluation: dict) -> list[str]:
    require(root.name == "join_tree_rewriting", "unreviewed task without beat_spec")
    declared = evaluation.get("solver_visible_metrics")
    require(
        isinstance(declared, list) and sorted(declared) == sorted(TRAIN_FEEDBACK),
        "native train feedback must be exactly the reviewed aggregate allowlist",
    )
    # The task ships a formal judge/beat_spec.json that transcribes this verdict
    # (docs/task-authoring/join_tree_rewriting/beat-spec-v2.md). It is hash-pinned
    # in REVIEWED_FILES (so it is required to be present, a regular non-symlink
    # file, and byte-exact) together with the adapter and the feedback helper
    # that produce the aggregate train feedback, so the spec cannot drift from
    # the code. A substitute or forged spec is rejected by the hash pin.
    for name, expected in REVIEWED_FILES.items():
        path = root / name
        require(
            path.is_file()
            and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()),
            f"missing regular reviewed native source: {name}",
        )
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            f"unreviewed native protocol source: {name}",
        )
    manifest = json.loads((root / "judge/judge/workloads/manifest.json").read_text())
    expected = [f"dev/{i}" for i in range(1, 21)] + [f"final/{i}" for i in range(1, 20)]
    require(
        set(manifest) == set(expected),
        "native final must retain all 39 query identities",
    )
    for mode, count, partitions in (
        ("train", 20, ["dev"]),
        ("eval", 39, ["dev", "final"]),
    ):
        selector = json.loads((root / f"datasets/{mode}.json").read_text())
        require(
            selector["required_query_count"] == count
            and selector["workload_partitions"] == partitions
            and selector["mode"] == mode
            and selector["problem_id"] == root.name
            and selector["measurement_protocol"]
            == "per-query-exact-strict-median-binary-log10-v1",
            "native dataset must preserve the approved 20/39 protocol",
        )
    return expected


def load_reviewed_functions(root: Path) -> tuple[dict, dict]:
    """Compile only already hash-reviewed functions; no workload, imports or I/O.

    The comparison module uses only stdlib imports. The adapter is compiled from
    function/class/constant AST nodes; its evaluator loading and evidence codec
    are supplied by the bounded probe. The actual comparison and projection
    functions, payload parser, role mapping and the shipped feedback helper are
    exercised unchanged.
    """
    comparison = {
        "__name__": "preview_join_comparison",
        "__file__": str(root / "judge/query_measurements.py"),
    }
    source = root / "judge/query_measurements.py"
    exec(compile(source.read_text(), str(source), "exec"), comparison)
    source = root / "verifier/fcs2_verifier.py"
    parsed = ast.parse(source.read_text(), filename=str(source))
    nodes = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))
    ]
    adapter = {
        name: value for name, value in globals().items() if not name.startswith("__")
    }
    adapter.update(__name__="preview_join_adapter", __file__=str(source))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), adapter)
    # The adapter's module-level import is not an AST node compiled above; bind
    # the shipped, hash-reviewed helper it names. Compiled in memory so the
    # locked source tree gains no bytecode.
    source = root / "verifier/beat_spec_feedback.py"
    helper = {"__name__": "preview_join_feedback", "__file__": str(source)}
    exec(compile(source.read_text(), str(source), "exec"), helper)
    adapter["beat_spec_public_metrics"] = helper["public_metrics"]
    return comparison, adapter


def query_probe(
    comparison: dict,
    identity: str,
    *,
    candidate: float = 10.0,
    identical: bool = False,
    correct: bool = True,
) -> dict:
    correctness = {
        "durations_us": {m: [1.0] for m in comparison["CORRECTNESS_METHODS"]},
        "matches_original": {
            m: [correct if m == "Candidate" else True]
            for m in comparison["CORRECTNESS_METHODS"]
        },
    }
    timing = {
        "durations_us": {
            m: [
                100.0
                if m.startswith("Binary")
                else 20.0
                if m.startswith("Reference")
                else candidate
            ]
            * 5
            for m in comparison["TIMING_METHODS"]
        },
        "matches_original": {
            m: [correct if m.startswith("Candidate") else True] * 5
            for m in comparison["TIMING_METHODS"]
        },
    }
    return comparison["compare_query"](
        identity=identity,
        correctness=correctness,
        timing=timing,
        identical=identical,
        repetitions=5,
    )


def rejected(call, exception: type[Exception], message: str) -> None:
    try:
        call()
    except exception:
        return
    raise NativeContractFailure(message)


def check_comparison(
    comparison: dict, adapter: dict, expected: list[str]
) -> tuple[float, dict]:
    """Use real native median/score/comparison functions on deterministic probes."""
    queries = {q: query_probe(comparison, q) for q in expected}
    score, metrics = comparison["summarize_queries"](expected, queries)
    require(
        metrics["expected_case_count"] == 39
        and metrics["case_count"] == 39
        and metrics["beats_reference"] == 1
        and score == 50.0,
        "native comparison/score changed",
    )
    private = adapter["private_verdict"](metrics)
    keys = {q + "/" + m for q in expected for m in ("exact_result", "median_latency")}
    require(
        set(private["beat_spec_progress"]) == keys
        and private["aggregation"]["workload_count"] == 39
        and private["beats_reference"] == 1,
        "native private result loses required comparisons",
    )
    for key in keys:
        require(
            private["beat_spec_progress"][key]
            == {
                "passed": 1,
                "failed": 0,
                "total": 1,
                "pass_percent": 100,
                "fail_percent": 0,
            },
            "native private comparison projection changed",
        )
    require(
        len(json.dumps(private).encode()) < 16384
        and len(json.dumps(metrics).encode()) < 512 * 1024,
        "native full final evidence exceeds its actual bounds",
    )
    for kwargs in (
        {"candidate": 20.0},
        {"candidate": 25.0},
        {"identical": True},
        {"correct": False},
    ):
        changed = copy.deepcopy(queries)
        changed[expected[-1]] = query_probe(comparison, expected[-1], **kwargs)
        _, failed = comparison["summarize_queries"](expected, changed)
        require(
            failed["expected_case_count"] == 39
            and failed["beats_reference"] == 0
            and adapter["private_verdict"](failed)["beats_reference"] == 0,
            "native PASS must require correctness and strict fresh median on every query",
        )
    missing = dict(queries)
    missing.pop(expected[-1])
    rejected(
        lambda: comparison["summarize_queries"](expected, missing),
        comparison["MeasurementError"],
        "native comparison accepts missing final query",
    )
    failed = copy.deepcopy(metrics)
    failed["queries"][expected[-1]]["passed"] = False
    rejected(
        lambda: adapter["private_verdict"](failed),
        adapter["VerifierFailure"],
        "native comparison accepts inconsistent query verdict",
    )
    failed = copy.deepcopy(metrics)
    failed["beats_reference"] = 0
    rejected(
        lambda: adapter["private_verdict"](failed),
        adapter["VerifierFailure"],
        "native comparison accepts inconsistent overall verdict",
    )
    return score, metrics


def expected_train_metrics(score: float, queries: dict) -> dict:
    """Independent statement of the complete solver-visible train metrics."""
    total = len(queries)

    def counts(passed: int) -> dict:
        return {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "pass_percent": round(100.0 * passed / total, 2),
            "fail_percent": round(100.0 * (total - passed) / total, 2),
        }

    latencies = [q.get("candidate_median_us") for q in queries.values()]
    return {
        "score_unbounded": score,
        "beat_spec_progress": {
            "exact_result": counts(sum(q["correctness_pass"] for q in queries.values())),
            "median_latency": counts(sum(q["latency_pass"] for q in queries.values())),
        },
        "all_metrics_progress": counts(sum(q["passed"] for q in queries.values())),
        "candidate_median_latency_us": (
            None if None in latencies else statistics.median(latencies)
        ),
    }


def train_probes(comparison: dict, selected: list[str]) -> list[dict]:
    """All-pass, all-incorrect, mixed and partly unmeasured train suites."""
    mixed = {
        q: query_probe(
            comparison,
            q,
            candidate=10.0 + index if index % 3 else 25.0 + index,
            correct=index != 4,
        )
        for index, q in enumerate(selected)
    }
    unmeasured = {q: query_probe(comparison, q) for q in selected}
    unmeasured[selected[0]] = comparison["compare_query"](
        identity=selected[0],
        correctness=None,
        timing=None,
        identical=False,
        repetitions=5,
        failure="candidate_lowering_failed",
    )
    return [
        {q: query_probe(comparison, q) for q in selected},
        {q: query_probe(comparison, q, correct=False) for q in selected},
        mixed,
        unmeasured,
    ]


def check_projection(
    root: Path, comparison: dict, adapter: dict, expected: list[str]
) -> None:
    """Exercise the actual verifier result/evidence projection with native metrics.

    The heavy evaluator/codec are replaced, not the score, comparison, bounds,
    payload validation, mode mapping, or public/private projection. This is a
    format check, never a native runtime or measured baseline qualification.
    """
    previous = os.environ.get("FRONTIER_VERIFIER_SCRATCH")
    previous_role = os.environ.get("FRONTIER_SUBMISSION_ROLE")
    try:
        with tempfile.TemporaryDirectory(prefix="preview-join-contract-") as text:
            scratch = Path(text)
            os.environ["FRONTIER_VERIFIER_SCRATCH"] = text
            bundle = scratch / "bundle"
            bundle.mkdir()
            (bundle / "evaluator.py").write_text(
                "# bounded fixture, no workload execution\n"
            )
            # Stage the real beat_spec.json: the train branch reads its
            # solver_feedback allowlist through the shipped helper.
            shutil.copy2(root / "judge/beat_spec.json", bundle / "beat_spec.json")
            payload = scratch / "payload.tar"
            with tarfile.open(payload, "w", format=tarfile.USTAR_FORMAT) as archive:
                directory = tarfile.TarInfo("payload/")
                directory.type = tarfile.DIRTYPE
                directory.mode = 0o755
                archive.addfile(directory)
                patch = tarfile.TarInfo("payload/solution.patch")
                patch.size = 1
                archive.addfile(patch, io.BytesIO(b"x"))

            for mode, selected in (("train", expected[:20]), ("eval", expected)):
                probes = (
                    train_probes(comparison, selected)
                    if mode == "train"
                    else [
                        {q: query_probe(comparison, q, correct=correct) for q in selected}
                        for correct in (True, False)
                    ]
                )
                for queries in probes:
                    correct = all(q["correctness_pass"] for q in queries.values())
                    score, metrics = comparison["summarize_queries"](selected, queries)

                    def evaluate(_, *, metrics=metrics, mode=mode, score=score):
                        require(
                            os.environ.get("FRONTIER_SUBMISSION_ROLE")
                            == ("final" if mode == "eval" else "development"),
                            "native suite mapping changed",
                        )
                        return score, score, "bounded projection probe", metrics

                    adapter["load_evaluator"] = lambda _: evaluate
                    evidence = {}
                    adapter["write_evidence"] = (
                        lambda path, entries, *, evidence=evidence: evidence.update(
                            entries
                        )
                    )
                    result_path = scratch / "result.json"
                    result = adapter["verify"](
                        bundle=bundle,
                        entrypoint="payload/solution.patch",
                        payload=payload,
                        dataset=root / f"datasets/{mode}.json",
                        mode=mode,
                        result_path=result_path,
                        evidence_path=scratch / "evidence.tar.zst",
                    )
                    require(
                        json.loads(result_path.read_text()) == result,
                        "native serialized feedback changed",
                    )
                    require(
                        result["metrics"]
                        == (
                            expected_train_metrics(score, queries)
                            if mode == "train"
                            else {}
                        ),
                        "native public feedback is not exactly the reviewed aggregates",
                    )
                    visible = json.dumps(result["metrics"])
                    require(
                        not any(json.dumps(q) in visible for q in selected),
                        "native public feedback names a query",
                    )
                    require(
                        ("private_metrics" in result) == (mode == "eval"),
                        "native final private verdict missing or leaked to train",
                    )
                    if mode == "eval":
                        private = result["private_metrics"]
                        require(
                            private["beats_reference"] == int(correct)
                            and len(private["beat_spec_progress"]) == 78,
                            "native final projection must retain every real query comparison",
                        )
                    recorded = json.loads(evidence["metrics.json"])
                    require(
                        recorded["metrics"] == metrics
                        and recorded["dataset_sha256"]
                        == adapter["sha256_file"](root / f"datasets/{mode}.json")
                        and recorded["payload_sha256"]
                        == adapter["sha256_file"](payload),
                        "native evidence lost or altered raw measurements/identity bindings",
                    )
    finally:
        for name, value in (
            ("FRONTIER_VERIFIER_SCRATCH", previous),
            ("FRONTIER_SUBMISSION_ROLE", previous_role),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_native_contract(root: Path, evaluation: dict) -> None:
    expected = check_source(root, evaluation)  # Before executing any task-owned Python.
    comparison, adapter = load_reviewed_functions(root)
    check_comparison(comparison, adapter, expected)
    check_projection(root, comparison, adapter, expected)
