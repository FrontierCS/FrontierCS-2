"""Bounded compatibility check for the reviewed native vLLM serving protocol.

This source-bound check is not a measurement or acceptance receipt. The task's
own CPU Verifier (``verifier/verify.py``) projects the private per-group
reference verdict into ``beat_spec_progress`` and ``beats_reference``; train
feedback exposes the exact reviewed public candidate aggregate whitelist. The formal ``judge/beat_spec.json`` transcribes exactly
those decisions and is hash-pinned here together with the verdict, projection
and scoring sources, so neither can drift from the other. The probe below
executes only the hash-reviewed projection functions on synthetic operator
results; no workload, GPU, model, patch or provider operation is involved.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

PROBLEM_ID = "vllm_llm_serving_optimization"
SOLVER_FEEDBACK = [
    'swe_public_geometric_mean_latency_seconds',
    'swe_public_speedup_over_vanilla',
    'swe_public_p90_latency_seconds',
    'swe_public_p95_latency_seconds',
    'swe_public_p99_latency_seconds',
    'swe_public_correct',
    'bfcl_public_geometric_mean_latency_seconds',
    'bfcl_public_speedup_over_vanilla',
    'bfcl_public_p90_latency_seconds',
    'bfcl_public_p95_latency_seconds',
    'bfcl_public_p99_latency_seconds',
    'bfcl_public_correct',
]


class NativeContractFailure(ValueError):
    pass


# Reviewed task sources. Hashes guard execution of task-owned Python and
# freeze the transcribed beat spec: projection and scoring code, the instance
# manifest, the datasets and the fixed-target release binding. Any change to a
# pinned file needs a new review and a re-pinned digest here.
REVIEWED_FILES = {
    'verifier/cpu_reference_verdict.py': '45b8de1070d36f42e9974debba0b90e71279d01b8fb60a4716461a13b6f5a104',
    'verifier/cpu_batch_measurements.py': '634dbc7da79b2f215429e7dc114a32418deeeac4693950e203183e889de46e58',
    "verifier/verify.py": "91a64f2a16981e9ae44582fe7d2d4c613550dbec1e82d099097a9e0c2607f8d5",
    "judge/reference_verdict.py": "e99952a4de5005ed3fcd2934fdf296f67a65bedd3e068fec38fc4efde06a29b9",
    "judge/batch_measurements.py": "96dad163185c2874251d4eac842750112e27f584263ea44ad4374545e7f3ee84",
    "judge/latency.py": "8301a579b7719108421098e9b80d5169bc537c5864bdd129310431af86b274d8",
    "judge/continuous.py": "7dac3a575912a638ae2114326b3a4a0002cbba5f9fa92aa1e326ff6640b6dbb6",
    "judge/instance_manifest.py": "fd1e031304f668a4e92f0c25d1cff68f20dfbdb98fe17c7c2d71d4ebf50fb314",
    "judge/instance_manifest.json": "9c536e1d371d2b0dd01875a755bcdacd649d73400374a4f497b3a3c1f12fea92",
    "judge/beat_spec.json": "902ea66cc965e5ba86ac8623ff1ee711cd51e735f69a88efdc77594eb1920b81",
    "judge/patch_policy.py": "f744a6a0251590c3578d3be03de2f43a4ec0b6c6b9bb3cbfeb6070235340af54",
    "judge/submission.py": "8aa39142aa3b6f85028cd08df41cab77d3feac080bddb3c6012740dd4dc05e33",
    "judge/reference.patch": "65aa904321e855d45d3f3b5a0fd92e7f687d729a4a0ff66b56eb2bdecfe551b9",
    "datasets/train.json": "d01bc3509809825288794d7ce5f526768e6953b10db5a6c310735dc6a3115b7b",
    "datasets/eval.json": "4a46f566c9ec8ffbda6896198f03d603d6655dff22fc276ae34db99908f9e618",
    'judge/fixed_targets.py': 'd00232b94255d8526f05e38423c05b823456fa6c24cf34ca4a25a553f469cd81',
    'judge/fixed_targets.schema.json': '91f1913b4841534e41977d85b9492d3396c2186e67df8e933281f655b3550ddc',
    'judge/controller_audit_plan.py': 'd1c371d3ff6e3863f949393f5f450ebac6adee8119111c69ba9c3f356a2fe5ca',
    'judge/accepted_controller_audit.py': '75401ce5179d100f38a8016382a4e37acc63bb01c52b5357b7ed49c6506525d6',
    'judge/sidecar_catalog.py': '9c12f9997a7af54ff918ea033a4015db64c8a91923cfbdf8c0103add3f4be1d1',
    'judge/scheduler_service.py': '41c4cc4df7cb84a398bdccf3fbff3b35861440421a48f7ba945d2e66d446e2a0',
    'judge/serving_eval/swe_correctness_service.py': 'df1329b4451503620ac8af54a7e964acd1325060148c48bb3392f5c29ee654f7',
    'verifier/fixed_target_release.py': '73fa09abda6078202c13851e70bf99df3198e61437064f741eb716a2cc644ca4',
    'verifier/parallel_phase.py': '3aee219f8966231125d5d346c9b733a55e684f5e9a8a156436ae10cc06cad3fe',
    'verifier/phase_run.py': 'c873ea7cad09b8410d9e71027e10c136cc0e2cd06de891e58bb3f5ecc460dc98',
    'verifier/runtime_config.py': 'f100971e896c717f421e7b0d1401119fa358dd4b29c4e0f008732c5b563edc9f',
    'verifier/runtime-config.json': '988de61a37d542bbc32c67b17dee518ff0c09e9a30e27dc0a54958666e8ee0b1',
    'verifier/runtime_plan.py': '6440b36dd4003b424a465cfd82334d6a38e6a36d8f77890bb4e9f36e908c527e',
    'verifier/provisioning_owner.py': '9875f8686c278f6b93242cfb21ef4fbf95326fef092c0bc9f89837db14dc7ef3',
    'verifier/gpu_session.py': '1a6858f28602ff2f34f43d7cb941275c6d4960d6dc6fb4fbd53adc185d92173d',
    'verifier/batch_transport.py': '1fbe7f00c7439e41e680b0eab5c126c92e079a2d16696051863c7eb0170f8f11',
}
METRICS = ["paired_timing", "p90", "p95", "p99", "correctness"]
COMPARISONS = METRICS[:5]
GROUPS = {"train": {"swe/public": 5, "bfcl/public": 8},
          "eval": {"swe/public": 48, "swe/hidden": 48, "bfcl/public": 155, "bfcl/hidden": 155}}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeContractFailure(message)


def check_source(root: Path, evaluation: dict) -> None:
    require(root.name == PROBLEM_ID, "unreviewed task without a native vLLM contract")
    require(
        evaluation.get("solver_visible_metrics") == SOLVER_FEEDBACK,
        "native train feedback must match the exact public whitelist",
    )
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
    manifest = json.loads((root / "judge/instance_manifest.json").read_text())
    for mode, groups in GROUPS.items():
        selector = json.loads((root / f"datasets/{mode}.json").read_text())
        require(
            selector.get("schema") == "frontier.vllm-dataset/v1"
            and selector.get("problem_id") == PROBLEM_ID
            and selector.get("mode") == mode
            and selector.get("swe_instance_ids") == manifest[mode]["swe"]
            and selector.get("bfcl_instance_ids") == manifest[mode]["bfcl"]
            and selector.get("rate_groups") == (["public"] if mode == "train" else ["public", "hidden"]),
            f"native {mode} dataset must select the complete original phase",
        )
        for workload in ("swe", "bfcl"):
            ids = manifest[mode][workload]
            require(
                len(ids) == groups[f"{workload}/public"] and len(set(ids)) == len(ids),
                "native instance manifest differs from the original fixed case counts",
            )
    for workload in ("swe", "bfcl"):
        public = manifest["train"][workload]
        require(
            manifest["eval"][workload][: len(public)] == public,
            "final must preserve the original public prefix",
        )
    spec = json.loads((root / "judge/beat_spec.json").read_text())
    require(
        [entry.get("name") for entry in spec.get("metrics", [])] == METRICS,
        "beat_spec.json metrics differ from the native verifier's per-group decisions",
    )
    require([entry.get("name") for entry in spec.get("solver_feedback", [])] == SOLVER_FEEDBACK,
            "native beat spec feedback must match the exact public whitelist")
    require(
        spec.get("beats_reference_aggregation") == {"kind": "conjunction"},
        "native beats_reference is the conjunction of every group decision",
    )


def load_reviewed_functions(root: Path) -> dict:
    """Compile only already hash-reviewed projection functions; no workload or I/O.

    ``verify.py`` is compiled from its function and class definitions, so its
    judge imports (evidence packaging, GPU orchestration) are never resolved.
    ``expected_groups`` comes from the reviewed instance manifest module, which
    reads only its own pinned JSON file.
    """
    manifest = {"__name__": "native_vllm_manifest", "__file__": str(root / "judge/instance_manifest.py")}
    source = root / "judge/instance_manifest.py"
    exec(compile(source.read_text(), str(source), "exec"), manifest)
    source = root / "verifier/verify.py"
    parsed = ast.parse(source.read_text(), filename=str(source))
    nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = {"__name__": "native_vllm_verify", "__file__": str(source),
                 "json": json, "math": __import__("math"), "hashlib": hashlib,
                 "expected_groups": manifest["expected_groups"]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def operator_result(mode: str, *, failing: dict | None = None, score: float = 12.5) -> dict:
    """A complete operator verdict in the shape ``reference_verdict`` emits."""
    failing = failing or {}
    groups, gates = {}, {}
    for group, count in GROUPS[mode].items():
        comparisons = {name: name not in failing.get(group, ()) for name in COMPARISONS}
        groups[group] = {"complete": True, "expected_instances": count,
                         "comparisons": comparisons, "pass": all(comparisons.values())}
    continuous = {}
    if mode == "train":
        for workload in ("swe", "bfcl"):
            groups[workload + "/public"].update(
                candidate_geometric_mean_latency_seconds=2.0,
                candidate_tails={"p90": 3.0, "p95": 4.0, "p99": 5.0},
                candidate_correct=GROUPS[mode][workload + "/public"],
                reference_tails={"p90": 99999}, required_correct_count=99999)
            continuous[workload] = {"groups": {"public": {
                "speedup": 1.5, "timing_complete": True}}}
    reference_pass = all(group["pass"] for group in groups.values())
    valid = all(gate["pass"] for gate in gates.values())
    return {"score": score, "pass": valid and reference_pass, "valid": valid,
            "bfcl_flip_gate_enabled": False, "continuous": continuous,
            "reference_verdict": {"pass": reference_pass, "groups": groups},
            "validity": {"pass": valid, "groups": gates}}


def check_projection(functions: dict) -> None:
    """The reviewed projection transcribes every group decision, and nothing else."""
    projection = functions["result_projection"]
    train = projection(operator_result("train"), "train")
    expected_metrics = {"beat_spec_progress": {name: {
        "passed": 2, "failed": 0, "total": 2, "pass_percent": 100.0, "fail_percent": 0.0}
        for name in COMPARISONS}}
    for workload in ("swe", "bfcl"):
        for metric, value in (("geometric_mean_latency_seconds", 2.0),
                ("speedup_over_vanilla", 1.5), ("p90_latency_seconds", 3.0),
                ("p95_latency_seconds", 4.0), ("p99_latency_seconds", 5.0),
                ("correct", GROUPS["train"][workload + "/public"])):
            expected_metrics[workload + "_public_" + metric] = value
    require(
        train == {"status": "passed", "raw_score": 12.5, "summary": "Verification completed.",
                  "metrics": expected_metrics},
        "train feedback must expose exactly the reviewed public aggregates and check counts",
    )
    complete = projection(operator_result("eval"), "eval")
    private = complete.get("private_metrics") if isinstance(complete, dict) else None
    progress = private.get("beat_spec_progress") if isinstance(private, dict) else None
    require(
        isinstance(progress, dict)
        and private.get("beats_reference") == 1
        and private.get("aggregation") == {"kind": "conjunction"}
        and list(progress) == METRICS
        and all(progress[name] == {"passed": 4, "failed": 0, "total": 4, "pass_percent": 100.0, "fail_percent": 0.0}
                for name in COMPARISONS),
        "complete final projection must count every group decision",
    )
    partial = projection(operator_result("eval", failing={"swe/hidden": ("p99",), "bfcl/public": ("correctness",)}),
                         "eval")
    private = partial.get("private_metrics", {})
    progress = private.get("beat_spec_progress", {})
    require(
        private.get("beats_reference") == 0
        and progress["p99"] == {"passed": 3, "failed": 1, "total": 4, "pass_percent": 75.0, "fail_percent": 25.0}
        and progress["correctness"]["failed"] == 1
        and progress["paired_timing"]["passed"] == 4,
        "a failed group decision must stay visible in the projection",
    )
    missing = operator_result("eval")
    del missing["reference_verdict"]["groups"]["swe/hidden"]
    missing["pass"] = False
    missing["reference_verdict"]["pass"] = False
    require(
        "beat_spec_progress" not in projection(missing, "eval").get("private_metrics", {}),
        "incomplete evidence must not become failed comparisons",
    )
    passing_without_evidence = operator_result("eval")
    del passing_without_evidence["reference_verdict"]["groups"]["bfcl/hidden"]
    try:
        projection(passing_without_evidence, "eval")
    except ValueError:
        pass
    else:
        raise NativeContractFailure("a passing verdict without complete comparison evidence must be rejected")
    inconsistent = operator_result("eval")
    inconsistent["reference_verdict"]["groups"]["swe/public"]["pass"] = False
    try:
        projection(inconsistent, "eval")
    except ValueError:
        pass
    else:
        raise NativeContractFailure("an inconsistent group conjunction must be rejected")
    for score in (float("nan"), 101, -1, True):
        try:
            projection({"score": score, "pass": False}, "eval")
        except ValueError:
            continue
        raise NativeContractFailure("scores outside the 0-100 contract must be rejected")


def check_native_contract(root: Path, evaluation: dict) -> None:
    check_source(root, evaluation)
    check_projection(load_reviewed_functions(root))
