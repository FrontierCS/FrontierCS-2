#!/usr/bin/env python3
"""Evaluator for fast_kv_compaction (Attention Matching KV compaction, QuALITY + Qwen3-4B).

Shared verifier, trusted command test sandbox (task.toml [test_sandboxes.shard]): on the
CPU control image the shared verifier applies the patch (edit scope, forbidden_added) and calls
evaluate(tree); evaluate() sends each shard of the suite (3 articles) with the applied tree and
this judge to its own single-H100 runner of the verifier-gpu image through the shared adapter's
run_sandboxes (train 4 runners / 12 articles, final 8 / 24; leases 3600 s / 4500 s per runner),
where `evaluator.py --shard-run` runs harness.py on the applied tree and returns one bounded
JSON document. Candidates are compared article by article
with frozen reference targets (judge/targets/manifest.json, from calibrate.py).
Repetitions (run_local, warmup=True in both roles): final runs harness.py with 3 fresh-worker
repetitions after an untimed warmup, QA on each; development runs `--repeats 1`, also after an
untimed warmup, QA once. harness.py's docstring still says `--no-warmup` for development: it
is a measured file of judge/calibration_compatibility.json and stays byte-identical.

Scoring. Mode (time or bytes) follows the declared target size. Per article:
  accuracy = correct / questions,
  e = clip(0.5 + log10(r) / 6, 0, 1), r = target/candidate on the mode's axis (seconds or bytes)
      times min(1, the same ratio on the other axis),
  value = accuracy * (0.75 + 0.25 e);  raw_score = score_unbounded = 100 * mean value.
The reference scores 100 * mean(reference accuracy) * 0.875 (r = 1, e = 0.5).
Metrics (per article, beat_spec_workload_passes) qa_correct, compaction_seconds,
compact_cache_bytes: pass/fail per judge/gate.py decide(). beats_reference = 1 iff every article
passes every metric. A failed candidate run scores 0 with every article failing; missing targets,
a GPU mismatch or a runner failure raise (an unscored verification).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
import gate  # noqa: E402
from gate import TargetsUnavailable  # noqa: E402

TASK_DIR = _MODULE_DIR
# Private reference and trusted decoding tree: the unchanged upstream pipeline
# plus the reference's solver_entry.py. Never in an agent image.
REFERENCE_TREE = TASK_DIR / "reference_tree"
MANIFEST_PATH = TASK_DIR / "targets" / "manifest.json"
HARNESS = TASK_DIR / "harness.py"
SOLVER_ENTRY = "compaction/solver_entry.py"

# Frozen article split of the 24-article QuALITY subset (sorted ids, alternating),
# chosen before any model outcome was seen. Final grades both halves.
PUBLIC_ARTICLES = ("22073", "22218", "22462", "22579", "22867", "22876",
                   "22966", "23104", "23563", "30029", "31282", "31736")
HIDDEN_ARTICLES = ("22102", "22346", "22524", "22590", "22875", "22958",
                   "22967", "23160", "24290", "30035", "31599", "32665")

EVALUATION_SEED = 20261019
# Three articles per shard, including warmup, compaction and QA on each repetition. These are
# the per-runner leases (task.toml [test_sandboxes.shard] lease_seconds): final stays below the
# 5400-s evaluation timeout so that provisioning, upload and runner deletion fit inside it.
TRAIN_TIMEOUT_SECONDS = 3600
FINAL_TIMEOUT_SECONDS = 4500
# Single-H100 Verifier GPU instances per role; articles are sharded across them
# (shard()). evaluate() on the CPU control image runs each on a runner of the trusted command
# test sandbox SANDBOX through the shared GPU adapter.
SANDBOX = "shard"
INSTANCES = {"train": 4, "final": 8}
VERIFIER_GPUS = max(INSTANCES.values())
# Final and calibration: 3 fresh-worker repetitions after a warmup, QA on each.
# Development: one untimed warmup, then one repetition, QA once (a single, noisier sample).
FINAL_REPEATS = 3
# After the harness's own budget ends: time to write the shard result.
SHARD_TIMEOUT_GRACE_SECONDS = 60
TRAIN_REPEATS = 1
INFRA_EXIT_CODE = 75
# Bound of one shard document, and time kept back from the end of the shard step's limit (its
# FRONTIER_VERIFIER_LOCAL_DEADLINE) to write it.
MAX_SHARD_BYTES = 4 * 1024 * 1024
RESULT_MARGIN_SECONDS = 240
# The least a shard can run in: the harness's trusted setup (engine anchor, model load,
# isolation probes, prefill anchors), then the trusted QA it reserves for every stored cache
# (harness.QA_RESERVE_SECONDS per article and repetition). A runner whose lease, after its
# provisioning and upload, leaves less than this is the platform's fault; above it the
# budget is only capped. Historical shards took at most 640 s (train) and 1781 s (final).
MIN_SHARD_SETUP_SECONDS = 300
MAX_HARNESS_LOG_BYTES = 64 * 1024

SPEC_METRICS = ("qa_correct", "compaction_seconds", "compact_cache_bytes")

# Continuous score (absolute): per article, accuracy x (0.75 + 0.25 e), mean x 100.
ACCURACY_WEIGHT = 0.75
EFFICIENCY_WEIGHT = 0.25


def efficiency(mode: str, candidate: dict, target: dict) -> float:
    """e = clip(0.5 + log10(r) / 6, 0, 1): 0.5 at parity, 1 at 1000x, 0 at 1/1000x.

    r is the ratio target/candidate on the mode's axis (seconds in time mode,
    bytes in bytes mode), multiplied by min(1, the same ratio on the other axis).
    """
    if mode == gate.TIME_MODE:
        ratio = (target["seconds"] / candidate["seconds"]) * min(1.0, target["bytes"] / candidate["bytes"])
    elif mode == gate.BYTES_MODE:
        ratio = (target["bytes"] / candidate["bytes"]) * min(1.0, target["seconds"] / candidate["seconds"])
    else:
        raise ValueError("unknown comparison mode")
    if not ratio > 0:
        return 0.0
    return min(1.0, max(0.0, 0.5 + math.log10(ratio) / 6))


def absolute_score(mode: str, workloads, rows, targets: dict) -> float:
    """100 x mean over articles of (correct / questions) x (0.75 + 0.25 e)."""
    values = [row["correct"] / row["questions"]
              * (ACCURACY_WEIGHT + EFFICIENCY_WEIGHT * efficiency(mode, row, targets[workload]))
              for workload, row in zip(workloads, rows, strict=True)]
    return 100 * sum(values) / len(values)


def reference_absolute_score(workloads, targets: dict) -> float:
    """The same score for the reference itself (r = 1, e = 0.5) at its median correct counts."""
    values = [targets[workload]["correct"] / targets[workload]["questions"]
              * (ACCURACY_WEIGHT + EFFICIENCY_WEIGHT * 0.5) for workload in workloads]
    return 100 * sum(values) / len(values)
assert SPEC_METRICS == gate.METRICS
VISIBLE_METRICS = ("dev_qa_accuracy", "dev_compaction_seconds", "dev_compact_cache_bytes")

# The trust boundary is the worker process (see harness.py): submitted code runs only there, as
# an unprivileged user without the judge, the model that answers, the timer or the byte count.
# The early token check ("FRONTIER_|HARBOR_" in added lines) is task.toml forbidden_added.


class HarnessInfrastructureFailure(RuntimeError):
    """The GPU process failed for a reason unrelated to the submission."""


def _workloads(final: bool) -> tuple[str, ...]:
    return PUBLIC_ARTICLES + HIDDEN_ARTICLES if final else PUBLIC_ARTICLES


def _failed_metrics(workloads: tuple[str, ...], targets: dict) -> dict[str, Any]:
    """A failed candidate run: every article fails every metric, the score 0 is the margin's.
    The shared verifier's final projection (private_metrics) needs reference_score and margin."""
    reference = reference_absolute_score(workloads, targets)
    return {
        "beats_reference": 0,
        "reference_score": reference,
        "margin": -reference,
        "beat_spec_workload_passes": [
            {"workload": workload, "metrics": {name: False for name in SPEC_METRICS}}
            for workload in workloads
        ],
    }


def declared_target_size(root: Path) -> float:
    return gate.declared_target_size(root)


def load_manifest() -> dict:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetsUnavailable("target manifest is unreadable") from exc


def load_targets(workloads: tuple[str, ...]) -> tuple[dict[str, dict], str, float]:
    manifest = load_manifest()
    return (gate.validate_manifest(manifest, workloads), gate.manifest_device(manifest),
            gate.manifest_engine_anchor(manifest))


def shard(workloads: tuple[str, ...], instances: int) -> dict[int, tuple[str, ...]]:
    """Deterministic data-parallel layout: workload i runs on instance i mod instances.

    Every repetition and every QA pass of an article stays on its instance. The
    same layout is used by ordinary evaluation and by calibration.
    """
    if not 1 <= instances <= VERIFIER_GPUS:
        raise ValueError("invalid instance count")
    return {index: tuple(workloads[index::instances]) for index in range(instances) if workloads[index::instances]}


def run_local(tree: Path | None, workloads: tuple[str, ...], *, role: str = "solver",
              seed: int = EVALUATION_SEED, timeout: int, gpu: str = "0",
              repeats: int = FINAL_REPEATS, warmup: bool = True, budget: float | None = None) -> dict:
    """The single-GPU harness on this machine. Raises HarnessInfrastructureFailure or RuntimeError.

    `budget` (seconds, trusted setup included) bounds the submission inside the
    harness, so a slow submission fails as itself before the runner's grant ends.
    """
    with tempfile.TemporaryDirectory(prefix="fastkv-") as scratch:
        output = Path(scratch) / "rows.json"
        command = [sys.executable, str(HARNESS), "--role", role, "--seed", str(seed),
                   "--articles", ",".join(workloads), "--pristine", str(REFERENCE_TREE),
                   "--gpu", gpu, "--repeats", str(repeats), "--output", str(output)]
        if not warmup:
            command.append("--no-warmup")
        if budget is not None:
            command += ["--budget", f"{budget:.0f}"]
        if tree is not None:
            command += ["--patched", str(tree)]
        env = {key: value for key, value in os.environ.items() if not key.startswith("FRONTIER_SUBMISSION")}
        try:
            proc = subprocess.run(command, cwd=scratch, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("evaluation exceeded its time limit") from exc
        if proc.returncode == INFRA_EXIT_CODE:
            raise HarnessInfrastructureFailure(proc.stderr[-MAX_HARNESS_LOG_BYTES:].decode(errors="replace"))
        if proc.returncode != 0 or not output.is_file():
            tail = proc.stderr[-2000:].decode(errors="replace").strip().splitlines()
            raise RuntimeError(f"compaction run failed: {tail[-1][:160] if tail else 'no output'}")
        result = json.loads(output.read_text(encoding="utf-8"))
    rows = result.get("rows") if isinstance(result, dict) else None
    if not isinstance(rows, list) or [row.get("workload") for row in rows] != list(workloads):
        raise RuntimeError("compaction run did not cover the suite")
    return result


def combine_shards(workloads: tuple[str, ...], instances: int, results: dict[int, dict]) -> dict:
    """One result from per-instance results: each instance covered exactly its shard,
    all on one GPU model; engine-anchor deviations are diagnostic only."""
    layout = shard(workloads, instances)
    if set(results) != set(layout):
        raise HarnessInfrastructureFailure("instance results do not match the shard layout")
    rows_by_workload = {}
    for index, result in results.items():
        rows = result.get("rows") if isinstance(result, dict) else None
        if not isinstance(rows, list) or [row.get("workload") for row in rows] != list(layout[index]):
            raise RuntimeError("compaction run did not cover the suite")
        for row in rows:
            rows_by_workload[row["workload"]] = dict(row, instance=index)
    devices = {result.get("device") for result in results.values()}
    if len(devices) != 1 or not isinstance(next(iter(devices)), str):
        raise gate.MachineMismatch("instances report different or no GPU models")
    engines = {index: float(result["engine_anchor_seconds"]) for index, result in results.items()}
    middle = statistics.median(engines.values())
    for index, value in engines.items():
        if not math.isfinite(value) or value <= 0:
            raise gate.MachineMismatch(f"instance {index} engine anchor was not measured")
    return {"device": devices.pop(), "engine_anchor_seconds": middle,
            "instance_engine_anchor_seconds": {str(index): value for index, value in engines.items()},
            "instance_engine_anchor_deviations": {str(index): value / middle - 1.0 for index, value in engines.items()},
            "rows": [rows_by_workload[w] for w in workloads]}


def shard_command(index: int, instances: int, role: str, budget: int, root: str) -> list[str]:
    """What one Verifier GPU runner runs: this evaluator (the uploaded judge copy) on its shard of
    the suite, against the applied tree, writing one bounded JSON document. `root` is the
    sandbox's workdir on the runner, holding input/tree, input/judge and output/."""
    return ["/opt/frontier/runtime-python", f"{root}/input/judge/evaluator.py", "--shard-run",
            str(index), str(instances), role, f"{root}/input/tree", f"{root}/output/shard.json",
            str(int(budget))]


def _shard_document(result, index: int) -> dict:
    """The one JSON document a shard's runner returned; anything else is the platform's."""
    step = result.steps[0]
    if step.lost_session or step.timed_out or step.exit_code != 0:
        raise HarnessInfrastructureFailure(f"shard {index}: the trusted shard command did not complete")
    path = None if result.outputs is None else result.outputs / "output" / "shard.json"
    if path is None or not path.is_file():
        raise HarnessInfrastructureFailure(f"shard {index}: the runner returned no bounded shard document")

    def finite(constant):
        raise ValueError(f"non-finite JSON number {constant}")
    try:
        return json.loads(path.read_bytes().decode("utf-8"), parse_constant=finite)
    except ValueError as exc:
        raise HarnessInfrastructureFailure(f"shard {index}: the shard result is not one JSON document") from exc


def _run_shards(tree: Path, role: str, instances: int, seconds: int) -> dict[int, dict]:
    """Every shard on its own single-GPU runner of the trusted command test sandbox SANDBOX
    (task.toml [test_sandboxes.shard]); {shard: document}. Each runner receives the applied tree
    and this judge (with config.yaml and pristine/) and runs one step for the rest of its grant;
    the harness budgets itself by FRONTIER_VERIFIER_LOCAL_DEADLINE. The adapter owns runner
    creation, the upload, deadlines and termination; a runner failure is an infrastructure
    error, never a score."""
    # The shared adapter on the control image. Any of its errors is the platform's, never a score.
    from fcs2_verifier_gpu import (SandboxRun, SandboxStep, VerifierGpuError, run_sandboxes,
                                   sandbox_workdir)
    with tempfile.TemporaryDirectory(prefix="fastkv-shards-") as scratch:
        files, into = Path(scratch) / "files", Path(scratch) / "outputs"
        shutil.copytree(tree, files / "input" / "tree", symlinks=True)
        shutil.copytree(TASK_DIR, files / "input" / "judge", symlinks=True)
        into.mkdir()
        try:
            root = sandbox_workdir(SANDBOX)
            runs = [SandboxRun(SANDBOX, files,
                               [SandboxStep(shard_command(index, instances, role,
                                                          max(1, seconds - RESULT_MARGIN_SECONDS), root),
                                            None, {"TMPDIR": "/workspace"})],
                               ["output/shard.json"], MAX_SHARD_BYTES)
                    for index in sorted(shard(_workloads(role == "final"), instances))]
            results = run_sandboxes(runs, into=into)
        except VerifierGpuError as exc:
            raise HarnessInfrastructureFailure(str(exc)) from exc
        return {index: _shard_document(result, index) for index, result in enumerate(results)}


def run_harness(tree: Path, workloads: tuple[str, ...], *, final: bool, timeout: int) -> dict:
    """Measure the applied tree on INSTANCES[role] single-GPU runners (control Verifier)."""
    role = "final" if final else "train"
    results = _run_shards(tree, role, INSTANCES[role], timeout)
    shard_results = {}
    for index, document in results.items():
        if not isinstance(document, dict) or document.get("shard") != index:
            raise HarnessInfrastructureFailure("a shard returned a malformed document")
        if document.get("status") == "infrastructure":
            raise HarnessInfrastructureFailure(f"shard {index}: {str(document.get('message'))[:300]}")
        if document.get("status") == "submission":
            raise RuntimeError(str(document.get("message") or "compaction run failed")[:180])
        shard_results[index] = document.get("result")
    return combine_shards(workloads, INSTANCES[role], shard_results)


def minimal_shard_seconds(articles: int, final: bool) -> int:
    """Trusted setup plus the trusted QA of every cache: the least a shard of `articles` needs."""
    from harness import QA_RESERVE_SECONDS
    repeats = FINAL_REPEATS if final else TRAIN_REPEATS
    return MIN_SHARD_SETUP_SECONDS + QA_RESERVE_SECONDS * articles * repeats


def shard_run(index: int, instances: int, role: str, tree: Path, output: Path,
              budget: float | None = None) -> int:
    """On one GPU runner: measure this shard's articles on the applied tree."""
    final = role == "final"
    workloads = shard(_workloads(final), instances).get(index, ())
    document = {"shard": index, "instances": instances, "role": role}
    try:
        limit = FINAL_TIMEOUT_SECONDS if final else TRAIN_TIMEOUT_SECONDS
        budget = min(limit, budget) if budget else limit
        # The shard step's limit ends at FRONTIER_VERIFIER_LOCAL_DEADLINE (this runner's
        # monotonic clock: the rest of the runner's grant, which provisioning and upload already
        # spent part of, less the adapter's collect reserve); keep the margin to write the shard
        # document before it. The budget is what is left, capped by the configured one; only a
        # lease below the shard's minimal need is infrastructure.
        deadline = os.environ.get("FRONTIER_VERIFIER_LOCAL_DEADLINE")
        if deadline:
            available = float(deadline) - time.monotonic() - RESULT_MARGIN_SECONDS
            needed = minimal_shard_seconds(len(workloads), final)
            if available < needed:
                raise HarnessInfrastructureFailure(
                    f"the runner's lease leaves {available:.0f} s, below the {needed:.0f} s "
                    f"this shard needs at least")
            budget = min(budget, available)
        document["result"] = run_local(
            Path(tree), workloads, timeout=int(budget) + SHARD_TIMEOUT_GRACE_SECONDS,
            repeats=FINAL_REPEATS if final else TRAIN_REPEATS, warmup=True, budget=budget)
        document["status"] = "ok"
    except HarnessInfrastructureFailure as exc:
        document.update(status="infrastructure", message=str(exc)[-2000:])
    except Exception as exc:  # noqa: BLE001 - the submission's failure on this shard
        document.update(status="submission", message=(str(exc).strip() or type(exc).__name__)[:180])
    output.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(document, allow_nan=False, sort_keys=True)
    if len(body.encode()) > MAX_SHARD_BYTES:
        body = json.dumps({"shard": index, "instances": instances, "role": role,
                           "status": "infrastructure", "message": "shard result exceeds its bound"})
    output.write_text(body)
    return 0


def _complete_metrics(mode, workloads, rows, targets, score) -> dict[str, Any]:
    reference = reference_absolute_score(workloads, targets)
    passes, paired = [], []
    correct = questions = 0
    seconds, footprints = [], []
    for workload, row in zip(workloads, rows, strict=True):
        decisions = gate.decide(mode, row, targets[workload])
        passes.append({"workload": workload, "metrics": decisions})
        paired.append({"workload": workload, "candidate": row, "target": targets[workload]})
        correct += row["correct"]
        questions += row["questions"]
        seconds.append(row["seconds"])
        footprints.append(row["bytes"])
    return {
        "beat_spec_workload_passes": passes,
        "beats_reference": int(all(value for row in passes for value in row["metrics"].values())),
        "reference_score": reference,
        "margin": float(score) - reference,
        "_operator_mode": mode,
        "_operator_paired_workload_metrics": paired,
        "dev_qa_accuracy": float(correct / questions),
        "dev_compaction_seconds": sorted(seconds),
        "dev_compact_cache_bytes": sorted(footprints),
    }


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    """The shared verifier has applied the patch (edit scope, forbidden_added); `tree` is the
    starter with only the allowed files overlaid. Runs on the CPU control image."""
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    workloads = _workloads(final)
    # No frozen targets, no verdict for anyone: raise before touching the patch.
    targets, expected_device, expected_engine = load_targets(workloads)
    calibrated = targets
    try:
        root = Path(tree)
        mode = gate.mode_for(declared_target_size(root))
        result = run_harness(root, workloads, final=final,
                             timeout=FINAL_TIMEOUT_SECONDS if final else TRAIN_TIMEOUT_SECONDS)
        # Preserve per-instance scaling; speed drift is diagnostic, GPU identity is required.
        engines = result.get("instance_engine_anchor_seconds") or {"0": result.get("engine_anchor_seconds")}
        ratios = gate.check_machine(result["device"], expected_device, result["rows"], targets,
                                    engines, expected_engine)
        targets = gate.scale_targets(targets, ratios)
        rows = [dict(gate.candidate_row(row), workload=row["workload"]) for row in result["rows"]]
        relative = [gate.article_score(mode, row, targets[workload])
                    for workload, row in zip(workloads, rows, strict=True)]
        score = absolute_score(mode, workloads, rows, targets)
        metrics = _complete_metrics(mode, workloads, rows, targets, score)
        # The earlier reference-relative score, kept for operators only.
        metrics["reference_relative_score"] = max(0.0, min(100.0, sum(relative) / len(relative)))
        metrics["_operator_anchor_diagnostics"] = {
            workload: {"ratio": ratio, "deviation": ratio - 1.0,
                       "exceeds_diagnostic_band": abs(ratio - 1.0) > gate.MAX_ANCHOR_DEVIATION}
            for workload, ratio in ratios.items()}
        metrics["_operator_instance_engine_anchor_deviations"] = result.get("instance_engine_anchor_deviations", {})
        passed = sum(all(row["metrics"].values()) for row in metrics["beat_spec_workload_passes"])
        return (score, score,
                f"Scored {len(workloads)} articles in {mode} mode: {passed} beat the reference; {score:.2f}/100.",
                metrics)
    except (TargetsUnavailable, HarnessInfrastructureFailure, gate.MachineMismatch):
        raise
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, SyntaxError, json.JSONDecodeError) as exc:
        message = str(exc).strip() or "evaluation failed"
        return 0.0, 0.0, message[:180], _failed_metrics(workloads, calibrated)


def _main() -> int:
    """`evaluator.py --shard-run INDEX INSTANCES ROLE TREE OUTPUT [BUDGET]` on a GPU runner."""
    if len(sys.argv) in (7, 8) and sys.argv[1] == "--shard-run":
        _flag, index, instances, role, tree, output = sys.argv[1:7]
        if role not in INSTANCES:
            raise SystemExit("role must be train or final")
        budget = float(sys.argv[7]) if len(sys.argv) == 8 else None
        return shard_run(int(index), int(instances), role, Path(tree), Path(output), budget)
    raise SystemExit("usage: evaluator.py --shard-run INDEX INSTANCES ROLE TREE OUTPUT [BUDGET]")


if __name__ == "__main__":
    raise SystemExit(_main())
