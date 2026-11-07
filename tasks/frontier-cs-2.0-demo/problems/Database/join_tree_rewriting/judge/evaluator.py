"""Join-tree rewriting judge: the candidate's rewrite_plan, lowered and executed fresh.

The shared verifier applies the patch (task.toml [submission]: only
2phase_nsa/binary_plan/rewrite_policy.py), runs check_tree (the module's source policy,
content_policy on a violation) and then evaluate(tree). rewrite_plan runs in plan_builder, one
confined process per query (the shared isolate: no judge access beyond the builder and its
query).

Score: per query q, ten fresh observations per role (binary, reference, candidate);
S_q = median(binary) / median(candidate), L = 0 if S_q <= 1 else
100 log S_q / (log 10 + log S_q); raw_score = score_unbounded = mean of L over the 20 train
(dev) or 39 final queries, a failed, unmeasured or incorrect query contributing 0.

Metrics, one row per query in beat_spec_workload_passes:
  exact_result    the candidate returns exactly the original binary result in the correctness
                  pass and every timed observation (higher is better; passes when true).
  median_latency  candidate median latency vs the reference's in the same evaluation (lower is
                  better; passes when strictly below; a byte-identical reference plan fails).
beats_reference = 1 iff every query passes both. Train also returns all_metrics_progress and
candidate_median_latency_us (query_measurements.summarize_queries); reference_score is the same
formula on the reference (unmeasured queries count 0, reference_score_complete says so).
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
import isolate  # noqa: E402  (the shared launcher, beside edit_scope in the Verifier base)
from query_measurements import (CORRECTNESS_METHODS, TIMING_METHODS, MeasurementError,
                                compare_query, read_measurements, require_complete, summarize_queries)


JUDGE_DIR = TASK_DIR / "judge"
PRISTINE = TASK_DIR / "pristine"  # the starter repo; the shared verifier overlays the submission on a copy
EDITABLE = "2phase_nsa/binary_plan/rewrite_policy.py"
REPETITIONS = 5
ENGINE_TIMEOUT_SECONDS = 600
PLAN_TIMEOUT_SECONDS = 30
FORBIDDEN_CALLS = {
    "open", "exec", "eval", "compile", "input", "breakpoint", "__import__",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
}
FORBIDDEN_TEXT = ("/proc", "/sys", "frontier_submission", "plan_builder", "evaluator.py")
ALLOWED_IMPORT_ROOTS = {"typing", "binary_plan"}


def _suite_name() -> str:
    return "final" if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else "dev"


def _source_policy(tree: Path) -> tuple[bool, str]:
    try:
        source = (tree / EDITABLE).read_text(encoding="utf-8")
        parsed = ast.parse(source)
    except (OSError, SyntaxError, UnicodeError):
        return False, "the editable module is not valid Python source"
    lowered = source.lower()
    if any(fragment in lowered for fragment in FORBIDDEN_TEXT):
        return False, "the rewrite module inspects judge infrastructure"
    has_entry = False
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                root = name.lstrip(".").split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    return False, f"import of {root or 'an unknown module'} is not allowed"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return False, f"call to {node.func.id} is not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "dunder attribute access is not allowed"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "rewrite_plan":
            has_entry = True
    return (True, "") if has_entry else (False, "rewrite_plan(plan) is missing")


def check_tree(tree: str, pristine: str, changes: list) -> tuple[str, list[str]] | None:
    """Shared-verifier hook: the editable module's source policy (imports from `typing` and
    `binary_plan` only, no forbidden calls, dunder attributes or judge paths, and a
    `rewrite_plan` definition). A violation is the shared `content_policy` rejection."""
    ok, _reason = _source_policy(Path(tree))
    return None if ok else ("content_policy", [EDITABLE])


def _reference_tree(scratch: Path) -> Path:
    """The pristine starter with judge/reference.patch applied (trusted, judge-owned)."""
    tree = scratch / "reference-tree"
    shutil.copytree(PRISTINE, tree, symlinks=True)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch.resolve())}
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(TASK_DIR / "reference.patch")],
                   cwd=tree, check=True, capture_output=True, timeout=60, env=env)
    return tree


def _build_plans(tree: Path, workloads: Path, output: Path) -> tuple[bool, str, dict[str, str]]:
    """Rewrite/lower each original query with its original per-query budget.

    Return explicit failures even when a failed process leaves an output file.
    The caller distinguishes invalid references from failed candidates.
    """
    ok, reason = _source_policy(tree)
    if not ok:
        return False, reason, {}
    output.mkdir(parents=True, exist_ok=True)
    plans = sorted(workloads.glob("*.json"), key=lambda p: int(p.stem))
    if not plans:
        return False, "no workload plans", {}
    # The declared limit is per query. Each plan is rewritten and lowered in its
    # own process with its own budget, so a policy that is slow on one join size
    # loses that query and keeps the rest.
    budget = PLAN_TIMEOUT_SECONDS
    failures: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="plan-one-") as one_text:
        one = Path(one_text)
        for plan in plans:
            case = one / plan.stem
            case.mkdir()
            shutil.copy(plan, case / plan.name)
            # The rewrite module runs inside plan_builder: each query is a fresh confined
            # process (the shared isolate) that reads only the tree, the builder and this
            # query, and writes only its own new directories; nothing else of the judge
            # (other plans, workloads, data, measurements) is reachable, and whatever it
            # leaves running is killed before the next query.
            command = [
                sys.executable, "-I", str(JUDGE_DIR / "plan_builder.py"),
                "--app", str(tree), "--workloads", str(case),
                "--output", "{out}", "--upstream", str(JUDGE_DIR / "upstream"),
            ]
            proc = isolate.run(
                command, timeout_seconds=budget, outputs=[plan.name],
                read_paths=[tree, JUDGE_DIR / "plan_builder.py", JUDGE_DIR / "upstream", case],
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0",
                     "PYTHONDONTWRITEBYTECODE": "1"})
            if proc.timed_out:
                failures[plan.stem] = "timed out"
                continue
            result = proc.outputs[plan.name]
            if proc.returncode != 0 or proc.limit or result is None:
                failures[plan.stem] = "rewrite or lowering failed"
                continue
            (output / plan.name).write_bytes(result)
    if len(failures) == len(plans):
        return False, "plan rewriting failed on every query", failures
    return True, "", failures


def _case_catalog() -> dict[str, Path]:
    """Final retains every original public and hidden case, with distinct IDs."""
    suites = ("dev", "final") if _suite_name() == "final" else ("dev",)
    manifest = json.loads((JUDGE_DIR / "workloads" / "manifest.json").read_text())
    catalog = {}
    for suite in suites:
        files = sorted((JUDGE_DIR / "workloads" / suite).glob("*.json"), key=lambda p: int(p.stem))
        expected = {key for key in manifest if key.startswith(suite + "/")}
        actual = {suite + "/" + path.stem for path in files}
        if actual != expected or not actual:
            raise MeasurementError("workload files do not match the complete manifest")
        catalog.update((suite + "/" + path.stem, path) for path in files)
    return catalog


def _write_configs(catalog: dict[str, Path], references: dict[str, Path],
                   candidates: dict[str, Path], config_dir: Path) -> dict[str, bool]:
    identical = {}
    for phase in ("correctness", "timing"):
        (config_dir / phase).mkdir(parents=True)
    for index, (identity, ir) in enumerate(catalog.items()):
        ref, cand = references[identity], candidates[identity]
        identical[identity] = ref.read_bytes() == cand.read_bytes()
        phases = {
            "correctness": [("BinaryJoin", ir), ("Reference", ref), ("Candidate", cand)],
            # Preserve RefA/CandA/CandB/RefB order; bracket with fresh binary runs.
            "timing": [("BinaryA", ir), ("ReferenceA", ref), ("CandidateA", cand),
                       ("CandidateB", cand), ("ReferenceB", ref), ("BinaryB", ir)],
        }
        for phase, plans in phases.items():
            payload = {"query": identity, "plans": [
                {"method": method, "path": str(path.resolve())} for method, path in plans]}
            (config_dir / phase / f"{index:03d}.json").write_text(json.dumps(payload))
    return identical


def _run_engine(configs: Path, timings: Path, repetitions: int, expected: set[str],
                methods: tuple[str, ...]) -> dict:
    timings.write_text("duration(µs),method,variant,query\n", encoding="utf-8")
    audit = timings.with_suffix(".validation.jsonl")
    audit.write_text("")
    command = [str(JUDGE_DIR / "bin" / "join_runner"), "--configs", str(configs),
               "--data", str(JUDGE_DIR / "data"), "--timings-outfile", str(timings),
               "--repetitions", str(repetitions), "--validation-outfile", str(audit)]
    try:
        proc = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              timeout=ENGINE_TIMEOUT_SECONDS, env={"PATH": os.environ.get("PATH", "")})
    except subprocess.TimeoutExpired as error:
        # This is a cumulative phase deadline, not a per-method execution timer.
        # Even a pending Candidate cannot attribute prior baseline/reference cost.
        raise MeasurementError("cumulative engine deadline exhausted; budget calibration required") from error
    measurements = read_measurements(timings, audit, expected, methods, repetitions)
    if proc.returncode != 0:
        raise MeasurementError("reference, baseline or engine execution failed")
    require_complete(measurements, methods, repetitions)
    return measurements


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    """Measure all expected workloads; invalid reference never changes coverage."""
    candidate_tree = Path(tree)
    catalog = _case_catalog()
    expected = list(catalog)
    failures = {q: "candidate_not_measured" for q in expected}
    queries = {q: compare_query(identity=q, correctness=None, timing=None, identical=False,
                                repetitions=REPETITIONS, failure=failures[q]) for q in expected}
    with tempfile.TemporaryDirectory(prefix="statsceb-eval-") as temp_text:
        temp = Path(temp_text)
        try:
            reference_tree = _reference_tree(temp)
        except (OSError, subprocess.SubprocessError) as error:
            raise MeasurementError("reference preparation failed") from error
        plans = {"reference": {}, "candidate": {}}
        for suite in dict.fromkeys(q.split("/")[0] for q in expected):
            workloads = JUDGE_DIR / "workloads" / suite
            for label, tree in (("reference", reference_tree), ("candidate", candidate_tree)):
                output = temp / label / suite
                built, build_reason, failed = _build_plans(tree, workloads, output)
                required = {q for q in expected if q.startswith(suite + "/")}
                if label == "reference" and (not built or failed):
                    raise MeasurementError("reference lowering failed for a required workload")
                for q in required:
                    stem = q.split("/")[1]
                    path = output / (stem + ".json")
                    # An output left by a failed/timeout process is not valid.
                    if built and stem not in failed and path.is_file():
                        plans[label][q] = path
                    elif label == "reference":
                        raise MeasurementError("reference omitted a required workload")
                    else:
                        failures[q] = "candidate_lowering_failed"
        live = {q: path for q, path in catalog.items() if q in plans["candidate"]}
        if live:
            identical = _write_configs(live, plans["reference"], plans["candidate"], temp / "configs")
            correctness = _run_engine(temp / "configs/correctness", temp / "correctness.csv",
                                      1, set(live), CORRECTNESS_METHODS)
            timing = _run_engine(temp / "configs/timing", temp / "timings.csv",
                                 REPETITIONS, set(live), TIMING_METHODS)
            for q in live:
                queries[q] = compare_query(identity=q,
                    correctness=correctness["queries"][q], timing=timing["queries"][q],
                    identical=identical[q], repetitions=REPETITIONS)
        for q in set(expected) - set(live):
            queries[q]["status"] = failures[q]
        score, metrics = summarize_queries(expected, queries)
        metrics["repetitions"] = REPETITIONS
        metrics["timing_order"] = list(TIMING_METHODS)
        metrics["correctness_order"] = list(CORRECTNESS_METHODS)
        return score, score, f"query evaluation complete; {metrics['case_count']} of {len(expected)} queries measured", metrics
