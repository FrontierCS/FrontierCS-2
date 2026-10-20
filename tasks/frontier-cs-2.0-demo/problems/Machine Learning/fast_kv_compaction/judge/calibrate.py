#!/usr/bin/env python3
"""Operator-only calibration: produce judge/targets/manifest.json from real GPU runs.

Run inside the verifier-gpu image (it has the models, the pristine tree and
this judge directory). Sharded (recommended): one job per container and H100,
all in parallel, never two jobs on one GPU, then merge anywhere:

  python3 /opt/frontier/task/calibrate.py --job reference_1001 --gpu 0 --output /tmp/reference_1001.json
  ...                                     (one per name in REFERENCE_JOBS)
  python3 /opt/frontier/task/calibrate.py --merge /tmp/*.json --output /tmp/manifest.json

Serial (required reference jobs, one after another on one GPU):

  python3 /opt/frontier/task/calibrate.py --output /tmp/manifest.json --gpu 0

Each partial seals its original measurement, anchors and source identity. Merge
requires complete coverage for reference_1001, reference_1002 and reference_1003.
Uncompressed, selftest and paper variants are optional diagnostics; absent jobs
are recorded as not_run and do not block READY. Supplied diagnostic jobs must
still have complete, nonduplicated coverage and valid machine anchors. All
measurements retain their original seals and compatible source hashes.

Targets and scientific scoring are unchanged: reference medians, the strict
time/bytes comparisons and the existing accuracy allowance. Serial mode runs
only the three required reference seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
import shutil
import sys
import tempfile

JUDGE = Path(__file__).resolve().parent
sys.path.insert(0, str(JUDGE))
import evaluator  # noqa: E402
import gate  # noqa: E402

CALIBRATION_SEEDS = (1001, 1002, 1003)
# Paper variants of the reference: (name, solver_entry.py text substitutions).
DIAGNOSTIC_VARIANTS = (
    ("am_hak_repeat_prefill_queries", (('load_query_config("ss-plus-repeat")', 'load_query_config("repeat")'),)),
    ("am_hak_context_prefill_queries", (('load_query_config("ss-plus-repeat")', 'load_query_config("context-prefill")'),)),
    ("am_hak_target_size_0_05", (("TARGET_SIZE = 0.1", "TARGET_SIZE = 0.05"),)),
)
CALIBRATION_TIMEOUT_SECONDS = 6 * 3600


def _reference_tree() -> Path:
    """A disposable copy of the private reference tree (judge/reference_tree)."""
    tree = Path(tempfile.mkdtemp(prefix="fastkv-reference-")) / "tree"
    shutil.copytree(evaluator.REFERENCE_TREE, tree, symlinks=False)
    return tree


INSTANCES = evaluator.INSTANCES["final"]


def _run(tree: Path | None, articles, role: str, seed: int, gpu: str) -> dict:
    # One single-GPU instance of the final layout: same harness, repetitions, warmup.
    return evaluator.run_local(tree, tuple(articles), role=role, seed=seed,
                               timeout=CALIBRATION_TIMEOUT_SECONDS, gpu=gpu,
                               repeats=evaluator.FINAL_REPEATS, warmup=True)


def run_serially(jobs: list, runner) -> list:
    """One job at a time on the node: one harness and one worker, like an evaluation."""
    return [runner(job) for job in jobs]


def _variant_tree(reference: Path, substitutions) -> Path:
    tree = Path(tempfile.mkdtemp(prefix="fastkv-variant-")) / "tree"
    shutil.copytree(reference, tree, symlinks=True)
    entry = tree / evaluator.SOLVER_ENTRY
    text = entry.read_text(encoding="utf-8")
    for old, new in substitutions:
        if old not in text:
            raise SystemExit(f"variant substitution not found: {old}")
        text = text.replace(old, new)
    # copytree preserves the immutable image's mode; edit only our disposable copy.
    entry.chmod(entry.stat().st_mode | 0o200)
    entry.write_text(text, encoding="utf-8")
    return tree


def variant_verdict(manifest: dict, mode: str, rows: list[dict], articles: tuple[str, ...]) -> dict:
    targets = gate.validate_manifest(dict(manifest, status=gate.READY), articles)
    decisions = [gate.decide(mode, gate.candidate_row(row), targets[article])
                 for article, row in zip(articles, rows, strict=True)]
    return {
        "mode": mode,
        "beats_reference": all(all(row.values()) for row in decisions),
        "articles_passing": sum(all(row.values()) for row in decisions),
        "per_metric_passing": {name: sum(row[name] for row in decisions) for name in gate.METRICS},
        "rows": rows,
    }


def build_manifest(captures: dict[int, dict], uncompressed: list[dict],
                   articles: tuple[str, ...], provenance: dict) -> dict:
    seeds = sorted(captures)
    devices = {result["device"] for result in list(captures.values()) + uncompressed}
    if len(devices) != 1:
        raise SystemExit(f"calibration ran on mixed GPUs: {sorted(devices)}")
    workloads = {}
    for index, article in enumerate(articles):
        rows = [captures[seed]["rows"][index] for seed in seeds]
        plain = [run["rows"][index] for run in uncompressed]
        if any(row["workload"] != article for row in rows + plain):
            raise SystemExit("calibration rows are out of order")
        workloads[article] = {
            "questions": rows[0]["questions"],
            "reference_correct": [row["correct"] for row in rows],
            "reference_seconds": [row["seconds"] for row in rows],
            "reference_timed_seconds": [row["timed_seconds"] for row in rows],
            "reference_bytes": [row["bytes"] for row in rows],
            "uncompressed_correct": [row["correct"] for row in plain],
            "anchor_seconds": [row["anchor_seconds"] for row in rows],
            "reference_correct_per_repetition": [row["correct_per_repetition"] for row in rows],
        }
    return {
        "schema": gate.MANIFEST_SCHEMA,
        "status": gate.READY,
        "reference_target_size": gate.REFERENCE_TARGET_SIZE,
        "device": devices.pop(),
        "engine_anchor_seconds": [value for seed in seeds for value in
                                  (captures[seed].get("instance_engine_anchor_seconds")
                                   or {"0": captures[seed]["engine_anchor_seconds"]}).values()],
        "captures": [{"seed": seed} for seed in seeds],
        "evaluation_seed": evaluator.EVALUATION_SEED,
        "provenance": provenance,
        "workloads": workloads,
    }


def selftest(manifest: dict, rows: list[dict], articles: tuple[str, ...]) -> dict:
    targets = gate.validate_manifest(manifest, articles)
    decisions = [gate.decide(gate.TIME_MODE, gate.candidate_row(row), targets[article])
                 for article, row in zip(articles, rows, strict=True)]
    beats = all(all(row.values()) for row in decisions)
    qa_all = all(row["qa_correct"] for row in decisions)
    return {
        "seed": evaluator.EVALUATION_SEED,
        "reference_beats_reference": beats,
        "reference_clears_qa_everywhere": qa_all,
        "articles_clearing_time": sum(row["compaction_seconds"] for row in decisions),
        "rows": rows,
        "passed": (not beats) and qa_all,
    }


def _spread(values: list[float]) -> float:
    return (max(values) - min(values)) / statistics.median(values)


def anchor_residual(captures: dict[int, dict], articles: tuple[str, ...]) -> dict:
    """How well the anchor rescaling explains timing drift between serial captures.

    Each capture's per-article median time is divided by that capture's own
    anchor ratio (geometric mean of HF-prefill and engine ratios to the median
    over captures). The residual is the per-article spread of those rescaled
    times, compared with the reference's own repetition-to-repetition spread.
    Recorded as a diagnostic only: it never adds a review flag or holds the
    manifest (strict median time target, zero margin).
    """
    seeds = sorted(captures)
    prefill = {seed: sum(row["anchor_seconds"] for row in captures[seed]["rows"]) for seed in seeds}
    engine = {seed: captures[seed]["engine_anchor_seconds"] for seed in seeds}
    prefill_mid, engine_mid = statistics.median(prefill.values()), statistics.median(engine.values())
    ratio = {seed: math.sqrt((prefill[seed] / prefill_mid) * (engine[seed] / engine_mid)) for seed in seeds}
    residuals, rep_spreads = [], []
    for index, _article in enumerate(articles):
        rescaled = [captures[seed]["rows"][index]["seconds"] / ratio[seed] for seed in seeds]
        residuals.append(_spread(rescaled))
        rep_spreads.extend(_spread(captures[seed]["rows"][index]["timed_seconds"]) for seed in seeds)
    residual = statistics.median(residuals)
    rep_spread = statistics.median(rep_spreads)
    return {"capture_anchor_ratios": {str(seed): ratio[seed] for seed in seeds},
            "median_rescaled_residual": residual, "median_rep_to_rep_spread": rep_spread,
            "exceeds_half_rep_spread": residual > 0.5 * rep_spread}


def finalize_status(manifest: dict, full_suite: bool) -> dict:
    """Require full reference coverage; absent diagnostics never block READY."""
    reasons = []
    if not full_suite:
        reasons.append("smoke subset; not the full suite")
    if manifest.get("review_flags"):
        reasons.append("paper variant passes; maintainer decides (see review_flags)")
    if reasons:
        manifest["status"] = "HOLD"
        manifest["hold_reason"] = "; ".join(reasons)
    return manifest


PARTIAL_SCHEMA = "frontier.fast-kv-compaction-calibration-partial/v1"
REFERENCE_JOBS = tuple(f"reference_{seed}" for seed in CALIBRATION_SEEDS)
JOB_NAMES = (REFERENCE_JOBS
             + ("uncompressed", "selftest") + tuple(f"variant_{name}" for name, _ in DIAGNOSTIC_VARIANTS))
SOURCE_FILES = ("reference_tree/compaction/solver_entry.py", "gate.py", "harness.py", "compaction_worker.py", "wire.py",
                "anchor.py", "evaluator.py", "calibrate.py", "data/quality_subset.jsonl")


def judge_sources_sha256() -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update(name.encode() + b"\0" + (JUDGE / name).read_bytes() + b"\0")
    return digest.hexdigest()


def measurement_sources_sha256() -> str:
    proof = json.loads((JUDGE / "calibration_compatibility.json").read_text())
    source_layout = (JUDGE.parent / "verifier-gpu/Dockerfile").is_file()
    actual = dict(proof["files"])
    runtime_paths = {
        "verifier-gpu/fastkv-worker": JUDGE.parent / "fastkv-worker",
        "agent-base/frontier/task/runtime/requirements-cp312-linux-x86_64.lock":
            JUDGE.parent / "python-lock/requirements.lock",
    }
    for name in proof["files"]:
        if source_layout:
            path = JUDGE.parent / name
        elif name.startswith("judge/"):
            path = JUDGE / name.removeprefix("judge/")
        elif name in runtime_paths:
            path = runtime_paths[name]
        elif name in ("verifier-gpu/Dockerfile", "model-assets.json"):
            # Build-only provenance is bound by the published image digest;
            # the flattened image does not contain these two input manifests.
            continue
        else:
            raise SystemExit(f"no runtime measurement path for {name}")
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not source_layout:
        for name in ("compaction_worker.py", "wire.py", "gpu_probe.py", "isolation_probe.py"):
            if hashlib.sha256((JUDGE.parent / "worker" / name).read_bytes()).hexdigest() != proof["files"]["judge/" + name]:
                raise SystemExit("runtime worker differs from the compatibility proof")
    if actual != proof["files"]:
        raise SystemExit("calibration measurement files differ from the compatibility proof")
    return hashlib.sha256(_canonical(actual)).hexdigest()


def validate_capture_sources(bodies: list[dict]) -> dict:
    proof = json.loads((JUDGE / "calibration_compatibility.json").read_text())
    measured = measurement_sources_sha256()
    if measured != proof["measurement_sources_sha256"]:
        raise SystemExit("calibration measurement fingerprint differs")
    current = judge_sources_sha256()
    sources = sorted({body["judge_sources_sha256"] for body in bodies})
    for source in sources:
        if source == current:
            continue
        certified = proof["compatible_judge_sources"].get(source)
        if not certified or certified["measurement_sources_sha256"] != measured:
            raise SystemExit("calibration partials come from incompatible judge sources")
    return {"current_judge_sources_sha256": current, "capture_judge_sources_sha256": sources,
            "measurement_sources_sha256": measured,
            "legacy_sources": {source: proof["compatible_judge_sources"][source]
                               for source in sources if source != current}}


def _canonical(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def seal(body: dict) -> dict:
    return dict(body, sha256=hashlib.sha256(_canonical(body)).hexdigest())


def unseal(partial: dict) -> dict:
    body = {key: value for key, value in partial.items() if key != "sha256"}
    if partial.get("schema") != PARTIAL_SCHEMA or hashlib.sha256(_canonical(body)).hexdigest() != partial.get("sha256"):
        raise SystemExit(f"calibration partial {partial.get('job')!r} is corrupt or not a partial")
    return body


def job_spec(name: str) -> tuple[str, int, tuple]:
    """(role, seed, solver_entry substitutions) for one job; role None = uncompressed."""
    for seed in CALIBRATION_SEEDS:
        if name == f"reference_{seed}":
            return "solver", seed, ()
    if name == "uncompressed":
        return "uncompressed", CALIBRATION_SEEDS[0], ()
    if name == "selftest":
        return "solver", evaluator.EVALUATION_SEED, ()
    for variant, substitutions in DIAGNOSTIC_VARIANTS:
        if name == f"variant_{variant}":
            return "solver", evaluator.EVALUATION_SEED, substitutions
    raise SystemExit(f"unknown calibration job {name!r}; expected one of {', '.join(JOB_NAMES)}")


def run_job(name: str, gpu: str, articles: tuple[str, ...], provenance: dict, shard: int,
            runner=None, capture_articles: tuple[str, ...] | None = None) -> dict:
    """One shard (instance) of one job on one GPU, laid out exactly as a final
    evaluation (evaluator.shard over INSTANCES); returns a sealed partial result."""
    role, seed, substitutions = job_spec(name)
    layout = evaluator.shard(tuple(articles), INSTANCES)
    if shard not in layout:
        raise SystemExit(f"shard {shard} has no articles in this layout")
    selected = layout[shard] if capture_articles is None else tuple(capture_articles)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(layout[shard]):
        raise SystemExit("capture articles must be a nonempty subset of this final-layout shard")
    runner = runner or _run
    reference = _reference_tree()
    tree = _variant_tree(reference, substitutions) if substitutions else reference
    try:
        mode = gate.mode_for(gate.declared_target_size(tree)) if role == "solver" else None
        result = runner(tree if role == "solver" else None, selected, role, seed, gpu)
    finally:
        if tree is not reference:
            shutil.rmtree(tree.parent, ignore_errors=True)
        shutil.rmtree(reference.parent, ignore_errors=True)
    return seal({"schema": PARTIAL_SCHEMA, "job": name, "shard": shard, "instances": INSTANCES,
                 "role": role, "seed": seed, "mode": mode,
                 "articles": list(articles), "judge_sources_sha256": judge_sources_sha256(),
                 "captured_articles": list(selected),
                 "provenance": provenance, "result": result})


def _anchor_point(result: dict) -> tuple[float, float]:
    return sum(row["anchor_seconds"] for row in result["rows"]), float(result["engine_anchor_seconds"])


def _instance_point(result: dict) -> dict[str, tuple[float, float]]:
    """{instance: (summed HF-prefill anchors of its articles, its engine anchor)}."""
    engines = result.get("instance_engine_anchor_seconds") or {"0": result["engine_anchor_seconds"]}
    prefill: dict[str, float] = {}
    for row in result["rows"]:
        key = str(row.get("instance", 0))
        prefill[key] = prefill.get(key, 0.0) + float(row["anchor_seconds"])
    return {key: (value, float(engines[key])) for key, value in prefill.items()}


def normalize(result: dict, ratios: dict[str, float]) -> dict:
    """Times divided by the anchor ratio of the instance that measured them (raw kept)."""
    rows = []
    for row in result["rows"]:
        row = dict(row)
        ratio = ratios[str(row.get("instance", 0))]
        if "seconds" in row:
            row["raw_seconds"], row["seconds"] = row["seconds"], row["seconds"] / ratio
            row["raw_timed_seconds"] = row["timed_seconds"]
            row["timed_seconds"] = [value / ratio for value in row["timed_seconds"]]
        rows.append(row)
    return dict(result, rows=rows, anchor_ratios=ratios)


def combine_capture_parts(articles: tuple[str, ...], parts: list[dict]) -> dict:
    layout = evaluator.shard(articles, INSTANCES)
    keys = [part.get("shard") for part in parts]
    if (set(keys) == set(layout) and len(keys) == len(layout)
            and all([r["workload"] for r in p["result"]["rows"]] == list(layout[p["shard"]])
                    for p in parts)):
        return evaluator.combine_shards(articles, INSTANCES,
                                        {p["shard"]: p["result"] for p in parts})
    # A sealed historical single-article capture keeps its own measured anchor.
    # Treat each disjoint capture as an instance; never synthesize a new anchor.
    rows, engines, devices = {}, {}, set()
    for index, part in enumerate(parts):
        result = part["result"]
        captured = result["rows"]
        names = [row["workload"] for row in captured]
        if (not names or len(set(names)) != len(names) or not set(names) <= set(articles)
                or set(names) & set(rows)):
            raise RuntimeError("calibration capture has missing, duplicate or foreign articles")
        if names != part.get("captured_articles", list(evaluator.shard(tuple(part["articles"]), INSTANCES)[part["shard"]])):
            raise RuntimeError("calibration capture rows differ from its sealed coverage")
        engines[str(index)] = float(result["engine_anchor_seconds"])
        devices.add(result["device"])
        rows.update({row["workload"]: dict(row, instance=index) for row in captured})
    if set(rows) != set(articles):
        raise RuntimeError("calibration captures do not cover the suite")
    if len(devices) != 1:
        raise gate.MachineMismatch("captures report different GPU models")
    middle = statistics.median(engines.values())
    if any(not math.isfinite(v) or v <= 0 for v in engines.values()):
        raise gate.MachineMismatch("capture engine anchors were not measured")
    return {"device": devices.pop(), "engine_anchor_seconds": middle,
            "instance_engine_anchor_seconds": engines,
            "instance_engine_anchor_deviations": {key: value / middle - 1.0 for key, value in engines.items()},
            "rows": [rows[w] for w in articles]}


def merge(partials: list[dict], provenance: dict | None = None) -> dict:
    """Validate sealed partials and build the manifest from anchor-normalized times."""
    shards = [unseal(partial) for partial in partials]
    sources = validate_capture_sources(shards)
    articles = tuple(max((body["articles"] for body in shards), key=len))
    if any(not set(body["articles"]) <= set(articles) or body.get("instances") != INSTANCES for body in shards):
        raise SystemExit("calibration partials cover different articles or layouts")
    layout = evaluator.shard(articles, INSTANCES)
    keys = [(body["job"], row["workload"]) for body in shards for row in body["result"]["rows"]]
    present = {body["job"] for body in shards}
    jobs = [name for name in JOB_NAMES if name in present or name in REFERENCE_JOBS]
    expected = [(name, article) for name in jobs for article in articles]
    if sorted(keys, key=str) != sorted(expected, key=str):
        missing = sorted(set(expected) - set(keys), key=str)
        extra = sorted({key for key in keys if keys.count(key) > 1 or key not in expected}, key=str)
        raise SystemExit(f"calibration partials incomplete or duplicated: missing {missing}, unexpected {extra}")
    # Each job's instances combine exactly as an evaluation's do (layout, one GPU
    # model, engine anchors consistent across instances).
    bodies = []
    for name in jobs:
        parts = [body for body in shards if body["job"] == name]
        try:
            combined = combine_capture_parts(articles, parts)
        except (gate.MachineMismatch, evaluator.HarnessInfrastructureFailure, RuntimeError) as exc:
            raise SystemExit(f"job {name}: {exc}") from exc
        first = parts[0]
        bodies.append(dict(first, result=combined, shard=None))
    by_name = {body["job"]: body for body in bodies}
    devices = {body["result"]["device"] for body in bodies}
    if len(devices) != 1:
        raise SystemExit(f"calibration ran on mixed GPUs: {sorted(devices)}")
    # Reference point per instance: the reference captures' median prefill over that
    # instance's articles, and the median engine anchor over all their instances.
    reference_results = [by_name[f"reference_{seed}"]["result"] for seed in CALIBRATION_SEEDS]
    references = [{row["workload"]: row for row in result["rows"]} for result in reference_results]
    engine_mid = statistics.median(value[1] for result in reference_results
                                   for value in _instance_point(result).values())
    ratios = {}
    for name, body in by_name.items():
        ratios[name] = {}
        for key, (prefill, engine) in _instance_point(body["result"]).items():
            coverage = [row["workload"] for row in body["result"]["rows"]
                        if str(row.get("instance", 0)) == key]
            prefill_mid = statistics.median(sum(reference[w]["anchor_seconds"] for w in coverage)
                                           for reference in references)
            ratio = math.sqrt((prefill / prefill_mid) * (engine / engine_mid))
            ratios[name][key] = ratio
    normalized = {name: normalize(body["result"], ratios[name]) for name, body in by_name.items()}
    captures = {seed: normalized[f"reference_{seed}"] for seed in CALIBRATION_SEEDS}
    manifest = build_manifest(captures, [normalized["uncompressed"]] if "uncompressed" in normalized else [], articles,
                              provenance or by_name[JOB_NAMES[0]]["provenance"])
    manifest["uncompressed"] = {"status": "completed" if "uncompressed" in normalized else "not_run"}
    manifest["calibration_sources"] = sources
    manifest["calibration_captures"] = [
        {"job": p["job"], "shard": p["shard"], "sealed_sha256": sealed["sha256"],
         "judge_sources_sha256": p["judge_sources_sha256"],
         "articles": [row["workload"] for row in p["result"]["rows"]],
         "engine_anchor_seconds": p["result"]["engine_anchor_seconds"],
         "provenance": p["provenance"]} for p, sealed in zip(shards, partials)]
    manifest["job_anchor_ratios"] = ratios
    manifest["anchor_diagnostics"] = {
        name: {"instances": {
            key: {"ratio": ratio, "deviation": ratio - 1.0,
                  "exceeds_diagnostic_band": abs(ratio - 1.0) > gate.MAX_ANCHOR_DEVIATION}
            for key, ratio in values.items()},
            "engine_anchor_deviations": by_name[name]["result"].get("instance_engine_anchor_deviations", {})}
        for name, values in ratios.items()}
    manifest["selftest"] = (selftest(manifest, normalized["selftest"]["rows"], articles)
                            if "selftest" in normalized else {"status": "not_run"})
    manifest["diagnostic_variants"] = {
        name.removeprefix("variant_"): (variant_verdict(manifest, by_name[name]["mode"], normalized[name]["rows"], articles)
                                        if name in normalized else {"status": "not_run"})
        for name in JOB_NAMES if name.startswith("variant_")}
    manifest["anchor_residual"] = anchor_residual(
        {seed: by_name[f"reference_{seed}"]["result"] for seed in CALIBRATION_SEEDS}, articles)
    manifest["review_flags"] = [
        f"paper variant {name} passes beats_reference: decide on the gate or the reference configuration"
        for name, verdict in manifest["diagnostic_variants"].items() if verdict.get("beats_reference")]
    full = evaluator.PUBLIC_ARTICLES + evaluator.HIDDEN_ARTICLES
    return finalize_status(manifest, articles == full)


def _summary(manifest: dict) -> str:
    return json.dumps({
        "status": manifest["status"],
        "selftest": {key: value for key, value in manifest["selftest"].items() if key != "rows"},
        "diagnostic_variants": {name: {key: value for key, value in verdict.items() if key != "rows"}
                                for name, verdict in manifest["diagnostic_variants"].items()},
        "anchor_residual": manifest["anchor_residual"],
        "review_flags": manifest["review_flags"],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="0", help="the one GPU this instance uses")
    parser.add_argument("--shard", type=int, help=f"which of the {INSTANCES} final-layout instances this is")
    parser.add_argument("--job", choices=JOB_NAMES, help="run exactly one job and write a sealed partial")
    parser.add_argument("--merge", nargs="+", metavar="PARTIAL", help="merge sealed partials into the manifest")
    parser.add_argument("--articles", default="", help="smoke subset (manifest then stays HOLD)")
    parser.add_argument("--capture-articles", default="", help="measure only missing articles of this shard")
    parser.add_argument("--provenance", default="{}", help="JSON object: image digests, GPU type, operator")
    options = parser.parse_args()
    full = evaluator.PUBLIC_ARTICLES + evaluator.HIDDEN_ARTICLES
    articles = tuple(options.articles.split(",")) if options.articles else full
    provenance = json.loads(options.provenance)
    if options.job and options.merge:
        parser.error("--job and --merge are exclusive")
    if options.job:
        if options.shard is None:
            parser.error("--job needs --shard (one single-GPU instance per shard)")
        selected = tuple(options.capture_articles.split(",")) if options.capture_articles else None
        partial = run_job(options.job, options.gpu, articles, provenance, options.shard,
                          capture_articles=selected)
        Path(options.output).write_text(json.dumps(partial, sort_keys=True, allow_nan=False))
        print(json.dumps({"job": options.job, "sha256": partial["sha256"]}))
        return 0
    if options.merge:
        partials = [json.loads(Path(path).read_text()) for path in options.merge]
        manifest = merge(partials, provenance or None)
    else:  # serial: required reference jobs only, one at a time on this one GPU
        layout = evaluator.shard(articles, INSTANCES)
        manifest = merge(run_serially([(name, index) for name in REFERENCE_JOBS for index in layout],
                                      lambda job: run_job(job[0], options.gpu, articles, provenance, job[1])),
                         provenance)
    Path(options.output).write_text(json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False))
    print(_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
