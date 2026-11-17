"""Judge of clusterwise_spgemm (fixed-length clustered SpGEMM) for the shared verifier.

Score: G = geometric mean over the suite's 8 workloads of reference / candidate median time
(7 interleaved fresh-process pairs per workload); raw_score = score_unbounded =
100 G / (1 + G) (computed overflow-safe, below 100; 50 at parity). Train runs the 8 development
matrices, final 8 matrices of disjoint parents. In the candidate phase a build, link-guard,
correctness, wall/CPU/memory or workspace failure scores 0 with every workload failing; the
overall deadline, reference failures (including its 5 % MAD / odd-even drift noise gates) and
judge faults raise.

Beat-spec metrics (beat_spec_workload_passes, one row per workload):
- execution_time_ms, lower is better: passes when candidate median < 0.85 x reference median.
- throughput_mflops, higher is better: passes when flop / candidate > (flop / reference) / 0.85.
beats_reference = 1 iff every workload passes both.

The shared verifier has already applied the patch (edit_scope, task.toml [submission]);
`evaluate(tree)` gets the pristine starter with only the allowed files overlaid. The judge builds
its reference tree itself, inside its scratch and deadline.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

from product_oracle import Expected
from dataset_loader import install
from runtime_context import CONFIG, LIMITS, ROOT, Context, ResourceCeiling, require_disk_capture
from trusted_runner import (TrustedAggregateDeadline, build, build_oracle, preprocess, remaining,
                            sample)


BEAT_METRICS = ("execution_time_ms", "throughput_mflops")
# The starter tree (repo/), copied here by the Verifier image.
PRISTINE = ROOT / "pristine"
EDITABLE = tuple(CONFIG["edit_scope"]["allowed_paths"])


def _reference_tree(scratch, deadline):
    """A copy of the pristine starter with reference.patch applied (trusted, judge-owned), in the
    same layout as the judged tree: scratch/reference/tree."""
    tree = Path(scratch) / "reference" / "tree"
    shutil.copytree(PRISTINE, tree, symlinks=True)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(Path(scratch).resolve()))
    timeout = max(1.0, min(60.0, remaining(deadline, "patch application", reserve=5)))
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(ROOT / "reference.patch")],
                   cwd=tree, check=True, capture_output=True, timeout=timeout, env=env)
    return tree


def _editable_files(tree):
    found = {}
    for path in sorted(Path(tree).rglob("*")):
        name = path.relative_to(tree).as_posix()
        if path.is_file() and any(name == p or (p.endswith("/") and name.startswith(p)) for p in EDITABLE):
            found[name] = hashlib.sha256(path.read_bytes()).digest()
    return found


def _failure_workloads(final):
    """Return the frozen suite identities for a bounded gate-failure result.

    A gate can fail before the normal workload loop has produced any rows.  The
    operator result still needs a complete, auditable denominator rather than
    an empty beat-spec projection (which Management correctly treats as no
    verdict).  This helper deliberately does not validate inputs: the normal
    suite validation remains the authority for a successful evaluation.
    """
    manifest = json.loads((ROOT / "workloads.json").read_text())
    return manifest["final" if final else "dev"]


def _failure_metrics(final, *, reason, completed_workloads, runtime):
    """Build a private scored-zero result for a scientific gate failure.

    Every fixed workload/metric is false because the whole suite was not
    established.  The bounded diagnostic stays in evidence; the shared
    verifier projects the valid beat aggregate to private Management metrics.
    """
    rows = [{"workload": row["id"], "metrics": {name: False for name in BEAT_METRICS}}
            for row in _failure_workloads(final)]
    diagnostic = {
        "kind": "scientific_gate_failure",
        "reason": str(reason)[:512],
        "completed_workloads": max(0, int(completed_workloads)),
        "runtime": runtime,
    }
    return {
        "beats_reference": 0,
        "beat_spec_workload_passes": rows,
        "reference_score": 50.0,
        "margin": -50.0,
        "private_gate_diagnostic": diagnostic,
    }


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _continuous_score(candidate, reference):
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("incomplete measurements")
    if any(not math.isfinite(x) or x <= 0 for x in [*candidate, *reference]):
        raise ValueError("invalid measurements")
    log_speedup = statistics.fmean(math.log(r) - math.log(c)
                                  for c, r in zip(candidate, reference))
    # Evaluate the logistic curve without overflowing finite timing ratios.
    if log_speedup >= 0:
        score = 100.0 / (1.0 + math.exp(-log_speedup))
    else:
        speedup = math.exp(log_speedup)
        score = 100.0 * speedup / (1.0 + speedup)
    # Floating-point rounding must not turn a finite speedup into full score.
    return min(score, math.nextafter(100.0, 0.0))


def noise(values):
    median = statistics.median(values)
    if not math.isfinite(median) or median <= 0:
        raise ValueError("invalid clock values")
    mad = statistics.median(abs(x-median) for x in values) / median
    drift = abs(statistics.median(values[::2]) / statistics.median(values[1::2]) - 1)
    if (mad > CONFIG["benchmark"]["reference_max_relative_mad"] or
            drift > CONFIG["benchmark"]["reference_max_median_drift"]):
        raise ValueError("reference timing noise gate failed")
    return {"relative_mad": mad, "odd_even_relative_drift": drift}


def _workloads(final):
    manifest = json.loads((ROOT / "workloads.json").read_text())
    if len(manifest["dev"]) != 8 or len(manifest["final"]) != 8:
        raise ValueError("incomplete suites")
    if {r["matrix_sha256"] for r in manifest["dev"]} & {r["matrix_sha256"] for r in manifest["final"]}:
        raise ValueError("suite collision")
    rows = manifest["final" if final else "dev"]
    checked = set()
    for row in rows:
        for key in ("matrix", "order"):
            path = ROOT / "judge_data" / row[key]
            if path in checked: continue
            if path.is_symlink() or not path.is_file() or path.stat().st_size != row[key+"_bytes"] or sha(path) != row[key+"_sha256"]:
                raise ValueError("trusted input identity mismatch")
            checked.add(path)
    return rows


def evaluate(tree):
    start = time.monotonic()
    deadline = start + LIMITS.evaluator_seconds
    timings = []
    context = None
    phase = "setup"
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    try:
        context = Context(workspace=ROOT.parent)
        mode = "eval" if final else "train"
        members = json.loads((ROOT / "dataset-members.json").read_text())[mode]
        install(ROOT, mode, CONFIG, members, lambda: remaining(deadline, "dataset extraction", reserve=5))
        workloads = _workloads(final)
        with tempfile.TemporaryDirectory(prefix="cw60-", dir=ROOT.parent) as directory:
            scratch = Path(directory)
            with tempfile.TemporaryFile(dir=scratch) as probe:
                require_disk_capture(probe)
            phase = "reference"
            reference = _reference_tree(scratch, deadline)
            # The exact reference (its editable files byte-identical to the reference's) is the
            # operator's self-comparison: a shorter deadline and a ratio gate.
            reference_replay = _editable_files(tree) == _editable_files(reference)
            if reference_replay: deadline = min(deadline, start + 5400)
            trees = {"candidate": Path(tree), "reference": reference}
            binaries = {}
            for label, judged in trees.items():
                phase = label
                # Each build has its own compilation budget, bounded by the overall deadline.
                build_deadline = min(deadline, time.monotonic() + LIMITS.build_seconds)
                try:
                    binaries[label] = build(judged, context, build_deadline)
                except TrustedAggregateDeadline as error:
                    if label != "candidate" or time.monotonic() >= deadline - 5:
                        raise
                    raise ValueError(f"compilation exceeded its {LIMITS.build_seconds}-second "
                                     "budget") from error
            build_deadline = min(deadline, time.monotonic() + LIMITS.build_seconds)
            phase = "oracle"
            oracle = build_oracle(scratch, context, build_deadline)
            passes = []
            for index,row in enumerate(workloads):
                phase = "oracle"
                matrix, order = (ROOT/"judge_data"/row[key] for key in ("matrix","order"))
                oracle_dir = scratch/"expected"
                preprocess(oracle, matrix, order, oracle_dir, context, deadline)
                expected = Expected(oracle_dir, CONFIG["benchmark"])
                try:
                    samples = {"candidate": [], "reference": []}
                    for repetition in range(CONFIG["benchmark"]["pairs"]):
                        labels = ("candidate","reference") if (index+repetition)%2 else ("reference","candidate")
                        for label in labels:
                            phase = label
                            samples[label].append(sample(binaries[label], matrix, order, expected, context, deadline, scratch, side=label))
                    cvals = [s["elapsed_ms"] for s in samples["candidate"]]
                    rvals = [s["elapsed_ms"] for s in samples["reference"]]
                    phase = "reference"
                    diagnostics = {"reference": noise(rvals)}
                    c, r = statistics.median(cvals), statistics.median(rvals)
                    if reference_replay:
                        diagnostics["candidate"] = noise(cvals)
                        if abs(c/r-1) > 0.05: raise ValueError("reference self-comparison ratio gate failed")
                    flop = expected.meta["scalar_operations"]
                    ratio = CONFIG["benchmark"]["strict_time_ratio"]
                    passes.append({"workload": row["id"], "metrics": {
                        "execution_time_ms": c < ratio*r,
                        "throughput_mflops": flop/c > flop/r/ratio}})
                    timings.append({"workload": row["id"], "candidate_ms": c, "reference_ms": r,
                                    "flop": flop, "samples": samples, "noise": diagnostics,
                                    "oracle": expected.meta})
                finally:
                    expected.close()
                    shutil.rmtree(oracle_dir)
            context.memory_check()
            phase = "aggregate"
            remaining(deadline, "evaluation")
            score = _continuous_score([r["candidate_ms"] for r in timings], [r["reference_ms"] for r in timings])
            metrics = {
                "candidate_median_execution_time_ms": statistics.median(r["candidate_ms"] for r in timings),
                "candidate_median_throughput_mflops": statistics.median(r["flop"]/r["candidate_ms"]/1000 for r in timings),
                "candidate_max_output_bytes": max(s["product_bytes"] for r in timings for s in r["samples"]["candidate"]),
                "beat_spec_workload_passes": passes,
                "beats_reference": int(all(v for p in passes for v in p["metrics"].values())),
                "reference_score": 50.0, "margin": score-50.0,
                "private_workload_measurements": [{k:v for k,v in row.items() if k != "samples"} for row in timings],
                "private_samples": [{"workload": row["workload"], "side": label, "repetition": i, **sample}
                                    for row in timings for label, samples in row["samples"].items()
                                    for i, sample in enumerate(samples)],
                "private_runtime": context.receipt,
            }
            return score, score, "valid; every timed output independently checked", metrics
    except Exception as error:
        if getattr(error, "native_diagnostic", None) is not None:
            print("clusterwise private failure context: " + json.dumps({
                "phase": phase, "completed_workloads": len(timings),
                "workload_id": locals().get("row", {}).get("id"),
                "native": error.native_diagnostic,
            }), file=sys.stderr, flush=True)
        # Candidate-caused failures are scored zeros (owner rule): a gate, a 180 s wall timeout,
        # a killed native process (CPU-time ceiling or out-of-memory kill), compilation past its
        # budget, and a measured memory or workspace ceiling. The overall deadline, the
        # reference and the judge's own faults stay evaluation failures.
        if phase != "candidate" or isinstance(error, MemoryError) or not isinstance(
                error, (ValueError, TimeoutError, ResourceCeiling)):
            raise
        diagnostic = json.dumps({"traceback": traceback.format_exc(limit=8),
                                 "completed_workloads": len(timings),
                                 "runtime": context.receipt if context else None})
        print("clusterwise private failure: " + diagnostic[-8192:], file=sys.stderr, flush=True)
        return (
            0.0,
            0.0,
            "build, isolation, correctness or measurement gate failed",
            _failure_metrics(
                final,
                reason=type(error).__name__ + ": " + str(error),
                completed_workloads=len(timings),
                runtime=context.receipt if context else None,
            ),
        )


if __name__ == "__main__":
    import sys
    print(json.dumps(dict(zip(("score", "score_unbounded", "message", "metrics"), evaluate(sys.argv[1])))))
