"""Judge of template_task for the shared verifier (#757): evaluate(tree).

This docstring is the task's one scoring note; fill in each part for a new task.

Score: raw_score = clamp(score_unbounded, 0, 100), score_unbounded = 50 + 25 * log2(S), where S
is the geometric mean over workloads of reference time / candidate time (50 at the reference's
speed; each doubling adds 25). A build failure or a wrong count scores 0. Train measures the
three public workloads, final those plus three held-out ones.

Beat-spec metrics (beat_spec_workload_passes, one row per workload):
- wall_seconds, lower is better: the smallest of five wall times of a fresh bench process,
  candidate and reference runs interleaved; passes when 0 < candidate < reference.
beats_reference = 1 iff every workload of the evaluated suite passes every metric; otherwise 0.
"""

# Contract with the shared verifier (tasks/shared/verifier/fcs2_verifier.py), which imports this
# file from a writable copy of the Verifier image's /opt/frontier/task:
#
#   evaluate(tree) -> (score, score_unbounded, message, metrics)
#
#   tree            the starter (pristine/) with only the submission's allowed files overlaid.
#                   The shared verifier has already applied the patch and checked the edit
#                   scope (task.toml [submission]); a submission outside it never reaches here.
#   role            os.environ["FRONTIER_SUBMISSION_ROLE"] == "final" selects the final suite;
#                   anything else is a development (train) submission
#   score           float in [0, 100], the task score
#   score_unbounded the same formula without the clamp
#   message         a short summary; it is evidence only, the solver never sees it
#   metrics         a JSON-serialisable dict. It always goes to private evidence. A train result
#                   returns only beat_spec_progress (aggregate counts the shared verifier
#                   computes from beat_spec_workload_passes) and the metrics task.toml
#                   [feedback] names. It must carry
#                     beats_reference            0 or 1, the rule stated in the docstring
#                     beat_spec_workload_passes  [{"workload": id, "metrics": {name: bool}}],
#                                                the same metric names in every row
#                     reference_score, margin    finite numbers (final results need them)
#
# A submission that fails here (build failure, wrong answer) is a scored zero, not an
# exception: return 0.0, 0.0, reason, {...}. Raise only for the judge's own faults.
# Rules task.toml cannot express go in an optional check_tree(tree, pristine, changes) hook
# (see TEMPLATE.md); this task needs none.
#
# Everything here runs offline, as uid 65532, with a read-only root and a writable scratch
# (TMPDIR, HOME). The suites below are the private data: they exist only in the Verifier image.

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRISTINE = ROOT / "pristine"         # docker/verifier.Dockerfile copies repo/ here; the reference applies to it
METRIC = "wall_seconds"              # the one beat-spec metric (see the docstring)
REPLICATES = 5
REFERENCE_SCORE = 50.0              # the score formula gives the reference exactly 50
RUN_TIMEOUT = 60
CFLAGS = ["-O2", "-std=c11", "-Wall", "-Wextra"]  # = repo/Makefile; the statement says so

# Private suites. Train measures PUBLIC; final measures PUBLIC + HELD_OUT. The expected counts
# are checked independently of both implementations.
PUBLIC = {"n5m": (5_000_000, 348513), "n10m": (10_000_000, 664579), "n20m": (20_000_000, 1270607)}
HELD_OUT = {"n8m": (8_000_000, 539777), "n15m": (15_000_000, 970704),
            "n25m": (25_000_000, 1565927)}


def _build(tree: Path, out: Path) -> str | None:
    """Compile bench/bench from `tree`; return an error line, or None."""
    proc = subprocess.run(
        ["cc", *CFLAGS, f"-I{tree / 'src'}", "-o", str(out),
         str(tree / "bench/main.c"), str(tree / "src/primes.c")],
        capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        return lines[-1][:200] if lines else "compiler failed"
    return None


def _time(binary: Path, n: int, expected: int) -> float | None:
    """Wall time of one run, or None if it fails or prints a wrong count."""
    start = time.monotonic()
    try:
        proc = subprocess.run([str(binary), str(n)], capture_output=True, text=True,
                              timeout=RUN_TIMEOUT, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None
    elapsed = time.monotonic() - start
    if proc.returncode != 0 or proc.stdout.strip() != str(expected):
        return None
    return elapsed


def _reference_tree(scratch: Path) -> Path:
    tree = scratch / "reference"
    shutil.copytree(PRISTINE, tree)
    # The ceiling keeps git from treating an enclosing checkout as the repository.
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch.resolve())}
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(ROOT / "reference.patch")],
                   cwd=tree, check=True, capture_output=True, timeout=60, env=env)
    return tree


def evaluate(tree: str):
    tree = Path(tree)
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    suite = {**PUBLIC, **HELD_OUT} if final else dict(PUBLIC)
    # A scored zero still carries every private field the final result needs: each workload
    # fails, and the reference comparison is reported as numbers (the shared verifier refuses
    # a final result without a finite reference_score and margin).
    rejected = {"beats_reference": 0, "beat_spec_workload_passes": [
        {"workload": wid, "metrics": {METRIC: False}} for wid in suite],
        "reference_score": REFERENCE_SCORE, "margin": -REFERENCE_SCORE}

    scratch = Path(tempfile.mkdtemp(prefix="template-judge-"))
    try:
        candidate, reference = scratch / "candidate", scratch / "reference.bin"
        error = _build(tree, candidate)
        if error:
            return 0.0, 0.0, f"build failed: {error}", rejected
        if _build(_reference_tree(scratch), reference):
            raise RuntimeError("the reference does not build")  # a judge fault, not a score

        cand, ref = {}, {}
        for _ in range(REPLICATES):          # interleave lanes so drift hits both alike
            for wid, (n, expected) in suite.items():
                c = _time(candidate, n, expected)
                if c is None:
                    return 0.0, 0.0, f"wrong or failed output on {'a held-out' if wid in HELD_OUT else 'a public'} workload", rejected
                r = _time(reference, n, expected)
                if r is None:
                    raise RuntimeError("the reference failed")
                cand.setdefault(wid, []).append(c)
                ref.setdefault(wid, []).append(r)
        c_min = {wid: min(v) for wid, v in cand.items()}
        r_min = {wid: min(v) for wid, v in ref.items()}
        decisions = [{"workload": wid, "metrics": {METRIC: 0 < c_min[wid] < r_min[wid]}}
                     for wid in suite]
        # Score: 50 at the reference's speed, doubling speed adds 25, clamped to [0, 100].
        speedup = math.exp(sum(math.log(r_min[w] / c_min[w]) for w in suite) / len(suite))
        unbounded = 50.0 + 25.0 * math.log2(speedup)
        score = min(100.0, max(0.0, unbounded))
        metrics = {
            "beats_reference": int(all(d["metrics"][METRIC] for d in decisions)),
            "beat_spec_workload_passes": decisions,
            "reference_score": REFERENCE_SCORE,
            "margin": score - REFERENCE_SCORE,
            "wall_seconds_candidate_by_workload": c_min,
            "wall_seconds_reference_by_workload": r_min,
            "wall_seconds_replicates_by_workload": {"candidate": cand, "reference": ref},
            # Candidate-only, no workload identities: a value task.toml [feedback] could name.
            "wall_seconds_total": sum(c_min.values()),
        }
        return score, unbounded, f"geometric-mean speedup {speedup:.3f}", metrics
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
