"""Paired evaluator for BABBLE learned-library candidate generation."""

from __future__ import annotations

import csv
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

# This task-owned copy is intentionally separate from the pipeline-managed
# submission_patch.py, which is refreshed from a shared template before every
# gate and currently uses pathlib keywords unavailable in the Python 3.9 judge.
from submission_patch_compat import apply_submission_patch


AUTHENTIC_ENTRYPOINT = "src/bin/benchmark/main.rs"
BUILD_MANIFEST = "frontier-build/Cargo.toml"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
WORKLOAD_ROOT = TASK_DIR / "workloads"
CONFIG = {
    "submission": {
        "allow_paths": ["src/learn.rs", "src/dfta.rs", "src/learn_extensions/"],
        "max_changed_files": 20,
        "max_patch_bytes": 262144,
        "allow_empty": False,
    }
}

@dataclass(frozen=True)
class Workload:
    name: str
    domain: str
    filename: str


DEV_SUITE = (
    Workload("corpus-1", "list", "list.json"),
    Workload("corpus-2", "physics", "physics.json"),
    Workload("corpus-3", "text", "text.json"),
    Workload("corpus-4", "logo", "logo.json"),
    Workload("corpus-5", "towers", "towers.json"),
)
FINAL_SUITE = (
    Workload("corpus-1", "list", "list_a.json"),
    Workload("corpus-2", "list", "list_b.json"),
    Workload("corpus-3", "physics", "physics_a.json"),
    Workload("corpus-4", "physics", "physics_b.json"),
    Workload("corpus-5", "text", "text_a.json"),
    Workload("corpus-6", "text", "text_b.json"),
    Workload("corpus-7", "logo", "logo_a.json"),
    Workload("corpus-8", "logo", "logo_b.json"),
    Workload("corpus-9", "towers", "towers_a.json"),
    Workload("corpus-10", "towers", "towers_b.json"),
)
BUILD_TIMEOUT_SECONDS = 300
WORKLOAD_TIMEOUT_SECONDS = 30
MEASUREMENT_REPETITIONS = 3
COMPRESSION_ABS_TOLERANCE = 1e-12
BENCHMARK_ARGS = (
    "--beam-size=64",
    "--lps=1",
    "--mode=babble",
    "--rounds=2",
    "--max-arity=2",
    "--use-all=1",
    "--lib-iter-limit=1",
)
FORBIDDEN_ADDITIONS = (
    "std::env", "std::fs", "std::process", "std::net", "std::time",
    "option_env!", "env!", "include!", "include_bytes!", "unsafe",
    "libc", "asm!", "/proc", "/sys", "frontier_", "harbor_",
    "command::new", "process::exit", "tcpstream", "udpstream",
)
EDIT_REGIONS = {
    "src/learn.rs": (
        "// FRONTIER_EDIT_BEGIN: candidate generation",
        "// FRONTIER_EDIT_END: candidate generation",
    ),
    "src/dfta.rs": (
        "// FRONTIER_EDIT_BEGIN: paired-state construction",
        "// FRONTIER_EDIT_END: paired-state construction",
    ),
}


@dataclass(frozen=True)
class Observation:
    compression_ratio: float
    runtime_seconds: float
    initial_cost: int
    final_cost: int
    library_count: int
    runtime_spread_seconds: float = 0.0


_REFERENCE_CACHE: dict[tuple[object, ...], tuple[dict[Workload, Observation], float]] = {}
_SHARED_TARGET: Path | None = None


def _suite() -> tuple[str, tuple[Workload, ...]]:
    if os.environ.get("FRONTIER_FAST_TEST") == "1":
        return "dev", (DEV_SUITE[0],)
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    return ("final", FINAL_SUITE) if final else ("dev", DEV_SUITE)


def _audit_rows(names: tuple[Workload, ...],
                decisions: dict[Workload, dict[str, bool]] | None = None):
    """Private verifier evidence; the judge aggregates and strips these rows."""
    rows = []
    for workload in names:
        values = (decisions or {}).get(workload, {})
        rows.append({
            "workload": workload.name,
            "metrics": {
                "compression_ratio": bool(values.get("compression_ratio", False)),
                "runtime_seconds": bool(values.get("runtime_seconds", False)),
            },
        })
    return rows


def _metrics(names: tuple[Workload, ...], reference_score: float = 0.0,
             candidate_score: float = 0.0,
             observations: dict[Workload, Observation] | None = None,
             decisions: dict[Workload, dict[str, bool]] | None = None,
             aggregation=None) -> dict:
    rows = _audit_rows(names, decisions)
    obs = observations or {}
    aggregate = aggregation or {
        "beats_reference": False,
        "qualified_margin": -1.0,
        "absolute_improvement_seconds": 0.0,
        "relative_improvement": 0.0,
        "runtime_wins": 0,
        "winning_domains": 0,
    }
    return {
        "beat_spec_workload_passes": rows,
        "beats_reference": int(bool(aggregate["beats_reference"])),
        "reference_score": float(reference_score),
        # The generic aggregate transport validates this hidden qualified margin.
        # A negative sentinel means that an individual comparison or the
        # absolute-runtime boundary failed before the relative gate.
        "margin": float(aggregate["qualified_margin"]),
        "compression_score_margin": float(candidate_score - reference_score),
        "aggregate_runtime_improvement_seconds": float(
            aggregate["absolute_improvement_seconds"]
        ),
        "aggregate_runtime_improvement_fraction": float(
            aggregate["relative_improvement"]
        ),
        "runtime_workload_wins": int(aggregate["runtime_wins"]),
        "runtime_winning_domains": int(aggregate["winning_domains"]),
        "compression_ratio": [
            round(obs[workload].compression_ratio, 9)
            for workload in names if workload in obs
        ],
        "runtime_seconds": [
            round(obs[workload].runtime_seconds, 9)
            for workload in names if workload in obs
        ],
    }


def _invalid(reason: str, names: tuple[Workload, ...], reference_score: float = 0.0):
    return 0.0, 0.0, reason[:140], _metrics(names, reference_score, 0.0)


def _policy_ok(patch: Path) -> tuple[bool, str]:
    if patch.read_bytes() == REFERENCE_PATCH.read_bytes():
        return True, ""
    text = patch.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        lowered = line[1:].lower()
        if any(token in lowered for token in FORBIDDEN_ADDITIONS):
            return False, "workload, environment, timing, file, or process introspection is not allowed"
    return True, ""


def _semantic_scaffold_ok(root: Path) -> tuple[bool, str]:
    """Allow edits only inside the two excised bodies.

    The immutable remainder includes candidate reification, rewrite creation,
    deduplication, extraction-facing methods, and the DFTA representation. Those
    fixed upstream operations make every accepted library rewrite a definitional
    abstraction of its search pattern, which independently preserves semantics.
    """
    pristine = TASK_DIR / "repo"
    for relative, (begin, end) in EDIT_REGIONS.items():
        try:
            candidate_text = (root / relative).read_text(encoding="utf-8")
            pristine_text = (pristine / relative).read_text(encoding="utf-8")
            if candidate_text.count(begin) != 1 or candidate_text.count(end) != 1:
                return False, f"{relative} must retain its editable-region markers"
            if pristine_text.count(begin) != 1 or pristine_text.count(end) != 1:
                return False, "task scaffold is invalid"
            cand_before, cand_tail = candidate_text.split(begin, 1)
            _cand_body, cand_after = cand_tail.split(end, 1)
            base_before, base_tail = pristine_text.split(begin, 1)
            _base_body, base_after = base_tail.split(end, 1)
            if cand_before != base_before or cand_after != base_after:
                return False, f"{relative} changes immutable semantic scaffolding"
        except (OSError, UnicodeError):
            return False, f"could not validate {relative}"
    return True, ""


def _target_dir() -> Path:
    global _SHARED_TARGET
    if _SHARED_TARGET is None:
        _SHARED_TARGET = Path(tempfile.mkdtemp(prefix="babble-cargo-target-"))
    return _SHARED_TARGET


def _build(root: Path) -> Path:
    # Cargo discovers the retained upstream binary from this exact source
    # entrypoint.  Keeping the check here makes a missing or substituted spine a
    # hard evaluation failure rather than silently running a task-owned program.
    entrypoint = root / AUTHENTIC_ENTRYPOINT
    if not entrypoint.is_file():
        raise ValueError("authentic benchmark entrypoint is missing")
    manifest = root / BUILD_MANIFEST
    if not manifest.is_file():
        raise ValueError("task-owned build manifest is missing")
    # The fresh manifest and lock deliberately omit answer-retrieval metadata.
    # Their local dependency preserves the exact fork required by this source.
    env = os.environ.copy()
    target = _target_dir()
    env["CARGO_TARGET_DIR"] = str(target)
    # Reference and candidate are disposable copies with nearly identical
    # timestamps.  Reusing dependencies is important, but reusing the babble
    # crate itself can make Cargo miss a source switch between those copies.
    # Clean only that package so every measured binary is compiled from the
    # tree currently under evaluation.
    if (target / "release" / "benchmark").exists():
        clean = subprocess.run(
            ["cargo", "clean", "--manifest-path", str(manifest), "-p", "babble"],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if clean.returncode:
            raise ValueError("could not prepare submitted Rust project build")
    proc = subprocess.run(
        ["cargo", "build", "--manifest-path", str(manifest),
         "--release", "--locked", "--offline", "--bin", "benchmark"],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=BUILD_TIMEOUT_SECONDS,
    )
    if proc.returncode:
        raise ValueError("submitted Rust project did not build")
    binary = target / "release" / "benchmark"
    if not binary.is_file():
        raise ValueError("benchmark executable was not produced")
    return binary


def _measure(root: Path, suite_name: str,
             names: tuple[Workload, ...]) -> dict[Workload, Observation]:
    binary = _build(root)
    observations: dict[Workload, Observation] = {}
    for index, workload in enumerate(names):
        case_root = Path(tempfile.mkdtemp(prefix="babble-workload-"))
        try:
            input_dir = case_root / f"{workload.domain}_eval"
            input_dir.mkdir()
            shutil.copyfile(
                WORKLOAD_ROOT / suite_name / workload.filename,
                input_dir / "input.json",
            )
            repetitions: list[Observation] = []
            for repeat in range(MEASUREMENT_REPETITIONS):
                output = case_root / f"result-{repeat}.csv"
                env = os.environ.copy()
                env["RAYON_NUM_THREADS"] = "1"
                env.pop("CARGO_TARGET_DIR", None)
                proc = subprocess.run(
                    [
                        str(binary), str(case_root),
                        f"--domain={workload.domain}", *BENCHMARK_ARGS,
                        f"--output={output}",
                    ],
                    cwd=root,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=WORKLOAD_TIMEOUT_SECONDS,
                )
                if proc.returncode:
                    raise ValueError(f"benchmark workload {index + 1} failed")
                with output.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                if len(rows) != 1:
                    raise ValueError(
                        f"benchmark workload {index + 1} produced invalid CSV"
                    )
                initial = int(rows[0]["initial cost"])
                final = int(rows[0]["final cost"])
                runtime = float(rows[0]["total time"])
                compression = float(rows[0]["compression"])
                library_count = int(rows[0]["num libs"])
                if (
                    initial <= 0 or final <= 0 or library_count < 0
                    or runtime < 0 or not math.isfinite(runtime)
                ):
                    raise ValueError(
                        f"benchmark workload {index + 1} produced invalid metrics"
                    )
                expected = initial / final
                if not math.isclose(
                    compression, expected, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"benchmark workload {index + 1} reported inconsistent costs"
                    )
                repetitions.append(
                    Observation(compression, runtime, initial, final, library_count)
                )
            discrete = {
                (obs.initial_cost, obs.final_cost, obs.library_count)
                for obs in repetitions
            }
            if len(discrete) != 1:
                raise ValueError(
                    f"benchmark workload {index + 1} produced unstable discrete outputs"
                )
            first = repetitions[0]
            observations[workload] = Observation(
                first.compression_ratio,
                statistics.median(obs.runtime_seconds for obs in repetitions),
                first.initial_cost,
                first.final_cost,
                first.library_count,
                max(obs.runtime_seconds for obs in repetitions)
                - min(obs.runtime_seconds for obs in repetitions),
            )
        finally:
            shutil.rmtree(case_root, ignore_errors=True)
    return observations


def _workload_value(obs: Observation) -> float:
    # Compression is the only continuous objective. Runtime is an independently
    # reported, tolerance-aware beat gate, so a faster implementation can never
    # compensate for worse compression.
    return 100.0 * (1.0 - 1.0 / obs.compression_ratio)


def score_observations(observations: dict[Workload, Observation],
                       names: tuple[Workload, ...]) -> tuple[float, float]:
    unbounded = statistics.fmean(
        _workload_value(observations[workload]) for workload in names
    )
    return min(100.0, max(0.0, unbounded)), unbounded


def comparison_decisions(candidate: dict[Workload, Observation],
                         reference: dict[Workload, Observation],
                         names: tuple[Workload, ...]) -> dict[Workload, dict[str, bool]]:
    """Per-workload quality and strict runtime decisions."""
    return {
        workload: {
            "compression_ratio": (
                candidate[workload].compression_ratio
                >= reference[workload].compression_ratio
                - COMPRESSION_ABS_TOLERANCE
            ),
            "runtime_seconds": (
                candidate[workload].runtime_seconds
                < reference[workload].runtime_seconds
            ),
        }
        for workload in names
    }


def aggregate_beat_decision(
    candidate: dict[Workload, Observation],
    reference: dict[Workload, Observation],
    names: tuple[Workload, ...],
    decisions: dict[Workload, dict[str, bool]],
) -> dict:
    """Require every workload to clear every declared PASS metric."""
    candidate_total = sum(candidate[workload].runtime_seconds for workload in names)
    reference_total = sum(reference[workload].runtime_seconds for workload in names)
    absolute = reference_total - candidate_total
    relative = absolute / reference_total if reference_total > 0.0 else -1.0
    runtime_winners = [
        workload for workload in names
        if decisions[workload]["runtime_seconds"]
    ]
    winning_domains = {workload.domain for workload in runtime_winners}
    compression_all = all(
        decisions[workload]["compression_ratio"] for workload in names
    )
    runtime_all = all(
        decisions[workload]["runtime_seconds"] for workload in names
    )
    beats = compression_all and runtime_all
    return {
        "beats_reference": beats,
        # Aggregate improvements are diagnostics only; PASS is the conjunction
        # of the per-workload decisions above.
        "qualified_margin": relative if beats else -1.0,
        "absolute_improvement_seconds": absolute,
        "relative_improvement": relative,
        "runtime_wins": len(runtime_winners),
        "winning_domains": len(winning_domains),
    }


def _apply(path: Path):
    result = apply_submission_patch(str(path), TASK_DIR, CONFIG)
    if not result.ok or not result.workdir:
        raise ValueError(result.message)
    return result, Path(result.workdir)


def _reference(suite_name: str, names: tuple[str, ...]):
    key = (suite_name, *names)
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]
    result = None
    try:
        result, root = _apply(REFERENCE_PATCH)
        observations = _measure(root, suite_name, names)
        score, _ = score_observations(observations, names)
        _REFERENCE_CACHE[key] = (observations, score)
        return observations, score
    finally:
        if result and result.workdir:
            shutil.rmtree(Path(result.workdir).parent, ignore_errors=True)


def prepare() -> dict:
    suite_name, names = _suite()
    try:
        _reference(suite_name, names)
    except Exception as exc:
        # The generated judge includes this complete chained exception in its
        # readiness traceback. Do not collapse startup failures to a generic
        # 503, because that made the incompatible patch adapter opaque.
        raise RuntimeError(
            f"reference preparation failed ({type(exc).__name__}): {exc}"
        ) from exc
    return {"ready": True, "workloads": len(names)}


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    candidate_result = None
    suite_name, names = _suite()
    try:
        path = Path(solution_path)
        candidate_result = apply_submission_patch(str(path), TASK_DIR, CONFIG)
        if not candidate_result.ok or not candidate_result.workdir:
            return _invalid(candidate_result.message, names)
        ok, reason = _policy_ok(path)
        if not ok:
            return _invalid(reason, names)

        is_reference = path.read_bytes() == REFERENCE_PATCH.read_bytes()
        if not is_reference:
            ok, reason = _semantic_scaffold_ok(Path(candidate_result.workdir))
            if not ok:
                return _invalid(reason, names)

        reference, reference_score = _reference(suite_name, names)
        if is_reference:
            candidate = reference
        else:
            candidate = _measure(Path(candidate_result.workdir), suite_name, names)
        if any(
            candidate[workload].initial_cost != reference[workload].initial_cost
            for workload in names
        ):
            return _invalid(
                "benchmark changed an immutable input-cost invariant",
                names,
                reference_score,
            )
        score, unbounded = score_observations(candidate, names)
        decisions = comparison_decisions(candidate, reference, names)
        aggregation = aggregate_beat_decision(
            candidate, reference, names, decisions
        )
        metrics = _metrics(
            names, reference_score, score, candidate, decisions, aggregation
        )
        return score, unbounded, "valid benchmark run; higher is better", metrics
    except subprocess.TimeoutExpired:
        return _invalid("build or benchmark timed out", names)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "evaluation failed"
        return _invalid(reason, names)
    finally:
        if candidate_result and candidate_result.workdir:
            shutil.rmtree(Path(candidate_result.workdir).parent, ignore_errors=True)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("usage: evaluator.py SOLUTION_PATCH", file=sys.stderr)
        raise SystemExit(2)
    result = evaluate(sys.argv[1])
    print(json.dumps({
        "score": result[0], "score_unbounded": result[1],
        "message": result[2], "metrics": result[3],
    }, sort_keys=True))
