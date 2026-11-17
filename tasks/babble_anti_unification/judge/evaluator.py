"""Judge of babble_anti_unification: evaluate(tree) on the shared verifier.

`tree` is pristine/ with only src/learn.rs, src/dfta.rs and files below src/learn_extensions/
overlaid; forbidden added substrings are task.toml forbidden_added, and check_tree keeps every
byte outside the two FRONTIER_EDIT bodies (the private reference, pristine/ + reference.patch,
is exempt, and a submission equal to it reuses its measurement). The Rust 1.64 release
benchmark runs every corpus three times; the initial AST cost must match the reference's.

The submission's build and every benchmark run (the reference's runs too) execute in the shared
isolate launcher: a fresh instance that reads only the system, the toolchain and its own
copy of the sources or the corpus, gets a clean environment and leaves nothing running. The
reference is built outside it first; its dependency artifacts seed each candidate build.

Score: per workload S_w = 100 * (1 - 1/r_w) with r_w the compression ratio (initial/final
cost); raw_score = mean S_w clipped to [0, 100], score_unbounded unclipped (see
score_observations). Build, run, CSV or contract failures are a scored zero.

Metrics (one beat_spec_workload_passes row per corpus; development: 5, final: 10):
  compression_ratio  higher; passes if candidate >= reference - 1e-12
  runtime_seconds    median of three runs; lower; passes if candidate < reference
beats_reference = 1 iff every corpus passes both (runtime equality fails).
The metric definitions before the task.toml migration are kept in the authoring records.
"""

from __future__ import annotations

import csv
import io
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import isolate  # the shared launcher, beside edit_scope in the Verifier base

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))


AUTHENTIC_ENTRYPOINT = "src/bin/benchmark/main.rs"
BUILD_MANIFEST = "frontier-build/Cargo.toml"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
WORKLOAD_ROOT = TASK_DIR / "workloads"
PRISTINE = TASK_DIR / "pristine"
EDITABLE_FILES = ("src/learn.rs", "src/dfta.rs")
EXTENSIONS = "src/learn_extensions"

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
_DEPENDENCIES: Path | None = None


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


def _semantic_scaffold_ok(root: Path) -> tuple[bool, str]:
    """Allow edits only inside the two excised bodies.

    The immutable remainder includes candidate reification, rewrite creation,
    deduplication, extraction-facing methods, and the DFTA representation. Those
    fixed upstream operations make every accepted library rewrite a definitional
    abstraction of its search pattern, which independently preserves semantics.
    """
    pristine = PRISTINE
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


def _dependencies() -> Path:
    """The judge's copy of the starter's fixed dependency sources (the vendored crates, the egg
    fork and babble-macros) at one path for every build: Cargo keys path and directory sources
    by absolute path, so a candidate build in its own instance directory reuses the reference
    build's dependency artifacts and recompiles only the babble crate."""
    global _DEPENDENCIES
    if _DEPENDENCIES is None:
        root = Path(tempfile.mkdtemp(prefix="babble-dependencies-"))
        for name in ("frontier-deps", "babble-macros"):
            shutil.copytree(PRISTINE / name, root / name, symlinks=True)
        _DEPENDENCIES = root
    return _DEPENDENCIES


def _cargo_arguments(root: Path) -> list[str]:
    """Check the tree's build spine, point its build manifest (task-owned, in the judge's own
    copy of the tree) at _dependencies() and return the cargo arguments, run from the tree."""
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
    dependencies = _dependencies()
    text = manifest.read_text(encoding="utf-8")
    for relative in ("../babble-macros", "../frontier-deps/egg"):
        if text.count(f'path = "{relative}"') != 1:
            raise RuntimeError("the build manifest's dependency paths changed")
        text = text.replace(f'path = "{relative}"', f'path = "{dependencies / relative[3:]}"')
    manifest.chmod(0o644)
    manifest.write_text(text, encoding="utf-8")
    return ["--manifest-path", BUILD_MANIFEST,
            "--config", 'source.crates-io.replace-with="vendored-sources"',
            "--config", f'source.vendored-sources.directory="{dependencies}/frontier-deps/vendor"']


BUILD = ["--release", "--locked", "--offline", "--bin", "benchmark"]


def _build_reference(root: Path) -> Path:
    """Build the private reference (trusted code, unconfined) into the shared target directory,
    whose dependency artifacts then seed every candidate build."""
    target = _target_dir()
    proc = subprocess.run(
        ["cargo", "build", *_cargo_arguments(root), *BUILD],
        cwd=root,
        env={**os.environ, "CARGO_TARGET_DIR": str(target)},
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


def _sandboxed(argv: list[str], timeout: float, **options) -> isolate.Result:
    """One instance; its stdout and stderr are discarded."""
    result = isolate.run(["sh", "-c", 'exec "$@" >/dev/null 2>&1', "sh", *argv],
                              timeout_seconds=timeout,
                              path=os.environ.get("PATH", isolate.DEFAULT_PATH), **options)
    if result.timed_out:
        raise subprocess.TimeoutExpired(argv, timeout)
    return result


def _build_candidate(root: Path, scratch: Path) -> bytes:
    """Build a copy of the submitted tree in the sandbox, on a copy of the shared target
    directory, and return the benchmark binary. The babble crate is cleaned first: reference
    and candidate are copies with nearly identical timestamps, so Cargo could miss the source
    switch."""
    stage = scratch / "build"
    shutil.copytree(root, stage / "tree", symlinks=True)
    arguments = _cargo_arguments(stage / "tree")
    if _target_dir().is_dir():
        shutil.copytree(_target_dir(), stage / "target", symlinks=True)
    script = ('cargo clean "$@" -p babble && cargo build "$@" ' + " ".join(BUILD)
              + ' || exit 1; cp "$CARGO_TARGET_DIR/release/benchmark" "$BINARY" || true')
    toolchain = {name: os.environ[name] for name in ("RUSTUP_HOME", "RUSTUP_TOOLCHAIN")
                 if name in os.environ}
    build = _sandboxed(["sh", "-c", script, "sh", *arguments], BUILD_TIMEOUT_SECONDS,
                       tree=stage, cwd="{work}/tree", read_paths=[_dependencies()],
                       outputs=["benchmark"], output_bytes=1 << 30,
                       env={**toolchain, "CARGO_HOME": "{home}/cargo",
                            "CARGO_TARGET_DIR": "{work}/target", "BINARY": "{out}/benchmark"})
    shutil.rmtree(stage)
    if build.returncode or build.limit:
        raise ValueError("submitted Rust project did not build")
    if not build.outputs.get("benchmark"):
        raise ValueError("benchmark executable was not produced")
    return build.outputs["benchmark"]


def _measure(root: Path, suite_name: str, names: tuple[Workload, ...],
             reference: bool = False) -> dict[Workload, Observation]:
    scratch = Path(tempfile.mkdtemp(prefix="babble-measure-"))
    try:
        (scratch / "bin").mkdir()
        binary = scratch / "bin" / "benchmark"
        if reference:
            shutil.copyfile(_build_reference(root), binary)
        else:
            binary.write_bytes(_build_candidate(root, scratch))
        binary.chmod(0o555)
        return _run_workloads(root, binary, suite_name, names)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _run_workloads(root: Path, binary: Path, suite_name: str,
                   names: tuple[Workload, ...]) -> dict[Workload, Observation]:
    observations: dict[Workload, Observation] = {}
    for index, workload in enumerate(names):
        # Each run gets its own copy of the tree's harness data (the domain rewrites, read
        # relative to the working directory) and of the corpus.
        case_root = Path(tempfile.mkdtemp(prefix="babble-workload-"))
        try:
            shutil.copytree(root / "harness", case_root / "harness", symlinks=True)
            input_dir = case_root / "cases" / f"{workload.domain}_eval"
            input_dir.mkdir(parents=True)
            shutil.copyfile(
                WORKLOAD_ROOT / suite_name / workload.filename,
                input_dir / "input.json",
            )
            repetitions: list[Observation] = []
            for repeat in range(MEASUREMENT_REPETITIONS):
                run = _sandboxed(
                    [
                        str(binary), "{work}/cases",
                        f"--domain={workload.domain}", *BENCHMARK_ARGS,
                        "--output={out}/result.csv",
                    ],
                    WORKLOAD_TIMEOUT_SECONDS,
                    tree=case_root, read_paths=[binary.parent], outputs=["result.csv"],
                    env={"RAYON_NUM_THREADS": "1"},
                )
                if run.returncode or run.limit:
                    raise ValueError(f"benchmark workload {index + 1} failed")
                if run.outputs.get("result.csv") is None:
                    raise FileNotFoundError("benchmark wrote no CSV")
                rows = list(csv.DictReader(io.StringIO(
                    run.outputs["result.csv"].decode("utf-8"), newline="")))
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


def _reference_tree() -> Path:
    """A fresh copy of the pristine tree with the private reference patch applied."""
    scratch = Path(tempfile.mkdtemp(prefix="babble-reference-"))
    root = scratch / "tree"
    shutil.copytree(PRISTINE, root, symlinks=True)
    for directory, _names, files in os.walk(root):
        for name in [directory, *(os.path.join(directory, f) for f in files)]:
            if not os.path.islink(name):
                os.chmod(name, os.stat(name).st_mode | 0o200)
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(REFERENCE_PATCH)], cwd=root,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch)},
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=300)
    if proc.returncode:
        shutil.rmtree(scratch, ignore_errors=True)
        raise ValueError("reference patch does not apply")
    return root


def _editable_files(root: Path) -> dict[str, bytes]:
    files = {name: (root / name).read_bytes() for name in EDITABLE_FILES
             if (root / name).is_file()}
    extensions = root / EXTENSIONS
    if extensions.is_dir():
        for path in sorted(extensions.rglob("*")):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def _is_reference(root: Path) -> bool:
    """The submission's editable files are exactly the reference's."""
    reference = _reference_tree()
    try:
        return _editable_files(root) == _editable_files(reference)
    finally:
        shutil.rmtree(reference.parent, ignore_errors=True)


def check_tree(tree, pristine, changes):
    """Shared-verifier hook: outside the two marked bodies, the editable files stay exactly
    as shipped (the reference is exempt, as it always was)."""
    root = Path(tree)
    if _is_reference(root):
        return None
    ok, _reason = _semantic_scaffold_ok(root)
    return None if ok else ("out_of_scope", list(EDITABLE_FILES))


def _reference(suite_name: str, names: tuple[str, ...]):
    key = (suite_name, *names)
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]
    root = None
    try:
        root = _reference_tree()
        observations = _measure(root, suite_name, names, reference=True)
        score, _ = score_observations(observations, names)
        _REFERENCE_CACHE[key] = (observations, score)
        return observations, score
    finally:
        if root is not None:
            shutil.rmtree(root.parent, ignore_errors=True)


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


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    suite_name, names = _suite()
    try:
        root = Path(tree)
        is_reference = _is_reference(root)
        reference, reference_score = _reference(suite_name, names)
        if is_reference:
            candidate = reference
        else:
            candidate = _measure(root, suite_name, names)
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
    except isolate.IsolateError:
        raise
    except subprocess.TimeoutExpired:
        return _invalid("build or benchmark timed out", names)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "evaluation failed"
        return _invalid(reason, names)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("usage: evaluator.py TREE", file=sys.stderr)
        raise SystemExit(2)
    result = evaluate(sys.argv[1])
    print(json.dumps({
        "score": result[0], "score_unbounded": result[1],
        "message": result[2], "metrics": result[3],
    }, sort_keys=True))
