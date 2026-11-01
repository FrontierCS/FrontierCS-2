from __future__ import annotations

import fcntl
import hashlib
import math
import os
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from submission_patch import apply_submission_patch

TASK_DIR = Path(__file__).resolve().parent
APP_DIR = TASK_DIR / "harbor" / "app"
EDITABLE = Path("lib/evmcow/execution_state.hpp")
CONFIG = {"submission": {"kind": "patch", "path": "/app/solution.patch",
          "allow_paths": [str(EDITABLE)], "max_changed_files": 1,
          "max_patch_bytes": 65536}}

# Generated workloads from the authors' parsers. Final seeds are judge-only.
DEV_SUITE = (("YCSB:10000:0.9", 1907), ("Smallbank:10000:0.9", 3253))
FINAL_SUITE = (("YCSB:15000:0.9", 24036583), ("Smallbank:15000:0.9", 24036619),
               ("TPCC:100:5", 24036671))
FORBIDDEN_ADDITIONS = ("#define", "getenv", "system(", "popen", "fork(",
                       "malloc", "calloc", "realloc", "aligned_alloc", "mmap", "sbrk",
                       "operator new", "operator delete", "Statistics", "checkpoint_bench",
                       "FRONTIER_", "int main", "mallctl", "jemalloc")
BEAT_METRICS = ("correctness", "committed_tps", "checkpoint_create_ns",
                "checkpoint_restore_ns", "retained_bytes_per_checkpoint")
BEAT_REPETITIONS = 5


def _workload_id(workload: str, seed: int) -> str:
    return f"{workload}:seed={seed}"


MICRO_WORKLOAD = "checkpoint-micro"


def _workload_ids(suite: tuple[tuple[str, int], ...]) -> list[str]:
    """Every Spectrum workload and one checkpoint-micro workload per suite seed.

    Each workload reports the same metric set (#646): throughput, checkpoint
    create and restore nanoseconds, retained bytes per checkpoint, correctness.
    """
    return ([_workload_id(workload, seed) for workload, seed in suite]
            + [_workload_id(MICRO_WORKLOAD, seed) for _, seed in suite])


def _beat_rows(final: bool, decisions: dict[str, dict[str, bool]] | None = None) -> list[dict]:
    suite = FINAL_SUITE if final else DEV_SUITE
    decisions = decisions or {}
    return [{"workload": workload,
             "metrics": {name: decisions.get(workload, {}).get(name, False) for name in BEAT_METRICS}}
            for workload in _workload_ids(suite)]


def _metrics(candidate: float, reference: float, extra: dict | None = None,
             *, final: bool = False, decisions: dict[str, dict[str, bool]] | None = None) -> dict:
    out = dict(extra or {})
    rows = _beat_rows(final, decisions)
    out.update({
        "beats_reference": int(all(value for row in rows for value in row["metrics"].values())),
        "beat_spec_workload_passes": rows,
        "reference_score": reference,
        "margin": candidate - reference,
    })
    return out


def _gate_metrics(final: bool, extra: dict | None = None) -> dict:
    """A failed gate retains all fixed workload rows, with every metric false."""
    return _metrics(0.0, _bounded_score(_REFERENCE_ANCHOR.get(final, 0.0)), extra, final=final)


def _cache_root() -> Path:
    configured = os.environ.get("FRONTIER_SPECTRUM_CACHE")
    if configured:
        return Path(configured)
    preferred = Path("/work/tmp/frontier-spectrum-checkpoint")
    # Fall back to the process temp directory rather than a literal /tmp: the
    # FrontierCS2 Verifier root filesystem is read-only and only TMPDIR is writable.
    return preferred if preferred.parent.is_dir() else Path(tempfile.gettempdir()) / "frontier-spectrum-checkpoint"


def _cache_lock(root: Path) -> Path:
    return root.with_name(root.name + ".lock")


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(APP_DIR.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(APP_DIR)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_checked(cmd: list[str], *, timeout: int, cwd: Path | None = None,
                 env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)


def _ensure_cache() -> tuple[Path, Path]:
    root = _cache_root()
    source = root / "source"
    marker = root / "fingerprint"
    fingerprint = _source_fingerprint()
    seed = os.environ.get("FRONTIER_SPECTRUM_CACHE_SEED")
    if seed and not marker.is_file() and (Path(seed) / "fingerprint").is_file():
        # The Verifier image prefetched and built the dependency cache; the
        # run-scoped container has no network, so copy the baked seed into the
        # writable cache root instead of configuring from scratch.
        shutil.rmtree(root, ignore_errors=True)
        shutil.copytree(seed, root, symlinks=True)
        # The seed is published read-only; the copy must be writable for the
        # patched source tree and incremental builds.
        for path in (root, *root.rglob("*")):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IWUSR)
    if marker.is_file() and marker.read_text().strip() == fingerprint and (source / "build/bin/bench").is_file():
        return root, source
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    shutil.copytree(APP_DIR, source)
    proc = _run_checked(["cmake", "-S", str(source), "-B", str(source / "build"),
                         "-DNDEBUG=ON"], timeout=1800)
    if proc.returncode:
        raise ValueError("Spectrum dependency configuration failed")
    proc = _run_checked(["cmake", "--build", str(source / "build"), "--target", "bench",
                         "-j", str(max(1, min(4, os.cpu_count() or 1)))], timeout=1800)
    if proc.returncode:
        raise ValueError("Spectrum initial build failed")
    marker.write_text(fingerprint)
    return root, source


def prepare() -> dict:
    try:
        root = _cache_root()
        root.parent.mkdir(parents=True, exist_ok=True)
        with _cache_lock(root).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _, source = _ensure_cache()
        return {"ready": True, "spectrum_bench": (source / "build/bin/bench").is_file()}
    except Exception:
        return {"ready": False, "reason": "Spectrum build preparation failed"}


def _reference_header(tmp: Path) -> bytes:
    tree = tmp / "reference-tree"
    shutil.copytree(APP_DIR, tree)
    proc = _run_checked(["git", "-C", str(tree), "apply", "--whitespace=nowarn",
                         str(TASK_DIR / "reference.patch")], timeout=30)
    if proc.returncode:
        raise ValueError("internal reference patch failed")
    return (tree / EDITABLE).read_bytes()


def _compile_micro(tree: Path, output: Path) -> None:
    proc = _run_checked(["g++", "-O3", "-DNDEBUG", "-std=c++20", f"-I{tree}",
                         f"-I{TASK_DIR / 'judge_deps/include'}",
                         str(TASK_DIR / "checkpoint_bench.cpp"), "-o", str(output)], timeout=60)
    if proc.returncode:
        raise ValueError("submitted checkpoint code does not compile")


def _checkpoint_measurement(binary: Path, seed: int) -> dict[str, float]:
    proc = _run_checked([str(binary), "400", str(seed)], timeout=30)
    if proc.returncode:
        raise ValueError("checkpoint lifetime or rollback validation failed")
    try:
        create_ns, restore_ns, retained_bytes, live_checkpoints, _checksum, end_to_end_ns = map(
            int, proc.stdout.strip().split())
    except Exception as exc:
        raise ValueError("checkpoint validation returned invalid output") from exc
    if not (0 < create_ns < 1_000_000_000 and 0 < restore_ns < 1_000_000_000
            and 0 < end_to_end_ns < 1_000_000_000):
        raise ValueError("checkpoint timing is invalid")
    if not (0 < retained_bytes <= 64 * 1024 * 1024 and live_checkpoints == 16):
        raise ValueError("retained checkpoint memory is invalid")
    return {"create_ns": float(create_ns), "restore_ns": float(restore_ns),
            "retained_bytes": float(retained_bytes),
            # Rounds per second of the timed save -> write -> restore -> access loop.
            "throughput": 1e9 / float(end_to_end_ns)}


def _checkpoint_profile(tree: Path, output: Path, seeds: tuple[int, ...]) -> dict:
    _compile_micro(tree, output)
    per_seed = {seed: [_checkpoint_measurement(output, seed)] for seed in seeds}
    creates = [per_seed[seed][0]["create_ns"] for seed in seeds]
    restores = [per_seed[seed][0]["restore_ns"] for seed in seeds]
    retained = [per_seed[seed][0]["retained_bytes"] for seed in seeds]
    return {"create_ns": statistics.median(creates),
            "restore_ns": statistics.median(restores),
            "retained_bytes": float(max(retained)), "per_seed": per_seed}


def _repeat_checkpoint_profiles(profiles: dict[str, dict], binaries: dict[str, Path],
                                seeds: tuple[int, ...]) -> None:
    # Preserve the first measurement used by the historical continuous score.
    # Four additional traces per seed and compared variant establish an observed
    # timing range for the new beat decision, without extra Spectrum builds/runs.
    names = ("candidate", "reference")
    for repetition in range(1, BEAT_REPETITIONS):
        for index, seed in enumerate(seeds):
            offset = (repetition + index) % len(names)
            for name in names[offset:] + names[:offset]:
                profiles[name]["per_seed"][seed].append(_checkpoint_measurement(binaries[name], seed))


def _separated(candidate: list[float], reference: list[float], *, higher: bool) -> bool:
    if len(candidate) != BEAT_REPETITIONS or len(reference) != BEAT_REPETITIONS:
        return False
    if any(not math.isfinite(value) or value <= 0 for value in candidate + reference):
        return False
    return min(candidate) > max(reference) if higher else max(candidate) < min(reference)


METRIC_FIELDS = (("committed_tps", "throughput", True),
                 ("checkpoint_create_ns", "create_ns", False),
                 ("checkpoint_restore_ns", "restore_ns", False),
                 ("retained_bytes_per_checkpoint", "retained_bytes", False))


def _row(candidate: list[dict], reference: list[dict]) -> dict[str, bool]:
    row = {"correctness": True}
    for metric, field, higher in METRIC_FIELDS:
        row[metric] = _separated([sample[field] for sample in candidate],
                                 [sample[field] for sample in reference], higher=higher)
    return row


def _beat_decisions(suite: tuple[tuple[str, int], ...], samples: dict,
                    profiles: dict[str, dict]) -> dict[str, dict[str, bool]]:
    """The same metric set, decided by the same strict-separation rule, on every workload."""
    decisions = {}
    for workload, seed in suite:
        decisions[_workload_id(workload, seed)] = _row(samples["candidate"][workload],
                                                       samples["reference"][workload])
        decisions[_workload_id(MICRO_WORKLOAD, seed)] = _row(profiles["candidate"]["per_seed"][seed],
                                                             profiles["reference"]["per_seed"][seed])
    return decisions


def _build_variant(source: Path, header: bytes, destination: Path) -> None:
    (source / EDITABLE).write_bytes(header)
    proc = _run_checked(["cmake", "--build", str(source / "build"), "--target", "bench",
                         "-j", str(max(1, min(4, os.cpu_count() or 1)))], timeout=600)
    if proc.returncode:
        raise ValueError("submitted Spectrum integration does not build")
    shutil.copy2(source / "build/bin/bench", destination)


def _runtime_env(source: Path) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
           "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    libs = [source / "build/_deps/fmtlib/fmt/install/lib",
            source / "build/_deps/gflags/gflags/install/lib",
            source / "build/_deps/google/glog/install/lib",
            source / "build/_deps/jemalloc/jemalloc/install/lib"]
    env["LD_LIBRARY_PATH"] = ":".join(str(x) for x in libs)
    return env


CHECKPOINT_LINE = re.compile(
    r"^checkpoint create_ns ([0-9.]+) restore_ns ([0-9.]+) retained_bytes ([0-9.]+) "
    r"creates ([0-9]+) restores ([0-9]+) retained_samples ([0-9]+)$", re.MULTILINE)


def _one_benchmark(binary: Path, source: Path, workload: str, seed: int) -> dict[str, float]:
    """Committed TPS and the harness's checkpoint metrics from one Spectrum run.

    The Spectrum harness (lib/transaction/evm-transaction.cpp) times every
    save_checkpoint() and load_checkpoint() made during real execution, measures
    the bytes held per live checkpoint, and checks every restore (height, code
    position, top() and every item); a failed check aborts the run.
    """
    proc = _run_checked([str(binary), "Spectrum:2:9973:CHECKPOINTED", workload, "1s", str(seed)],
                        timeout=20, env=_runtime_env(source))
    if proc.returncode:
        raise ValueError("Spectrum workload crashed, failed a restore check or timed out")
    match = re.search(r"^commit\s+([0-9]+(?:\.[0-9]+)?) tx/s$", proc.stderr, re.MULTILINE)
    checkpoint = CHECKPOINT_LINE.search(proc.stderr)
    if not match or not checkpoint:
        raise ValueError("Spectrum workload returned invalid statistics")
    tps = float(match.group(1))
    create_ns, restore_ns, retained = (float(checkpoint.group(i)) for i in (1, 2, 3))
    creates, restores, retained_samples = (int(checkpoint.group(i)) for i in (4, 5, 6))
    if not math.isfinite(tps) or tps <= 0:
        raise ValueError("Spectrum committed no transactions")
    if not (creates > 0 and restores > 0 and retained_samples > 0
            and all(math.isfinite(x) and x > 0 for x in (create_ns, restore_ns, retained))):
        raise ValueError("Spectrum workload returned invalid checkpoint statistics")
    return {"throughput": tps, "create_ns": create_ns, "restore_ns": restore_ns,
            "retained_bytes": retained}


def _qualities(binaries: dict[str, Path], source: Path,
               suite: tuple[tuple[str, int], ...], profiles: dict[str, dict],
               repetitions: int) -> tuple[dict[str, float], dict]:
    samples = {name: {case: [] for case, _ in suite} for name in binaries}
    names = tuple(binaries)
    for repetition in range(repetitions):
        for case_index, (workload, seed) in enumerate(suite):
            offset = (repetition + case_index) % len(names)
            for name in names[offset:] + names[:offset]:
                samples[name][workload].append(_one_benchmark(binaries[name], source, workload, seed))
    quality = {}
    for name in names:
        # One composite per workload over the uniform metric set, then the
        # geometric mean over every Spectrum and checkpoint-micro workload.
        observations = _observations(name, suite, samples, profiles)
        logs = [math.log(_workload_quality(observation)) for observation in observations.values()]
        quality[name] = math.exp(sum(logs) / len(logs))
    return quality, samples


OBSERVED_FIELDS = ("throughput", "create_ns", "restore_ns", "retained_bytes")


def _observations(name: str, suite: tuple[tuple[str, int], ...], samples: dict,
                  profiles: dict[str, dict]) -> dict[str, dict[str, float]]:
    """The values each workload contributes to the composite, by workload id."""
    observations = {}
    for case, seed in suite:
        runs = samples[name][case]
        observations[_workload_id(case, seed)] = {
            field: statistics.median(run[field] for run in runs) for field in OBSERVED_FIELDS}
        observations[_workload_id(MICRO_WORKLOAD, seed)] = {
            field: profiles[name]["per_seed"][seed][0][field] for field in OBSERVED_FIELDS}
    return observations


def _workload_values(suite: tuple[tuple[str, int], ...], samples: dict,
                     profiles: dict[str, dict]) -> dict[str, dict]:
    """Per-workload values of every compared variant (evidence only, never feedback).

    For each workload and variant: the values that enter the composite, that
    workload's quality, and every observation behind them, so a later change of
    the score mapping or composite can be recomputed exactly from the evidence.
    """
    values: dict[str, dict] = {}
    for name in samples:
        observed = _observations(name, suite, samples, profiles)
        for case, seed in suite:
            for workload, runs in ((_workload_id(case, seed), samples[name][case]),
                                   (_workload_id(MICRO_WORKLOAD, seed), profiles[name]["per_seed"][seed])):
                values.setdefault(workload, {})[name] = {
                    **observed[workload],
                    "quality": _workload_quality(observed[workload]),
                    "observations": {field: [run[field] for run in runs] for field in OBSERVED_FIELDS},
                }
    return values


def _workload_quality(observation: dict[str, float]) -> float:
    checkpoint_latency = math.sqrt(observation["create_ns"] * observation["restore_ns"])
    checkpoint_cost = observation["retained_bytes"] * checkpoint_latency
    return observation["throughput"] / (checkpoint_cost ** (1.0 / 3.0))


# The score is 100 * tanh(decades / SCORE_SCALE_DECADES), where decades is
# log10 of composite quality over the checkpoint backend the task ships.
# A linear one-decade scale clipped at 100 and recorded finals reached 12x. The
# scale puts the withheld implementation (about 5.6x on the platform) at 50 and
# keeps every larger quality strictly higher without reaching 100.
SCORE_SCALE_DECADES = 1.36
_REFERENCE_ANCHOR: dict[bool, float] = {}


def _bounded_score(unbounded: float) -> float:
    """Map the unbounded log-ratio score (100 per decade) onto [0, 100)."""
    if not (math.isfinite(unbounded) and unbounded > 0):
        return 0.0
    return 100.0 * math.tanh(unbounded / 100.0 / SCORE_SCALE_DECADES)


def _score(quality: float, baseline: float) -> float:
    """Unbounded score: 100 * log10 of composite quality over the shipped backend.

    It was `100 * r / (1 + r)` against the *reference*, which pays exactly 50 of
    100 for reproducing the withheld implementation's own quality and asymptotes
    at 100 — so 2x scored 66.7, 10x scored 90.9 and everything past that was
    indistinguishable. The reference could not score anything but 50, on either
    suite, forever, and its position therefore said nothing about how much the
    removed contribution is worth.

    The shipped backend was already being built and profiled here as
    `qualities["naive"]` and the number was thrown away. It is the honest zero:
    quality is throughput divided by the cube root of retained-bytes times the
    geometric mean of create and restore latency, so matching the shipped backend
    on all four of those is worth nothing and every axis has to be earned.
    """
    if not (quality > 0 and baseline > 0):
        return 0.0
    return 100.0 * math.log10(quality / baseline)


def _forbidden_patch(patch: Path) -> str | None:
    text = patch.read_text(encoding="utf-8", errors="replace")
    added = "\n".join(line[1:] for line in text.splitlines()
                       if line.startswith("+") and not line.startswith("+++"))
    for token in FORBIDDEN_ADDITIONS:
        if token.lower() in added.lower():
            return token
    return None


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    result = None
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    try:
        result = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not result.ok:
            return 0.0, 0.0, result.message, _gate_metrics(final, result.metrics)
        forbidden = _forbidden_patch(Path(solution_path))
        if forbidden:
            return 0.0, 0.0, "patch uses a forbidden benchmark-access token", _gate_metrics(final, result.metrics)
        suite = FINAL_SUITE if final else DEV_SUITE
        repetitions = int(os.environ.get("FRONTIER_SPECTRUM_REPETITIONS", "5"))
        repetitions = max(1, min(5, repetitions))
        root = _cache_root()
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="spectrum-eval-") as temp_name, _cache_lock(root).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _, source = _ensure_cache()
            temp = Path(temp_name)
            candidate_header = (result.workdir / EDITABLE).read_bytes()
            naive_header = (APP_DIR / EDITABLE).read_bytes()
            reference_header = _reference_header(temp)
            micro_seeds = tuple(seed for _, seed in suite)
            profiles = {
                "candidate": _checkpoint_profile(result.workdir, temp / "micro-candidate", micro_seeds),
                "naive": _checkpoint_profile(APP_DIR, temp / "micro-naive", micro_seeds),
            }
            reference_tree = temp / "reference-tree"
            profiles["reference"] = _checkpoint_profile(reference_tree, temp / "micro-reference", micro_seeds)
            binaries = {name: temp / f"bench-{name}" for name in ("candidate", "naive", "reference")}
            try:
                _build_variant(source, candidate_header, binaries["candidate"])
                _build_variant(source, naive_header, binaries["naive"])
                _build_variant(source, reference_header, binaries["reference"])
                qualities, samples = _qualities(binaries, source, suite, profiles, repetitions)
                _repeat_checkpoint_profiles(
                    profiles, {name: temp / f"micro-{name}" for name in ("candidate", "reference")},
                    micro_seeds)
            finally:
                (source / EDITABLE).write_bytes(naive_header)
            # Both the shipped backend and the withheld one are fixed artifacts, so
            # their quality ratio is a property of the task rather than of any
            # submission. Measured once per judge process and reused, which keeps a
            # reference submission exactly repeatable — the old rule got that for
            # free by comparing the reference against itself, and got a meaningless
            # constant 50 with it.
            if _REFERENCE_ANCHOR.get(final) is None:
                _REFERENCE_ANCHOR[final] = _score(qualities["reference"], qualities["naive"])
            reference_unbounded = _REFERENCE_ANCHOR[final]
            if Path(solution_path).read_bytes() == (TASK_DIR / "reference.patch").read_bytes():
                unbounded = reference_unbounded
            else:
                unbounded = _score(qualities["candidate"], qualities["naive"])
            score = _bounded_score(unbounded)
            reference_score = _bounded_score(reference_unbounded)
            decisions = _beat_decisions(suite, samples, profiles)
            metrics = _metrics(score, reference_score, result.metrics, final=final, decisions=decisions)
            candidate_profile = profiles["candidate"]
            metrics.update({"correct": 1, "spectrum_quality": round(qualities["candidate"], 6),
                            "committed_tps": math.exp(sum(math.log(statistics.median(
                                run["throughput"] for run in samples["candidate"][workload]))
                                for workload, _ in suite) / len(suite)),
                            "retained_bytes_per_checkpoint": int(candidate_profile["retained_bytes"]),
                            "checkpoint_create_ns": candidate_profile["create_ns"],
                            "checkpoint_restore_ns": candidate_profile["restore_ns"],
                            "shipped_backend_quality": round(qualities["naive"], 6),
                            "quality_over_shipped": round(
                                qualities["candidate"] / max(qualities["naive"], 1e-12), 4),
                            "workload_cases": 2 * len(suite),
                            "reference_quality": round(qualities["reference"], 6),
                            "reference_score_unbounded": reference_unbounded,
                            "workload_values": _workload_values(suite, samples, profiles)})
            return score, unbounded, f"valid Spectrum checkpoint backend; score {score:.3f}", metrics
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return 0.0, 0.0, str(exc)[:160], _gate_metrics(final)
    except Exception:
        return 0.0, 0.0, "evaluation failed", _gate_metrics(final)
    finally:
        if result is not None and result.workdir is not None:
            shutil.rmtree(result.workdir.parent, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: evaluator.py PATCH", file=sys.stderr)
        raise SystemExit(2)
    print(evaluate(sys.argv[1]))
