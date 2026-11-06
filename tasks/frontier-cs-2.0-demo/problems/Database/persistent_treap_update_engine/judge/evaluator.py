"""Judge of persistent_treap_update_engine (#757): evaluate(tree) on the shared verifier.

`tree` is pristine/ (the Agent's repo/) with only the three update-engine headers overlaid; the
shared edit scope has already rejected anything else (task.toml [submission]). The candidate,
the private reference (pristine/ + reference.patch) and the fixed serial scaffold (pristine/)
are built, checked with the correctness harness, and timed interleaved on six generated traces
(development or final seeds) x four (threads, clients) configurations, five repetitions each.
Every build and run is a fresh shared isolate instance (no judge files, environment or
network; nothing persists between runs).

Score: G[i,q] = median over repetitions of candidate/scaffold throughput; Q = geomean of G over
the 24 pairs; raw_score = score_unbounded = 0 if Q <= 1 else 100*ln Q / (ln 10 + ln Q).
A build, correctness, timeout or output failure is a scored zero; a reference or scaffold
failure raises (a judging error, never a candidate verdict).

Metrics (one beat_spec_workload_passes row per case/configuration pair):
  case_throughput_ktps  median KTPS of the timed benchmark; higher; passes if candidate > reference
  peak_memory_mib       median ru_maxrss of those processes; lower; passes if candidate < reference
  historical_query_us   median us per historical range query; lower; passes if every candidate
                        checksum equals the scaffold's and candidate < reference
beats_reference = 1 iff every pair passes all three metrics (equality fails).
The metric definitions before #757 are archived at docs/task-authoring/persistent_treap_update_engine/provenance/beat_spec.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any

import isolate  # the shared launcher, beside edit_scope in the Verifier base



TASK_DIR = Path(__file__).resolve().parent
APP_DIR = TASK_DIR / "pristine"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
CORRECTNESS_SOURCE = TASK_DIR / "judge" / "correctness.cpp"
HISTORY_SOURCE = TASK_DIR / "judge" / "history_bench.cpp"
ALLOW_PATHS = {
    "lib/treap/scheduler.hpp",
    "lib/treap/search_insert.hpp",
    "lib/treap/search_delete.hpp",
}
TX = struct.Struct("@BQQ")
HEADER = struct.Struct("@QQ")
THROUGHPUT_RE = re.compile(r"\ncontreap\t[^\n]*\t(\d+)\t([0-9.eE+-]+)\s*$")
HISTORY_RE = re.compile(r"^history\t(\d+)\t([0-9.eE+-]+)\t(\d+)\s*$")
BUILD_TIMEOUT = 300
RUN_TIMEOUT = 30
REPETITIONS = 5
BLOCK_SIZE = 1000
CONFIGURATIONS = ((4, 1), (5, 1), (5, 2), (6, 2))
# Paper Section 6.2 reports the average time of 10^5 historical range statistics.
HISTORY_QUERIES = 100_000
BEAT_METRICS = ("case_throughput_ktps", "peak_memory_mib", "historical_query_us")


class ReferenceFailure(RuntimeError):
    """The private reference or the fixed serial baseline could not be measured.

    This is a judging error, never a candidate verdict: it propagates out of
    evaluate() so the Verifier reports a failed verification without a score,
    instead of scoring a correct candidate zero or comparing against an invalid
    baseline.
    """


def _baseline(role: str, function, *args, **kwargs):
    """Run one reference/serial-baseline step and classify any failure as judging error."""
    try:
        return function(*args, **kwargs)
    except ReferenceFailure:
        raise
    except Exception as exc:  # includes subprocess.TimeoutExpired
        raise ReferenceFailure(f"{role} baseline failed: {type(exc).__name__}") from exc


def _workload_id(index: int, threads: int, clients: int) -> str:
    return f"case-{index}/threads-{threads}/clients-{clients}"


def _failed_beat_rows() -> list[dict[str, Any]]:
    return [
        {"workload": _workload_id(index, threads, clients),
         "metrics": {name: False for name in BEAT_METRICS}}
        for index in range(6)
        for threads, clients in CONFIGURATIONS
    ]


def _complete_metrics(candidate: float, reference: float, *,
                      beat_rows: list[dict[str, Any]] | None = None,
                      **extra: Any) -> dict[str, Any]:
    rows = beat_rows if beat_rows is not None else _failed_beat_rows()
    beats_reference = all(
        decision
        for row in rows
        for decision in row["metrics"].values()
    )
    metrics: dict[str, Any] = {
        "beat_spec_workload_passes": rows,
        "beats_reference": int(beats_reference),
        "reference_score": reference,
        "margin": candidate - reference,
    }
    metrics.update(extra)
    return metrics


def _invalid(reason: str, reference: float = 0.0) -> tuple[float, float, str, dict]:
    return 0.0, 0.0, reason, _complete_metrics(0.0, reference, valid=0)


def _seed(label: str, index: int) -> int:
    if label == "dev":
        material = f"contreap-feedback-v2:{index}"
    else:
        secret = os.environ.get("FRONTIER_FINAL_SUITE_KEY", "local-verifier-preflight")
        material = f"{secret}:contreap-graded-v2:{index}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little")


def _suite() -> list[tuple[int, str, int, int]]:
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    label = "final" if final else "dev"
    count = 6
    profiles = ("update_heavy", "hotspot", "balanced", "read_heavy",
                "hotspot", "update_heavy")
    out = []
    for i in range(count):
        seed = _seed(label, i)
        profile = profiles[i % len(profiles)]
        initial = 80000 + (seed % 10000)
        transactions = 450000 + (seed % 90000)
        out.append((seed, profile, initial, transactions))
    return out


def _write_workload(path: Path, seed: int, profile: str, n: int, m: int) -> None:
    rng = random.Random(seed)
    initial = list(range(2, 2 * n + 2, 2))
    rng.shuffle(initial)
    counts = {key: 1 for key in initial}
    live = list(initial)
    live_pos = {key: i for i, key in enumerate(live)}
    maximum = 4 * n

    def erase_live(key: int) -> None:
        pos = live_pos.pop(key)
        tail = live.pop()
        if pos < len(live):
            live[pos] = tail
            live_pos[tail] = pos

    if profile == "read_heavy":
        insert_cut, delete_cut = 18, 28
    elif profile == "update_heavy":
        insert_cut, delete_cut = 60, 88
    elif profile == "hotspot":
        insert_cut, delete_cut = 55, 90
    else:
        insert_cut, delete_cut = 42, 67
    hot_limit = max(128, n // 128)

    with path.open("wb") as stream:
        stream.write(HEADER.pack(n, m))
        stream.write(struct.pack(f"@{n}Q", *initial))
        for _ in range(m):
            pick = rng.randrange(100)
            if pick < insert_cut:
                key = (1 + rng.randrange(hot_limit)
                       if profile == "hotspot" and rng.randrange(100) < 98
                       else 1 + rng.randrange(maximum))
                if key not in counts:
                    live_pos[key] = len(live)
                    live.append(key)
                    counts[key] = 0
                counts[key] += 1
                stream.write(TX.pack(1, key, key))
            elif pick < delete_cut and live:
                if profile == "hotspot" and rng.randrange(100) < 98:
                    key = 1 + rng.randrange(hot_limit)
                    while key not in counts:
                        key = 1 + rng.randrange(hot_limit)
                else:
                    key = live[rng.randrange(len(live))]
                counts[key] -= 1
                if counts[key] == 0:
                    del counts[key]
                    erase_live(key)
                stream.write(TX.pack(2, key, key))
            else:
                left = (1 + rng.randrange(max(1, hot_limit - 128))
                        if profile == "hotspot" and rng.randrange(100) < 98
                        else 1 + rng.randrange(maximum - 128))
                stream.write(TX.pack(0, left, left + 127))


def _apply_reference(root: Path) -> Path:
    work = root / "reference"
    shutil.copytree(APP_DIR, work)
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(REFERENCE_PATCH)],
        cwd=work, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return work


def _sandboxed(command: list[str], read: list[Path], layers=()) -> isolate.Result:
    """One run of a built binary in a fresh isolate instance; a timeout raises."""
    result = isolate.run(command, timeout_seconds=RUN_TIMEOUT, syscalls="no_fork",
                              read_paths=[str(path) for path in read], layers=layers)
    if result.timed_out:
        raise subprocess.TimeoutExpired(command, RUN_TIMEOUT)
    return result


def _compile(tree: Path, output_dir: Path, name: str,
             history: bool = False) -> tuple[Path, ...]:
    include = os.environ.get("MIMALLOC_INCLUDE")
    library = os.environ.get("MIMALLOC_LIBRARY")
    common = ["g++", "-std=c++20", "-O3", "-DNDEBUG", "-march=native", "-msse4", "-maes"]
    includes = ["-I{work}"] + ([f"-I{include}"] if include else [])
    links = ["-lpthread"] + ([library] if library else ["-lmimalloc"])
    main = output_dir / f"main-{name}"
    check = output_dir / f"check-{name}"
    targets = [("{work}/main.cpp", main), ("{in}/correctness.cpp", check)]
    if history:
        targets.append(("{in}/history_bench.cpp", output_dir / f"history-{name}"))
    for source, target in targets:
        # The compiler reads the tree's headers, so it runs in the sandbox too.
        result = isolate.run(
            common + [source] + includes + links + ["-o", "{out}/binary"],
            timeout_seconds=BUILD_TIMEOUT, tree=tree, outputs=["binary"],
            inputs={"correctness.cpp": CORRECTNESS_SOURCE, "history_bench.cpp": HISTORY_SOURCE},
            output_bytes=256 << 20)
        if result.timed_out:
            raise subprocess.TimeoutExpired(common, BUILD_TIMEOUT)
        if result.returncode or result.limit or not result.outputs["binary"]:
            raise RuntimeError(f"{name} build failed")
        target.write_bytes(result.outputs["binary"])
        target.chmod(0o755)
    return tuple(target for _, target in targets)


def _check(binary: Path, suite: list[tuple[int, str, int, int, int, int, int]]) -> None:
    for seed, profile, n, m, threads, block_size, clients in suite:
        for attempt in range(2):
            try:
                result = _sandboxed(
                    [str(binary), str(seed), str(n), str(m), profile,
                     str(threads), str(block_size), str(clients)], [binary])
                # Query-client count follows the corresponding timed configuration.
                # It is a separate argument because scheduler construction does not receive it.
                break
            except subprocess.TimeoutExpired:
                if attempt == 1:
                    raise
        if result.returncode or result.limit:
            raise RuntimeError("persistent-version correctness check failed")


def _measured_process(command: list[str], read: list[Path]) -> tuple[str, int, int]:
    """Run one judge process; return stdout, exit code and its own peak RSS in KiB.

    The launcher reaps exactly this process with wait4, so the peak is its kernel-recorded
    resident-set high-water mark (ru_maxrss), measured outside the instance.
    """
    result = _sandboxed(command, read, [isolate.PeakRss()])
    returncode = result.returncode if result.limit is None else -9
    return (result.stdout.decode(errors="replace"), returncode,
            result.measurements["max_rss_bytes"] // 1024)


def _retry_timeout(command: list[str], read: list[Path]) -> tuple[str, int, int]:
    for attempt in range(2):
        try:
            return _measured_process(command, read)
        except subprocess.TimeoutExpired:
            if attempt == 1:
                raise
    raise AssertionError("unreachable")


def _run(binary: Path, workload: Path, threads: int, clients: int) -> tuple[float, float]:
    """Return (throughput KTPS, peak resident memory MiB) of one benchmark run."""
    stdout, returncode, peak_kib = _retry_timeout(
        [str(binary), "contreap", str(workload), "-threads", str(threads),
         "-clients", str(clients), "-batchsize", str(BLOCK_SIZE)], [binary, workload])
    if returncode:
        raise RuntimeError("benchmark run failed")
    match = THROUGHPUT_RE.search("\n" + stdout)
    if not match:
        raise RuntimeError("benchmark produced no throughput result")
    value = float(match.group(2))
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("benchmark produced invalid throughput")
    peak_mib = peak_kib / 1024.0
    if not math.isfinite(peak_mib) or peak_mib <= 0:
        raise RuntimeError("benchmark produced no peak memory measurement")
    return value, peak_mib


def _history_seed(seed: int, threads: int, clients: int) -> int:
    material = f"contreap-history-v1:{seed}:{threads}:{clients}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little")


def _history(binary: Path, workload: Path, threads: int, seed: int) -> tuple[float, int]:
    """Return (average microseconds per historical range query, answer checksum)."""
    stdout, returncode, _ = _retry_timeout(
        [str(binary), str(workload), str(threads), str(BLOCK_SIZE),
         str(HISTORY_QUERIES), str(seed)], [binary, workload])
    if returncode:
        raise RuntimeError("historical query benchmark failed")
    match = HISTORY_RE.match(stdout.strip())
    if not match or int(match.group(1)) != HISTORY_QUERIES:
        raise RuntimeError("historical query benchmark produced no result")
    seconds = float(match.group(2))
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("historical query benchmark produced invalid time")
    return 1e6 * seconds / HISTORY_QUERIES, int(match.group(3))


def _geomean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _paired_median_ratio(numerator: list[float], denominator: list[float]) -> tuple[float, list[float]]:
    if len(numerator) != REPETITIONS or len(denominator) != REPETITIONS:
        raise ValueError("incomplete paired throughput samples")
    ratios = [a / b for a, b in zip(numerator, denominator, strict=True)]
    return sorted(ratios)[REPETITIONS // 2], ratios


def _strictly_beats(candidate: list[float], reference: list[float]) -> bool:
    if len(candidate) != REPETITIONS or len(reference) != REPETITIONS:
        raise ValueError("incomplete comparison throughput samples")
    return sorted(candidate)[REPETITIONS // 2] > sorted(reference)[REPETITIONS // 2]


def _strictly_below(candidate: list[float], reference: list[float]) -> bool:
    if len(candidate) != REPETITIONS or len(reference) != REPETITIONS:
        raise ValueError("incomplete comparison samples")
    return sorted(candidate)[REPETITIONS // 2] < sorted(reference)[REPETITIONS // 2]


def _same_engine(left: Path, right: Path) -> bool:
    return all((left / name).read_bytes() == (right / name).read_bytes()
               for name in ALLOW_PATHS)


def _score(gain_over_naive: float) -> tuple[float, float]:
    if gain_over_naive <= 1.0:
        return 0.0, 0.0
    log_gain = math.log(gain_over_naive)
    score = 100.0 * log_gain / (math.log(10.0) + log_gain)
    return score, score


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    root = Path(tempfile.mkdtemp(prefix="run-"))
    try:
        candidate_tree = root / "candidate"
        shutil.copytree(tree, candidate_tree)
        reference_tree = _baseline("reference", _apply_reference, root)
        naive_tree = root / "naive"
        shutil.copytree(APP_DIR, naive_tree)
        build_dir = root / "build"
        data_dir = root / "data"
        build_dir.mkdir()
        data_dir.mkdir()
        cand_main, cand_check, cand_history = _compile(
            candidate_tree, build_dir, "candidate", history=True)
        ref_main, ref_check, ref_history = _baseline(
            "reference", _compile, reference_tree, build_dir, "reference", history=True)
        naive_main, naive_check, naive_history = _baseline(
            "naive", _compile, naive_tree, build_dir, "naive", history=True)

        suite = _suite()
        check_suite = [
            (seed ^ 0x9E3779B97F4A7C15 ^ threads ^ (clients << 8),
             profile, n, m, threads, BLOCK_SIZE, clients)
            for seed, profile, n, m in suite
            for threads, clients in CONFIGURATIONS
        ]
        _check(cand_check, check_suite)
        _baseline("reference", _check, ref_check, check_suite)
        _baseline("naive", _check, naive_check, check_suite)

        workloads = []
        for index, case in enumerate(suite):
            seed, profile, n, m = case
            path = data_dir / f"case-{index}.data"
            _write_workload(path, seed, profile, n, m)
            workloads.append((path, profile, seed))

        results: dict[str, list[float]] = {"candidate": [], "reference": [], "naive": []}
        candidate_naive_gains: list[float] = []
        reference_naive_gains: list[float] = []
        beat_rows: list[dict[str, Any]] = []
        case_throughput: list[dict[str, Any]] = []
        case_memory: list[dict[str, Any]] = []
        case_history: list[dict[str, Any]] = []
        performance_measurements: list[dict[str, Any]] = []
        binaries = {"candidate": cand_main, "reference": ref_main, "naive": naive_main}
        history_binaries = {"candidate": cand_history, "reference": ref_history}
        orders = (
            ("candidate", "reference", "naive"),
            ("naive", "reference", "candidate"),
            ("reference", "candidate", "naive"),
            ("naive", "candidate", "reference"),
            ("candidate", "naive", "reference"),
        )
        for index, (workload, _profile, case_seed) in enumerate(workloads):
            for threads, clients in CONFIGURATIONS:
                samples = {name: [] for name in binaries}
                memory = {name: [] for name in binaries}
                history = {name: [] for name in history_binaries}
                history_seed = _history_seed(case_seed, threads, clients)
                # The fixed serial scaffold is the deterministic answer oracle:
                # historical answers depend only on the trace and query seed.
                _, oracle_checksum = _baseline(
                    "naive", _history, naive_history, workload, threads, history_seed)
                candidate_history_mismatches = 0
                for repetition in range(REPETITIONS):
                    order = orders[(index + threads + clients + repetition) % len(orders)]
                    for name in order:
                        if name == "candidate":
                            ktps, peak_mib = _run(binaries[name], workload, threads, clients)
                        else:
                            ktps, peak_mib = _baseline(
                                name, _run, binaries[name], workload, threads, clients)
                        samples[name].append(ktps)
                        memory[name].append(peak_mib)
                    history_order = (("candidate", "reference")
                                     if (index + threads + clients + repetition) % 2 == 0
                                     else ("reference", "candidate"))
                    for name in history_order:
                        if name == "candidate":
                            query_us, checksum = _history(
                                history_binaries[name], workload, threads, history_seed)
                            candidate_history_mismatches += int(checksum != oracle_checksum)
                        else:
                            query_us, checksum = _baseline(
                                name, _history, history_binaries[name], workload,
                                threads, history_seed)
                            if checksum != oracle_checksum:
                                # A reference that answers a historical query
                                # wrongly is not a valid performance baseline;
                                # none of its samples may set the bar.
                                raise ReferenceFailure(
                                    "reference historical range answers disagree "
                                    "with the serial scaffold")
                        history[name].append(query_us)
                candidate_naive_gain, candidate_naive_ratios = _paired_median_ratio(
                    samples["candidate"], samples["naive"]
                )
                reference_naive_gain, reference_naive_ratios = _paired_median_ratio(
                    samples["reference"], samples["naive"]
                )
                candidate_naive_gains.append(candidate_naive_gain)
                reference_naive_gains.append(reference_naive_gain)
                medians: dict[str, float] = {}
                for name in binaries:
                    value = sorted(samples[name])[REPETITIONS // 2]
                    medians[name] = value
                    results[name].append(value)
                workload_id = _workload_id(index, threads, clients)
                case_throughput.append({
                    "workload": workload_id,
                    "value": round(medians["candidate"], 3),
                })
                case_memory.append({
                    "workload": workload_id,
                    "value": round(sorted(memory["candidate"])[REPETITIONS // 2], 3),
                })
                case_history.append({
                    "workload": workload_id,
                    "value": round(sorted(history["candidate"])[REPETITIONS // 2], 4),
                })
                performance_measurements.append({
                    "workload": workload_id,
                    "candidate_throughput_ktps": samples["candidate"],
                    "reference_throughput_ktps": samples["reference"],
                    "naive_throughput_ktps": samples["naive"],
                    "candidate_naive_ratios": candidate_naive_ratios,
                    "reference_naive_ratios": reference_naive_ratios,
                    "candidate_naive_median_ratio": candidate_naive_gain,
                    "reference_naive_median_ratio": reference_naive_gain,
                    "candidate_peak_memory_mib": memory["candidate"],
                    "reference_peak_memory_mib": memory["reference"],
                    "naive_peak_memory_mib": memory["naive"],
                    "candidate_historical_query_us": history["candidate"],
                    "reference_historical_query_us": history["reference"],
                    "historical_queries": HISTORY_QUERIES,
                    "historical_query_oracle_checksum": oracle_checksum,
                    "candidate_historical_checksum_mismatches": candidate_history_mismatches,
                    "reference_historical_answers_verified": True,
                })
                beat_rows.append({
                    "workload": workload_id,
                    "metrics": {
                        "case_throughput_ktps": _strictly_beats(
                            samples["candidate"], samples["reference"]
                        ),
                        "peak_memory_mib": _strictly_below(
                            memory["candidate"], memory["reference"]
                        ),
                        # A fast wrong historical answer never clears the metric.
                        "historical_query_us": (
                            candidate_history_mismatches == 0
                            and _strictly_below(history["candidate"], history["reference"])
                        ),
                    },
                })

        perf = {name: _geomean(values) for name, values in results.items()}
        aggregate_gain = _geomean(candidate_naive_gains)
        reference_gain = _geomean(reference_naive_gains)
        candidate_score, unbounded = _score(aggregate_gain)
        reference_score, _ = _score(reference_gain)
        metrics = _complete_metrics(
            candidate_score, reference_score, beat_rows=beat_rows,
            valid=1, cases=len(suite) * len(CONFIGURATIONS),
            case_throughput_ktps=case_throughput,
            peak_memory_mib=case_memory,
            historical_query_us=case_history,
            performance_measurements=performance_measurements,
            aggregate_candidate_naive_ratio=aggregate_gain,
            aggregate_reference_naive_ratio=reference_gain,
            throughput_ktps=round(perf["candidate"], 3),
            naive_throughput_ktps=round(perf["naive"], 3),
            scalability_ratio=round(results["candidate"][-1] / results["candidate"][-2], 4),
        )
        return candidate_score, unbounded, "valid; mixed-workload throughput scored", metrics
    except (ReferenceFailure, isolate.IsolateError):
        # Judging error: the Verifier reports a failed verification, not a score.
        raise
    except subprocess.TimeoutExpired:
        return _invalid("build or benchmark timed out")
    except Exception as exc:
        reason = str(exc)
        if not reason or len(reason) > 120:
            reason = "evaluation failed"
        return _invalid(reason)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"score": 0.0, "message": "expected one judged tree"}))
        return 2
    score, unbounded, message, metrics = evaluate(sys.argv[1])
    print(json.dumps({"score": score, "score_unbounded": unbounded,
                      "message": message, "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
