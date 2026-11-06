#!/usr/bin/env python3
"""batch_local_counting judge: evaluate(tree) for the shared FrontierCS2 verifier (#757).

Scoring. Three trees are built (Release, -j1): the independent path (pristine/ +
baseline_naive.patch, B), the reference (pristine/ + reference.patch) and the candidate (the
judged tree, C). Five rounds in rotating order each run the six cases (three graphs x the
5voc/4voc batch) with `scope.out -b -share`; every output file must equal the independent
path's, byte for byte, in every round. raw_score = score_unbounded =
100 * L / (2 + L), L = max(0, median_r log2(B_r / C_r)); any validity failure scores 0.

Metrics (one row per case in beat_spec_workload_passes):
  exact_counts           every output filename and byte equals the reference; passes when exact.
  batch_latency_seconds  case wall time, lower is better; passes when correct and the
                         candidate is strictly faster than the reference in all five rounds.
beats_reference = 1 iff both metrics pass on all six cases. Train also returns
median_seconds (the candidate's median full-suite seconds).

The shared verifier has already applied the patch with the shared edit scope (task.toml
[submission]); identical source trees are built and timed once. Every build and counting run
of every tree is a fresh instance of the shared isolate launcher (no judge access, private
directories deleted after the run, every process killed), so no state survives between runs;
case times are the launcher's fork-to-exit stamp of scope.out alone.
"""
from __future__ import annotations

import hashlib
import gzip
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
PRISTINE = TASK_DIR / "pristine"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
import isolate  # the shared launcher, beside edit_scope in the Verifier base

# Fixed independent halves of the upstream queries for which the artifact's
# independent and optimized paths agree exactly on every evaluated graph.
_COMPATIBLE_5 = tuple(n for n in range(15, 73)
                      if n not in {22, 23, 49, 50, 54, 55, 68, 69, 70, 71})
_COMPATIBLE_4 = (0, 3, 4, 5, 9, 10, 11, 14)
DEV_5 = _COMPATIBLE_5[::2]
FINAL_5 = _COMPATIBLE_5[1::2]
DEV_4 = _COMPATIBLE_4[::2]
FINAL_4 = _COMPATIBLE_4[1::2]
RUNS = 5
BUILD_TIMEOUT = 120 + 300 + 120 + 600 + 60  # the four step limits below, plus copying out
RUN_TIMEOUT = 180
OUTPUT_BYTES = 256 << 20
BUILT = ".frontier-build"  # scope.out and the shared libraries it loads, side by side
# The CMake targets are fixed (no build file is editable): scope.out links libplan, libgraph and
# libutility as shared libraries; the instance that built them is gone when they run, so the
# runs load them from the judge's copy through LD_LIBRARY_PATH.
ARTIFACTS = ("scope.out", "libplan.so", "libgraph.so", "libutility.so")
# nauty's configure rewrites tracked files, so the build runs in the instance's own copy. Each
# step keeps its own limit; `timeout` exits 124 when one is exceeded.
BUILD_SCRIPT = r"""set -e
cd utility/automorphism
timeout -s KILL 120 ./configure >/dev/null 2>&1
sed -i '0,/^CFLAGS=/s//CFLAGS= -fPIC /' makefile
timeout -s KILL 300 make -j1 nauty.a >/dev/null 2>&1
mv nauty.a libnauty.a
cd ../..
timeout -s KILL 120 cmake -S . -B .frontier-build -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
timeout -s KILL 600 cmake --build .frontier-build --target scope.out -j1 >/dev/null 2>&1
cp .frontier-build/executable/scope.out .frontier-build/plan/libplan.so \
   .frontier-build/graph/libgraph.so .frontier-build/utility/libutility.so "$1/"
"""


CASE_NAMES = tuple(f"{graph}-{size}" for graph in
                   ("web-spam", "rec-movielens", "bio-grid-yeast") for size in (5, 4))


def _metrics(candidate_score: float, reference_score: float, rows: list[dict] | None = None) -> dict:
    if rows is None:
        rows = [{"workload": name, "metrics": {
            "exact_counts": False, "batch_latency_seconds": False,
        }} for name in CASE_NAMES]
    if tuple(row["workload"] for row in rows) != CASE_NAMES:
        raise ValueError("incomplete fixed workload decisions")
    return {
        "beat_spec_workload_passes": rows,
        "beats_reference": int(all(value for row in rows for value in row["metrics"].values())),
        "reference_score": reference_score,
        "margin": candidate_score - reference_score,
    }


def _score(naive_seconds: float, candidate_seconds: float) -> float:
    if naive_seconds <= 0 or candidate_seconds <= 0:
        return 0.0
    return _score_log_ratio(math.log2(naive_seconds / candidate_seconds))


def _score_log_ratio(log_ratio: float) -> float:
    improvement = max(0.0, log_ratio)
    return 100.0 * (improvement / (2.0 + improvement))


def _score_log_ratios(log_ratios: list[float]) -> tuple[float, float]:
    median_log_ratio = statistics.median(log_ratios)
    score = _score_log_ratio(median_log_ratio)
    return score, 2.0 ** median_log_ratio


def _build(tree: Path) -> Path:
    """Build `tree` in an isolate instance; return the judge's copy of scope.out (its
    shared libraries beside it)."""
    result = isolate.run(["sh", "-c", BUILD_SCRIPT, "build", "{out}"], tree=tree,
                              outputs=ARTIFACTS, timeout_seconds=BUILD_TIMEOUT,
                              output_bytes=OUTPUT_BYTES, env={"LC_ALL": "C"})
    if result.timed_out or result.returncode in (124, 137):
        raise subprocess.TimeoutExpired("build", BUILD_TIMEOUT)
    if not result.ok or not all(result.outputs.get(name) for name in ARTIFACTS):
        raise RuntimeError("C++ build failed")
    built = tree / BUILT
    built.mkdir(exist_ok=True)
    for name in ARTIFACTS:
        (built / name).write_bytes(result.outputs[name])
        (built / name).chmod(0o555)
    return built / "scope.out"


def _make_workload(tree: Path, size: int, ids: tuple[int, ...], name: str) -> Path:
    workload = tree / f".frontier-workload-{name}-{size}"
    workload.mkdir()
    source = tree / "exp" / "pattern_graph" / f"{size}voc"
    for number in ids:
        shutil.copy2(source / f"{number}.txt", workload / f"{number}.txt")
    return workload


def _inflate_cache(tree: Path, stem: str) -> Path:
    source = tree / "exp" / "data_graph" / f"{stem}.bin.gz"
    destination = tree / ".frontier-assets" / f"{stem}.bin"
    destination.parent.mkdir(exist_ok=True)
    with gzip.open(source, "rb") as compressed, destination.open("wb") as expanded:
        shutil.copyfileobj(compressed, expanded)
    return destination


def _case_specs(tree: Path, final: bool) -> list[tuple[str, Path, Path, Path]]:
    ids5 = FINAL_5 if final else DEV_5
    ids4 = FINAL_4 if final else DEV_4
    data = tree / "exp" / "data_graph"
    graphs = [
        ("web-spam", data / "web-spam.txt", data / "web-spamt.bin"),
        ("rec-movielens", data / "rec-movielens.txt", _inflate_cache(tree, "rec-movielenst")),
        ("bio-grid-yeast", data / "bio-grid-yeast.txt", _inflate_cache(tree, "bio-grid-yeastt")),
    ]
    cases: list[tuple[str, Path, Path, Path]] = []
    for name, graph, triangle in graphs:
        cases.append((f"{name}-5", _make_workload(tree, 5, ids5, name), graph, triangle))
        cases.append((f"{name}-4", _make_workload(tree, 4, ids4, name), graph, triangle))
    return cases


def _digest_outputs(outputs: dict) -> str:
    if not any(data is not None for data in outputs.values()):
        raise RuntimeError("counter emitted no result files")
    digest = hashlib.sha256()
    for name in sorted(outputs):
        data = outputs[name]
        digest.update(name.encode())
        digest.update(b"\0")
        # A missing file differs from every present one (no empty-file collision).
        digest.update(b"\1missing" if data is None else data)
    return digest.hexdigest()


def _patched_pristine(patch: Path) -> Path:
    """A writable copy of the pristine starter with a trusted judge patch applied."""
    tree = Path(tempfile.mkdtemp(prefix="judge-tree-")) / "tree"
    shutil.copytree(PRISTINE, tree, symlinks=True)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(tree.parent.resolve())}
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=tree,
                   check=True, capture_output=True, timeout=300, env=env)
    return tree


def _tree_key(tree: Path) -> str:
    """Identity of a source tree before it is built: the names and bytes of every file."""
    digest = hashlib.sha256()
    for path in sorted(p for p in tree.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(tree)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prepare_tree(tree: Path, final: bool) -> dict:
    binary = _build(tree)
    cases = []
    for name, workload, graph, triangle in _case_specs(tree, final):
        # The results come back as the instance's out/<query file name>, one per query.
        command = [
            str(binary), "-q", str(workload), "-d", str(graph),
            "-t", str(triangle), "-r", "{out}", "-b", "-share",
        ]
        reads = [str(p.resolve()) for p in (binary.parent, workload, graph, triangle)]
        names = sorted(p.name for p in workload.iterdir())
        cases.append((name, (command, reads, names), str(binary.parent.resolve())))
    return {
        "tree": tree,
        "cases": cases,
        "digest": "",
        "times": [],
        "case_times": {name: [] for name, _, _ in cases},
        "case_digests": {},
    }


def _run_prepared(prepared: dict) -> float:
    seconds = 0.0
    digest = hashlib.sha256()
    for name, (command, reads, names), libraries in prepared["cases"]:
        result = isolate.run(command, read_paths=reads, outputs=names, cwd="{work}",
                                  timeout_seconds=RUN_TIMEOUT, output_bytes=OUTPUT_BYTES,
                                  env={"LC_ALL": "C", "LD_LIBRARY_PATH": libraries})
        if result.timed_out:
            raise subprocess.TimeoutExpired(command[0], RUN_TIMEOUT)
        # The case time is scope.out alone (fork to exit, stamped by the launcher); the round
        # adds the judge's own fixed hashing, not the instance's staging and cleanup.
        prepared["case_times"][name].append(result.command_seconds)
        if not result.ok:
            raise RuntimeError("counting run failed")
        hashing = time.monotonic()
        digest.update(name.encode())
        case_digest = _digest_outputs(result.outputs)
        previous = prepared["case_digests"].get(name)
        if previous is not None and previous != case_digest:
            raise RuntimeError("counts changed between repeated runs")
        prepared["case_digests"][name] = case_digest
        digest.update(case_digest.encode())
        seconds += result.command_seconds + (time.monotonic() - hashing)
    current = digest.hexdigest()
    if prepared["digest"] and current != prepared["digest"]:
        raise RuntimeError("counts changed between repeated runs")
    prepared["digest"] = current
    prepared["times"].append(seconds)
    return seconds


def _workload_decisions(naive: dict, reference: dict, candidate: dict) -> list[dict]:
    rows = []
    for name in CASE_NAMES:
        ref_times, cand_times = reference["case_times"][name], candidate["case_times"][name]
        if len(ref_times) != RUNS or len(cand_times) != RUNS:
            raise ValueError("incomplete per-workload timing rounds")
        if any(not math.isfinite(t) or t <= 0 for t in ref_times + cand_times):
            raise ValueError("invalid per-workload time")
        if naive["case_digests"][name] != reference["case_digests"][name]:
            raise ValueError("reference count validation failed")
        correct = candidate["case_digests"][name] == reference["case_digests"][name]
        # The existing five interleaved rounds are paired observations. Require
        # consistent strict improvement, rather than inventing a percentage
        # band from the old aggregate score. A tie (including a deduplicated
        # exact reference replay) cannot be a timing win.
        faster = all(c < r for c, r in zip(cand_times, ref_times, strict=True))
        rows.append({"workload": name, "metrics": {
            "exact_counts": bool(correct), "batch_latency_seconds": bool(correct and faster),
        }})
    return rows


def _paired_protocol(trees: dict[str, Path], final: bool) -> dict:
    # Identical source trees are built and timed once (a resubmitted reference is a tie,
    # never a timing win), exactly as identical patches were before.
    label_keys = {label: _tree_key(tree) for label, tree in trees.items()}
    unique_keys: list[str] = []
    key_trees: dict[str, Path] = {}
    for label in ("naive", "reference", "candidate"):
        key = label_keys[label]
        if key not in key_trees:
            unique_keys.append(key)
            key_trees[key] = trees[label]

    prepared: dict[str, dict] = {}
    try:
        for key in unique_keys:
            prepared[key] = _prepare_tree(key_trees[key], final)

        reference_logs: list[float] = []
        candidate_logs: list[float] = []
        for round_id in range(RUNS):
            shift = round_id % len(unique_keys)
            order = unique_keys[shift:] + unique_keys[:shift]
            round_times: dict[str, float] = {}
            for key in order:
                round_times[key] = _run_prepared(prepared[key])
            naive_time = round_times[label_keys["naive"]]
            reference_logs.append(math.log2(
                naive_time / round_times[label_keys["reference"]]))
            candidate_logs.append(math.log2(
                naive_time / round_times[label_keys["candidate"]]))

        naive = prepared[label_keys["naive"]]
        reference = prepared[label_keys["reference"]]
        candidate = prepared[label_keys["candidate"]]
        reference_score, reference_speedup = _score_log_ratios(reference_logs)
        candidate_score, candidate_speedup = _score_log_ratios(candidate_logs)
        return {
            "correct": (naive["digest"] == reference["digest"]
                        and candidate["digest"] == reference["digest"]),
            "reference_score": reference_score,
            "candidate_score": candidate_score,
            "reference_speedup": reference_speedup,
            "candidate_speedup": candidate_speedup,
            "candidate_median_seconds": statistics.median(candidate["times"]),
            "naive_median_seconds": statistics.median(naive["times"]),
            "beat_spec_workload_passes": _workload_decisions(naive, reference, candidate),
        }
    finally:
        for item in prepared.values():
            # Built outputs are large; the trees themselves are removed by their owners.
            shutil.rmtree(item["tree"] / ".frontier-build", ignore_errors=True)


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    """Score the judged tree (pristine + the submission's allowed files)."""
    # The judge's own trees: pristine/ with its trusted patches. A failure here is the
    # judge's fault and raises. The candidate tree belongs to the shared verifier.
    trees = {
        "naive": _patched_pristine(TASK_DIR / "baseline_naive.patch"),
        "reference": _patched_pristine(TASK_DIR / "reference.patch"),
        "candidate": Path(tree),
    }
    try:
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        result = _paired_protocol(trees, final)
        reference_score = result["reference_score"]
        candidate_score = result["candidate_score"]
        if not result["correct"]:
            return (0.0, 0.0, "per-query counts do not exactly match",
                    _metrics(0.0, reference_score, result["beat_spec_workload_passes"]))
        metrics = _metrics(candidate_score, reference_score, result["beat_spec_workload_passes"])
        metrics.update({
            "valid_patch": 1,
            "median_seconds": result["candidate_median_seconds"],
            "naive_median_seconds": result["naive_median_seconds"],
            "paired_speedup": result["candidate_speedup"],
        })
        return candidate_score, candidate_score, "exact counts; timed batch accepted", metrics
    except subprocess.TimeoutExpired:
        return 0.0, 0.0, "build or counting timed out", _metrics(0.0, 0.0)
    except isolate.IsolateError:
        raise  # the boundary could not be built: a verification failure, not a score
    except Exception as exc:
        reason = str(exc).strip() or exc.__class__.__name__
        return 0.0, 0.0, reason[:180], _metrics(0.0, 0.0)
    finally:
        for owned in (trees["naive"], trees["reference"]):
            shutil.rmtree(owned.parent, ignore_errors=True)
