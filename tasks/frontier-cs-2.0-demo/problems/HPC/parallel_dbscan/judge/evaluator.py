"""Judge of parallel_dbscan for the shared verifier (#757): check_tree(...) and evaluate(tree).

`tree` is the starter repo (`pristine/`) with only `code/DBSCAN.C` and `code/frontier_solution/**`
overlaid; check_tree() rejects a DBSCAN.C changed outside its public reconstruction region. The
reference is `pristine/` + reference.patch, built and measured in the same paired runs.

Score: C = geometric mean over the small workloads (4 train, 8 final) of the candidate's workload
time (median of 8 repeat medians of 5 fresh `-r 1` totaltime runs); raw_score = score_unbounded
= 100 * 0.020 / (0.020 + C). The 200,000-point workloads and the reference do not enter it.
Any correctness, build or run failure is a scored zero; a train evaluation over 240 seconds
raises (reported as failed, as the old adapter did). The submission is built and every candidate
execution runs in a fresh shared isolate instance (no judge access, private TMPDIR, nothing
survives the run); the reference and the oracles are trusted and run directly.

Beat-spec metric (beat_spec_workload_passes, one row per workload, 6 train / 12 final):
- elapsed_seconds, lower is better: passes when the candidate's workload time C_w is strictly
  below the reference's R_w; no tolerance.
beats_reference = 1 iff every workload passes. reference_score applies the same formula to the
reference's geometric mean; margin = raw_score - reference_score.

The full prior specification is archived at
docs/task-authoring/parallel_dbscan/provenance/beat_spec.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import isolate  # the shared launcher, beside edit_scope in the Verifier base

TASK_DIR = Path(__file__).resolve().parent
PRISTINE = TASK_DIR / "pristine"
AUTHENTIC_ENTRYPOINT = Path("code/DBSCANTime.C")
WORKERS = 8
REPEATS = 8
TIMED_ROUNDS = 5
BEAT_MARGIN = 0.0  # M004: beat_spec strict improvement; no tolerance.
BUILD_TIMEOUT = 90
RUN_TIMEOUT = 45
ORACLE_TIMEOUT = 120
# Beat spec v3: paper-regime seed-spreader workloads (Section 7, SS datasets).
WALK_DOMAIN = 100000.0  # coordinates in [0, 1e5)^d
WALK_RESET = 100        # points emitted at one spreader location before it moves
WALK_RADIUS = 100.0     # half-width of the emission hypercube
WALK_NOISE = 10000      # one uniform noise point per 10,000 points
WALK_RESTARTS = 10      # expected number of random restarts (clusters)


class EvaluatorRuntimeError(RuntimeError):
    """Failure of the trusted reference/oracle, distinct from candidate invalidity."""


# The old task adapter killed a train evaluation after 240 seconds and reported the
# verification as failed (not a scored zero); final ran to its 600-second budget. The same
# train limit is kept here: past it the evaluation raises, so the shared verifier reports a
# failure exactly as before. Final keeps the platform's eval_timeout_seconds (600).
TRAIN_DEADLINE_SECONDS = 240


class EvaluationDeadline(EvaluatorRuntimeError):
    """The train evaluation exceeded the task's 240-second limit."""


def _deadline(signum, frame):
    raise EvaluationDeadline("train evaluation exceeded 240 seconds")


def _trusted(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        raise EvaluatorRuntimeError("trusted evaluator component failed") from exc


def _invalid(reason: str):
    return 0.0, 0.0, reason, {}


def _suite(final: bool):
    seeds = (73471, 73483, 73517, 73529) if final else (19301, 19309, 19333, 19373)
    specs = [
        ("ss-sim-3d", 3, 12000, 0.56, 8, "sim"),
        ("ss-var-3d", 3, 14000, 0.62, 10, "var"),
        ("uniform-7d", 7, 12000, 0.78, 5, "uniform"),
        ("ss-var-7d", 7, 14000, 0.92, 8, "var"),
    ]
    return [dict(id=f"{name}-{i}", dim=d, n=n, eps=e, minpts=m, family=f, seed=seeds[i])
            for i, (name, d, n, e, m, f) in enumerate(specs)]


def _evaluation_suite(final: bool):
    if not final:
        return _suite(False)
    # M004: remeasure both original partitions, with no new shapes or seeds.
    return [dict(spec, id=f"{partition}-{spec['id']}")
            for partition, hidden in (("public", False), ("hidden", True))
            for spec in _suite(hidden)]


def _large_suite(final: bool):
    # Beat spec v3: paper-regime seed-spreader instances at 200,000 points for
    # the paper's 5D-SS-varden and 7D-SS-simden families (minPts 10, the
    # paper's Figure 5 default for both). Hidden instances keep the shapes and
    # use distinct seeds.
    seeds = (88261, 88289) if final else (52363, 52369)
    specs = [
        ("walk-varden-5d", 5, 200000, 800.0, 10, "walk-var"),
        ("walk-simden-7d", 7, 200000, 1600.0, 10, "walk-sim"),
    ]
    return [dict(id=f"{name}-{i}", dim=d, n=n, eps=e, minpts=m, family=f, seed=seeds[i],
                 restarts=WALK_RESTARTS)
            for i, (name, d, n, e, m, f) in enumerate(specs)]


def _large_evaluation_suite(final: bool):
    if not final:
        return _large_suite(False)
    return [dict(spec, id=f"{partition}-{spec['id']}")
            for partition, hidden in (("public", False), ("hidden", True))
            for spec in _large_suite(hidden)]


def _walk_points(spec):
    """Seed-spreader points (Gan and Tao's generator family used by the paper).

    A spreader emits WALK_RESET points uniformly in a hypercube of half-width r
    around its location, then moves 50*d in a uniformly random direction; with
    probability WALK_RESTARTS/n per point it restarts at a uniform location.
    Similar density keeps r = 100; variable density redraws r from
    {100, 200, 400} at every restart. n // WALK_NOISE points are uniform noise.
    """
    if spec["family"] not in ("walk-sim", "walk-var"):
        raise ValueError("unknown walk family")
    rng = random.Random(spec["seed"])
    d, n = spec["dim"], spec["n"]
    noise = n // WALK_NOISE
    clustered = n - noise
    restart = spec["restarts"] / clustered
    shift = 50.0 * d
    radii = ((WALK_RADIUS,) if spec["family"] == "walk-sim"
             else (WALK_RADIUS, 2 * WALK_RADIUS, 4 * WALK_RADIUS))
    location = [rng.random() * WALK_DOMAIN for _ in range(d)]
    radius = radii[0]
    points = []
    for i in range(clustered):
        if i and rng.random() < restart:
            location = [rng.random() * WALK_DOMAIN for _ in range(d)]
            radius = rng.choice(radii)
        elif i and i % WALK_RESET == 0:
            direction = [rng.gauss(0.0, 1.0) for _ in range(d)]
            norm = math.sqrt(sum(x * x for x in direction)) or 1.0
            location = [a + shift * b / norm for a, b in zip(location, direction)]
        points.append([c + (2.0 * rng.random() - 1.0) * radius for c in location])
    points.extend([rng.random() * WALK_DOMAIN for _ in range(d)] for _ in range(noise))
    return points


def _evaluation_correctness_cases(final: bool):
    public = _correctness_cases(False)
    # The first three fixed probes are identical in both partitions.
    return public + _correctness_cases(True)[3:] if final else public


def _points(spec):
    rng = random.Random(spec["seed"])
    d, n, family = spec["dim"], spec["n"], spec["family"]
    pts = []
    if family == "uniform":
        side = n ** (1.0 / d) * 1.15
        pts = [[rng.random() * side for _ in range(d)] for _ in range(n)]
    else:
        centers = [[(k * 3.7 + j * 1.3) % 17 for j in range(d)] for k in range(12)]
        for i in range(n):
            k = i % len(centers)
            sigma = 0.22 if family == "sim" else 0.08 + 0.035 * k
            pts.append([centers[k][j] + rng.gauss(0, sigma) for j in range(d)])
        for i in range(0, min(24, n), 2):
            pts[i + 1] = list(pts[i])
    return pts


def _correctness_cases(final: bool):
    # Fixed exact probes: duplicates, epsilon equality, sparse noise, shared border,
    # and a multi-cell chain. Final adds a distinct seeded 3D/7D clustered probe.
    cases = [
        ("duplicates-boundary", [[0,0],[0,0],[1,0],[2,0],[9,9]], 1.0, 2),
        ("shared-border", [[-1,0],[-1,.1],[-1,-.1],[-1.1,0],[-.9,0],
                           [1,0],[1,.1],[1,-.1],[1.1,0],[.9,0],[0,0]], .91, 4),
        ("chain-noise", [[i*.75,0,0] for i in range(14)] + [[30,30,30]], .8, 2),
    ]
    rng = random.Random(998244353 if final else 424242)
    for d in (3, 7):
        pts = []
        for c in range(3):
            center = [c * 4.0] + [c * .3] * (d - 1)
            pts += [[center[j] + rng.gauss(0, .17) for j in range(d)] for _ in range(24)]
        pts += [[20 + i + j for j in range(d)] for i in range(3)]
        cases.append((f"seeded-{d}d", pts, .7, 5))
    return cases


def _write_points(path: Path, points):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"dim {len(points[0])}\n")
        for p in points:
            f.write(" ".join(f"{x:.12g}" for x in p) + "\n")


def _oracle(points, eps, minpts):
    n = len(points); e2 = eps * eps
    near = [[sum((a-b)**2 for a,b in zip(points[i], points[j])) <= e2
             for j in range(n)] for i in range(n)]
    core = [sum(row) >= minpts for row in near]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        if core[i]:
            for j in range(i):
                if core[j] and near[i][j]:
                    a,b=find(i),find(j)
                    if a != b: parent[a] = b
    comps = {i: frozenset(j for j in range(n) if core[j] and find(j)==find(i)) for i in range(n) if core[i]}
    status=[]; memberships=[]
    for i in range(n):
        if core[i]: status.append("core"); memberships.append(frozenset([comps[i]]))
        else:
            ms=frozenset(comps[j] for j in range(n) if core[j] and near[i][j])
            status.append("border" if ms else "noise"); memberships.append(ms)
    return status, memberships


_COMPONENTS = {}


def _canonical(labels, expected_status):
    core_by_label = {}
    for i, (values, status) in enumerate(zip(labels, expected_status)):
        if status == "noise":
            if values is not None:
                raise ValueError("noise point assigned")
        elif values is None or not values or len(set(values)) != len(values):
            raise ValueError("missing or duplicate memberships")
        elif status == "core":
            if len(values) != 1:
                raise ValueError("core point has multiple labels")
            core_by_label.setdefault(values[0], set()).add(i)
    components = {}
    for label, points in core_by_label.items():
        component = frozenset(points)
        # Equal components share one object, so comparing two canonical results
        # (once per execution) does not re-compare every component point by point.
        components[label] = _COMPONENTS.setdefault(component, component)
    memberships = []
    for values in labels:
        memberships.append(frozenset(components[label] for label in values)
                           if values is not None else frozenset())
    return list(expected_status), memberships


def _parse_output(path: Path, n: int, expected_status):
    if not path.is_file():
        raise ValueError("missing output file")
    labels = [None] * n
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        i = int(fields[0]); values = tuple(int(x) for x in fields[1:])
        if not 0 <= i < n or labels[i] is not None or any(x < 0 for x in values):
            raise ValueError("malformed output")
        labels[i] = values
    return _canonical(labels, expected_status)


def _expected(checker: Path, points, eps, minpts):
    with tempfile.TemporaryDirectory() as directory:
        inp = Path(directory) / "points.txt"
        _write_points(inp, points)
        result = subprocess.run([str(checker), str(inp), str(eps), str(minpts)],
                                capture_output=True, text=True, timeout=RUN_TIMEOUT, check=True)
    labels, status = [], []
    for i, line in enumerate(result.stdout.splitlines()):
        fields = line.split()
        if int(fields[0]) != i or fields[1] not in {"core", "border", "noise"}:
            raise ValueError("invalid independent oracle output")
        status.append(fields[1])
        labels.append(tuple(map(int, fields[2:])) if fields[1] != "noise" else None)
    if len(labels) != len(points):
        raise ValueError("incomplete independent oracle output")
    return _canonical(labels, status)


def _canonical_linear(labels, expected_status):
    # Same acceptance as _canonical, in linear time for the 200,000-point
    # workloads: a core component is named by its smallest core index, which
    # identifies the component's core-point set uniquely.
    first = {}
    for i, (values, status) in enumerate(zip(labels, expected_status)):
        if status == "noise":
            if values is not None:
                raise ValueError("noise point assigned")
        elif values is None or not values or len(set(values)) != len(values):
            raise ValueError("missing or duplicate memberships")
        elif status == "core":
            if len(values) != 1:
                raise ValueError("core point has multiple labels")
            first.setdefault(values[0], i)
    memberships = []
    for values in labels:
        if values is None:
            memberships.append(())
        elif len(values) == 1:
            if values[0] not in first:
                raise ValueError("membership names no core component")
            memberships.append((first[values[0]],))
        else:
            if any(label not in first for label in values):
                raise ValueError("membership names no core component")
            memberships.append(tuple(sorted(first[label] for label in values)))
    return list(expected_status), memberships


def _read_labels(path: Path, n: int):
    if not path.is_file():
        raise ValueError("missing output file")
    labels = [None] * n
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            i = int(fields[0]); values = tuple(int(x) for x in fields[1:])
            if not 0 <= i < n or labels[i] is not None or any(x < 0 for x in values):
                raise ValueError("malformed output")
            labels[i] = values
    return labels


def _expected_large(checker: Path, inp: Path, n: int, eps, minpts):
    result = subprocess.run([str(checker), str(inp), str(eps), str(minpts)],
                            capture_output=True, text=True, timeout=ORACLE_TIMEOUT, check=True)
    labels, status = [], []
    for i, line in enumerate(result.stdout.splitlines()):
        fields = line.split()
        if int(fields[0]) != i or fields[1] not in {"core", "border", "noise"}:
            raise ValueError("invalid independent oracle output")
        status.append(fields[1])
        labels.append(tuple(map(int, fields[2:])) if fields[1] != "noise" else None)
    if len(labels) != n:
        raise ValueError("incomplete independent oracle output")
    return _canonical_linear(labels, status)


def _execute(binary: Path, sandboxed: bool, inp: Path, eps, minpts, rounds, td: Path):
    """One DBSCAN execution; returns (stdout, output path). A candidate binary runs in a fresh
    isolate instance that reads only itself and the input and writes only its own dirs."""
    out=td/"output.txt"
    omp={"OMP_NUM_THREADS":str(WORKERS),"OMP_PROC_BIND":"true","OMP_PLACES":"cores"}
    args=["-eps",str(eps),"-minpts",str(minpts),"-r",str(rounds),"-o"]
    if not sandboxed:
        p=subprocess.run([str(binary),*args,str(out),str(inp)],cwd=binary.parent,env={**os.environ,**omp},
                         capture_output=True,text=True,timeout=RUN_TIMEOUT)
        if p.returncode: raise RuntimeError("DBSCAN execution failed")
        return p.stdout, out
    r=isolate.run([str(binary),*args,"{out}/output.txt",str(inp)],timeout_seconds=RUN_TIMEOUT,
                       read_paths=[str(binary),str(inp)],outputs=["output.txt"],env=omp)
    if r.timed_out: raise RuntimeError("DBSCAN execution timed out")
    if r.returncode or r.limit: raise RuntimeError("DBSCAN execution failed")
    if r.outputs.get("output.txt") is not None:
        out.write_bytes(r.outputs["output.txt"])
    return r.stdout.decode("utf-8","replace"), out


def _run_large(binary: Path, sandboxed: bool, workdir: Path, inp: Path, n: int, eps, minpts, rounds, expected):
    # The input file is written once per workload and shared by both implementations.
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        stdout, out = _execute(binary, sandboxed, inp, eps, minpts, rounds, Path(td))
        times=[]
        for line in stdout.splitlines():
            if line.startswith(">totaltime*"):
                times.append(float(line.split("*")[1]))
        if len(times)!=rounds or any(not math.isfinite(x) or x<=0 for x in times): raise RuntimeError("invalid timing output")
        counts = {}
        for line in stdout.splitlines():
            for kind in ("core", "border", "noise"):
                if line.startswith(">num" + kind + "*"):
                    counts[kind] = int(line.split("*")[1])
        status = expected[0]
        if counts != {kind: status.count(kind) for kind in ("core", "border", "noise")}:
            raise ValueError("core/border/noise counts disagree")
        if _canonical_linear(_read_labels(out, n), status) != expected:
            raise ValueError("timed workload correctness check failed")
        return times


def _measure_large(spec, candidate: Path, refbin: Path, checker: Path, workdir: Path):
    """Paired timing of one large workload; None if the candidate is invalid.

    Same protocol as the original workloads: one -r 1 warm-up per
    implementation, then REPEATS alternating groups of TIMED_ROUNDS fresh -r 1
    executions, every execution fully checked against the independent oracle.
    `candidate` and `refbin` are the two DBSCAN binaries (the candidate's runs sandboxed).
    """
    inp = Path(_trusted(tempfile.mkdtemp, prefix="walk-", dir=workdir)) / "input.txt"
    _trusted(_write_points, inp, _trusted(_walk_points, spec))
    expected = _trusted(_expected_large, checker, inp, spec["n"], spec["eps"], spec["minpts"])

    def execute(repo):
        args = (repo, repo != refbin, workdir, inp, spec["n"], spec["eps"], spec["minpts"], 1, expected)
        if repo == refbin:
            return _trusted(_run_large, *args)
        try:
            return _run_large(*args)
        except (isolate.IsolateError, EvaluationDeadline):
            raise
        except Exception:
            return None

    for repo in (candidate, refbin):
        if execute(repo) is None:
            return None
    ct=[]; rt=[]
    for r in range(REPEATS):
        order=((candidate,ct),(refbin,rt)) if r%2==0 else ((refbin,rt),(candidate,ct))
        for repo,bucket in order:
            # Each measured round is its own process: nothing a round computes
            # can be reused by a later round.
            ts = []
            for _ in range(TIMED_ROUNDS):
                t = execute(repo)
                if t is None:
                    return None
                ts.extend(t)
            bucket.append(statistics.median(ts))
    shutil.rmtree(inp.parent, ignore_errors=True)
    return statistics.median(ct), statistics.median(rt)


def _region_preserved(candidate: Path) -> bool:
    """`code/DBSCAN.C` keeps the retained orchestration around its reconstruction region (or
    is the exact upstream file)."""
    source = (candidate / "code/DBSCAN.C").read_bytes()
    baseline = (PRISTINE / "code/DBSCAN.C").read_bytes()
    prefix, rest = baseline.split(b"// Public reconstruction interfaces.", 1)
    _, suffix = rest.split(b"/**\n * Assigns border points after assigning core points to clusters.", 1)
    prefix += b"// Public reconstruction interfaces."
    suffix = b"/**\n * Assigns border points after assigning core points to clusters." + suffix
    lock = json.loads((TASK_DIR / "source_lock.json").read_text())
    exact_upstream = hashlib.sha256(source).hexdigest() == lock["upstream_dbscan_sha256"]
    return exact_upstream or (source.startswith(prefix) and source.endswith(suffix))


def check_tree(tree, pristine, changes):
    """Shared-verifier hook: only the reconstruction region of code/DBSCAN.C may change."""
    if not _region_preserved(Path(tree)):
        return "out_of_scope", ["code/DBSCAN.C"]
    return None


def _build(repo: Path):
    # The upstream Makefile compiles the retained code/DBSCANTime.C entrypoint
    # into the DBSCAN CLI. Refuse any tree in which that authentic spine is gone.
    if not (repo / AUTHENTIC_ENTRYPOINT).is_file():
        raise RuntimeError("authentic DBSCAN entrypoint is missing")
    subprocess.run(["make", "clean"], cwd=repo/"code", capture_output=True, timeout=BUILD_TIMEOUT)
    p=subprocess.run(["make", "OPENMP=1", "-j8"], cwd=repo/"code", capture_output=True, timeout=BUILD_TIMEOUT)
    if p.returncode: raise RuntimeError("build failed")
    return repo/"code"/"DBSCAN"


def _build_candidate(tree: Path, workdir: Path) -> Path:
    """The same build in an isolate instance (a private copy of the tree); returns the
    judge-held copy of the binary."""
    if not (tree / AUTHENTIC_ENTRYPOINT).is_file():
        raise RuntimeError("authentic DBSCAN entrypoint is missing")
    r=isolate.run(["sh","-c",'make clean >/dev/null 2>&1; make OPENMP=1 -j8 >/dev/null 2>&1 && cp DBSCAN "$0"',
                        "{out}/DBSCAN"],timeout_seconds=BUILD_TIMEOUT,tree=str(tree),cwd="{work}/code",
                       outputs=["DBSCAN"],output_bytes=64<<20)
    if r.returncode or r.limit or not r.outputs.get("DBSCAN"): raise RuntimeError("build failed")
    binary=Path(tempfile.mkdtemp(prefix="candidate-",dir=workdir))/"DBSCAN"
    binary.write_bytes(r.outputs["DBSCAN"]); binary.chmod(0o755)
    return binary


def _run(binary: Path, sandboxed: bool, workdir: Path, points, eps, minpts, rounds=1, expected_status=None, shared=None):
    if expected_status is None:
        raise ValueError("expected point status is required for every execution")
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        if shared is None:
            inp=Path(td)/"input.txt"
            _write_points(inp, points)
        else:
            # A timing workload's input is written once and shared by both implementations.
            inp = shared
        stdout, out = _execute(binary, sandboxed, inp, eps, minpts, rounds, Path(td))
        times=[]
        for line in stdout.splitlines():
            if line.startswith(">totaltime*"):
                times.append(float(line.split("*")[1]))
        if len(times)!=rounds or any(not math.isfinite(x) or x<=0 for x in times): raise RuntimeError("invalid timing output")
        counts = {}
        for line in stdout.splitlines():
            for kind in ("core", "border", "noise"):
                if line.startswith(">num" + kind + "*"):
                    counts[kind] = int(line.split("*")[1])
        if counts != {kind: expected_status.count(kind) for kind in ("core", "border", "noise")}:
            raise ValueError("core/border/noise counts disagree")
        return times, _parse_output(out,len(points),expected_status)


def _p90(values):
    s=sorted(values); return s[math.ceil(.9*len(s))-1]


def _continuous_score(candidate_geomean):
    # M010: fixed physical 20ms anchor, never fitted to the reference.
    if not math.isfinite(candidate_geomean) or candidate_geomean <= 0:
        raise ValueError("candidate geometric mean must be finite positive seconds")
    raw=100.0*0.020/(0.020+candidate_geomean)
    return raw, raw


def evaluate(tree: str):
    if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final":
        return _evaluate(tree)
    previous = signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, TRAIN_DEADLINE_SECONDS)
    try:
        return _evaluate(tree)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _evaluate(tree: str):
    refroot=None
    try:
        candidate=Path(tree)
        refbase=Path(_trusted(tempfile.mkdtemp,prefix="dbscan-ref-",dir=TASK_DIR)); refroot=refbase/"repo"
        _trusted(shutil.copytree,PRISTINE,refroot)
        git_env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(refbase)}
        rp=_trusted(subprocess.run,["git","apply",str((TASK_DIR/"reference.patch").resolve())],cwd=refroot,capture_output=True,env=git_env,timeout=BUILD_TIMEOUT)
        if rp.returncode: raise EvaluatorRuntimeError("reference could not be prepared")
        candidate=_build_candidate(Path(tree), refbase); refbin=_trusted(_build,refroot)
        checker = refbase / "exact-oracle"
        _trusted(subprocess.run,["g++", "-O2", "-std=c++17", str(TASK_DIR / "exact_oracle.cpp"), "-o", str(checker)],
                       capture_output=True, timeout=BUILD_TIMEOUT, check=True)
        grid_checker = refbase / "grid-oracle"
        _trusted(subprocess.run,["g++", "-O2", "-std=c++17", "-fopenmp", str(TASK_DIR / "grid_oracle.cpp"),
                                 "-o", str(grid_checker)], capture_output=True, timeout=BUILD_TIMEOUT, check=True)
        final=os.environ.get("FRONTIER_SUBMISSION_ROLE")=="final"
        # `candidate` / `refbin` are the two binaries; the candidate's runs are sandboxed.
        def candidate_run(binary, *args, **kwargs):
            return _run(binary, True, refbase, *args, **kwargs)
        def reference_run(binary, *args, **kwargs):
            return _trusted(_run, binary, False, refbase, *args, **kwargs)
        for _,pts,eps,m in _evaluation_correctness_cases(final):
            quantized = [[float(f"{value:.12g}") for value in point] for point in pts]
            expected = _trusted(_expected,checker, pts, eps, m)
            if expected != _trusted(_oracle,quantized, eps, m):
                raise EvaluatorRuntimeError("independent oracles disagree")
            for repo in (candidate, refbin):
                _, got = (reference_run(repo, pts, eps, m, expected_status=expected[0])
                          if repo == refbin else candidate_run(repo, pts, eps, m, expected_status=expected[0]))
                if got != expected:
                    if repo == refbin:
                        raise EvaluatorRuntimeError("reference correctness failed")
                    return _invalid("exact DBSCAN correctness check failed")
        cand=[]; ref=[]; rows=[]
        for spec in _evaluation_suite(final):
            pts=_points(spec)
            expected = _trusted(_expected,checker, pts, spec["eps"], spec["minpts"])
            inp = Path(_trusted(tempfile.mkdtemp, prefix="small-", dir=refbase)) / "input.txt"
            _trusted(_write_points, inp, pts)
            shared = inp
            # Every execution, including warm-ups, must produce the full exact output.
            for repo in (candidate, refbin):
                _, got = (reference_run(repo, pts, spec["eps"], spec["minpts"], expected_status=expected[0], shared=shared)
                          if repo == refbin else candidate_run(repo, pts, spec["eps"], spec["minpts"], expected_status=expected[0], shared=shared))
                if got != expected:
                    if repo == refbin:
                        raise EvaluatorRuntimeError("reference correctness failed")
                    return _invalid("timed workload correctness check failed")
            ct=[]; rt=[]
            for r in range(REPEATS):
                order=((candidate,ct),(refbin,rt)) if r%2==0 else ((refbin,rt),(candidate,ct))
                for repo,bucket in order:
                    # Each measured round is its own process: nothing a round
                    # computes can be reused by a later round.
                    ts = []
                    for _ in range(TIMED_ROUNDS):
                        t, got = (reference_run(repo, pts, spec["eps"], spec["minpts"],
                                                expected_status=expected[0], shared=shared) if repo == refbin else
                                  candidate_run(repo, pts, spec["eps"], spec["minpts"],
                                                expected_status=expected[0], shared=shared))
                        if got != expected:
                            if repo == refbin:
                                raise EvaluatorRuntimeError("reference correctness failed")
                            return _invalid("timed workload correctness check failed")
                        ts.extend(t)
                    # Stabilize each sample before taking the outer paired median.
                    bucket.append(statistics.median(ts))
            shutil.rmtree(inp.parent, ignore_errors=True)
            cm=statistics.median(ct); rm=statistics.median(rt)
            cand.append(cm); ref.append(rm)
            rows.append({"workload":hashlib.sha256(spec["id"].encode()).hexdigest()[:16],
                         "metrics":{"elapsed_seconds":bool(cm < rm)}})
        # M010 continuous score stays the geometric mean over the original
        # workloads; the v3 paper-regime workloads enter the strict verdict.
        cg=math.exp(statistics.mean(math.log(x) for x in cand)); rg=math.exp(statistics.mean(math.log(x) for x in ref))
        for spec in _large_evaluation_suite(final):
            measured = _measure_large(spec, candidate, refbin, grid_checker, refbase)
            if measured is None:
                return _invalid("timed workload correctness check failed")
            cm, rm = measured
            cand.append(cm); ref.append(rm)
            rows.append({"workload":hashlib.sha256(spec["id"].encode()).hexdigest()[:16],
                         "metrics":{"elapsed_seconds":bool(cm < rm)}})
        score,raw=_continuous_score(cg)
        reference_score,_=_continuous_score(rg)
        metrics={
            "beat_spec_workload_passes":rows,
            "beats_reference":int(all(x["metrics"]["elapsed_seconds"] for x in rows)),
            "reference_score":reference_score,
            "margin":score-reference_score,
            "candidate_geomean_seconds":cg,
            "reference_geomean_seconds":rg,
            "candidate_elapsed_seconds_median":statistics.median(cand),
            "candidate_elapsed_seconds_p90":_p90(cand),
            "development_elapsed_seconds_by_workload":cand if not final else [],
            "reference_elapsed_seconds_by_workload":ref if not final else [],
            "elapsed_seconds_ratio_by_workload":[a/b for a,b in zip(cand,ref)],
            "elapsed_seconds_margin_by_workload":[b-a for a,b in zip(cand,ref)],
            "development_workload_identities":[] if final else [x["workload"] for x in rows],
            "final_candidate_elapsed_seconds":cand if final else [],
            "final_reference_elapsed_seconds":ref if final else [],
            "final_workload_identities":[x["workload"] for x in rows] if final else [],
        }
        return score,raw,"exact result accepted; paired performance measured",metrics
    except (EvaluatorRuntimeError, isolate.IsolateError):
        raise
    except Exception as exc:
        return _invalid("evaluation failed: "+type(exc).__name__)
    finally:
        if refroot: shutil.rmtree(refroot.parent,ignore_errors=True)


def prepare():
    return {"task":"dbscan_core_connectivity","workers":WORKERS,
            "workloads":len(_suite(False))+len(_large_suite(False)),
            "paired_repeats":REPEATS,"timed_rounds_per_repeat":TIMED_ROUNDS,
            "rounds_per_process":1,
            "beat_margin":BEAT_MARGIN}

