"""Judge of parbutterfly: evaluate(tree) on the shared verifier.

`tree` is pristine/ with only the seven butterfly headers and files below
apps/hyper/solver_extensions/ overlaid; it is copied and built by trusted_build.py (the
candidate as a worker library the pristine BipartiteButterflies CLI loads). The library
compile and every counting run execute in fresh shared isolate instances; counting runs
use syscalls="no_fork" (threads, no new processes).
The reference (pristine/ + reference.patch) and the frozen naive baseline
(pristine/ + baseline_naive.patch) are built the same way; every graph is generated, counted
exactly (serial oracle) and timed over 31 alternating pairs.

Score: s_i = median(baseline_i / candidate_i) per graph; S = geomean of s_i over the uniform-
degree graphs; raw_score = score_unbounded = 0 if any count is inexact or S <= 1, else
100*ln S / (ln 10 + ln S). reference_score is the same formula for the reference.

Metrics (one beat_spec_workload_passes row per graph; development: 4, final: 8):
  butterfly_count   exact count; passes if candidate and reference equal the oracle
  counting_seconds  judge-side timing of CountTotal; lower; passes if exact and
                    median(reference_i / candidate_i) > 1.05
beats_reference = 1 iff every graph passes both (equality fails).
The metric definitions before the task.toml migration are kept in the authoring records.
"""
from __future__ import annotations
import bisect, itertools, math, os, random, re, shutil, signal, statistics, subprocess, sys, tempfile, time
from trusted_build import PRISTINE, build, clean_environment, make_owner_writable as _make_owner_writable
from pathlib import Path
import isolate  # the shared launcher, beside edit_scope in the Verifier base

TASK_DIR = Path(__file__).resolve().parent
AUTHENTIC_ENTRYPOINT = Path("apps/hyper/BipartiteButterflies.C")
AUTHENTIC_BINARY = Path("apps/hyper/BipartiteButterflies")
MATERIAL_SPEEDUP = 1.05

def paired_speedup(candidate_times, reference_times):
    if len(candidate_times) != REPETITIONS or len(reference_times) != REPETITIONS:
        raise RuntimeError("invalid paired timing observations")
    if any(not math.isfinite(t) or t <= 0 for t in (*candidate_times, *reference_times)):
        raise RuntimeError("invalid paired timing observations")
    ratios = [r / c for c, r in zip(candidate_times, reference_times)]
    if any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise RuntimeError("invalid paired timing observations")
    return statistics.median(ratios)


def timing_win(speedup):
    return math.isfinite(speedup) and speedup > MATERIAL_SPEEDUP

def validate_repeated_counts(expected_candidate, expected_reference, expected_baseline,
                             candidate_reference_pair, reference,
                             candidate_baseline_pair, baseline):
    if (candidate_reference_pair != expected_candidate
            or candidate_baseline_pair != expected_candidate
            or reference != expected_reference
            or baseline != expected_baseline):
        raise RuntimeError("nondeterministic butterfly count")

MAX_WEDGES = 200_000_000
MAX_ARRAY = 200_000_000
REPETITIONS = 31
# Workloads are tagged by generator. "uniform" graphs are the v2 suites, byte
# for byte. Beat spec v3 adds "skewed" Chung-Lu graphs with power-law degrees on
# BOTH sides (exponents 0.75-0.85, hubs of 14k-19k neighbours), the regime of
# the paper's real-world graphs where vertex ranking decides the wedge count
# (Shi & Shun, Section 3 and Section 6 ranking experiments).
DEV = [("uniform", 18_000, 20_000, 88, 1103), ("uniform", 20_000, 22_000, 96, 2207),
       ("uniform", 22_000, 24_000, 104, 3301),
       ("skewed", 20_000, 22_000, 1_900_000, 0.8, 0.8, 9005)]
HIDDEN = [("uniform", 19_000, 21_000, 92, 4409), ("uniform", 21_000, 23_000, 100, 5519),
          ("uniform", 23_000, 25_000, 108, 6637),
          ("skewed", 21_000, 23_000, 2_000_000, 0.75, 0.85, 9008)]
FINAL = DEV + HIDDEN
READY = b"FS2_READY\n"
COUNT_RE = re.compile(rb"FS2_COUNT=(-?[0-9]+)\n")

class ReferenceFailure(RuntimeError):
    """The source-locked reference or frozen baseline did not count exactly.

    An evaluator error, never a candidate result: a failed reference cannot set
    a bar and the candidate is not scored against it.
    """


def _graph(path: Path, kind: str, *args) -> None:
    if kind == "uniform":
        _uniform_graph(path, *args)
    elif kind == "skewed":
        _skewed_graph(path, *args)
    else:
        raise ValueError("unknown workload generator")


def _skewed_graph(path: Path, nu: int, nv: int, edges: int, alpha_u: float, alpha_v: float, seed: int) -> None:
    # Deterministic Chung-Lu style simple bipartite graph: endpoints are drawn
    # independently with weights (rank+1)^-alpha on each side, duplicates are
    # dropped until exactly `edges` distinct edges exist, and vertex identifiers
    # are shuffled so identifiers carry no degree order.
    rng = random.Random(seed)
    cum_u = list(itertools.accumulate((i + 1) ** -alpha_u for i in range(nu)))
    cum_v = list(itertools.accumulate((j + 1) ** -alpha_v for j in range(nv)))
    perm_u = list(range(nu)); rng.shuffle(perm_u)
    perm_v = list(range(nv)); rng.shuffle(perm_v)
    seen = set()
    while len(seen) < edges:
        need = edges - len(seen)
        us = rng.choices(range(nu), cum_weights=cum_u, k=need + need // 8 + 16)
        vs = rng.choices(range(nv), cum_weights=cum_v, k=len(us))
        for u, v in zip(us, vs):
            seen.add(perm_u[u] * nv + perm_v[v])
            if len(seen) >= edges:
                break
    rows = [[] for _ in range(nu)]
    for code in sorted(seen):
        rows[code // nv].append(code % nv)
    _write_graph(path, rows, nv)


def _uniform_graph(path: Path, nu: int, nv: int, degree: int, seed: int) -> None:
    # Deterministic simple bipartite graph; both incidence directions are emitted.
    rows=[]
    for u in range(nu):
        vals=set()
        x=(seed ^ (u * 2654435761)) & 0xffffffff
        while len(vals) < degree:
            x=(1664525*x+1013904223) & 0xffffffff
            vals.add(x % nv)
        rows.append(sorted(vals))
    _write_graph(path, rows, nv)


def _write_graph(path: Path, rows, nv: int) -> None:
    nu = len(rows); m = sum(len(r) for r in rows)
    cols=[[] for _ in range(nv)]
    for u, vs in enumerate(rows):
        for v in vs: cols[v].append(u)
    def offsets(groups):
        out=[]; n=0
        for g in groups: out.append(n); n += len(g)
        return out
    values=["AdjacencyHypergraph", str(nu), str(m), str(nv), str(m)]
    values += [str(x) for x in offsets(rows)]
    values += [str(x) for g in rows for x in g]
    values += [str(x) for x in offsets(cols)]
    values += [str(x) for g in cols for x in g]
    path.write_text("\n".join(values)+"\n", encoding="ascii")

def _build(repo: Path):
    # The retained apps/hyper/BipartiteButterflies.C CLI is the pristine counting program.
    return build(repo)

def _run(program, graph: Path, kind: str) -> tuple[int,float]:
    # One fresh isolate instance per run, syscalls="no_fork": the pristine CLI reads and
    # preprocesses the graph, then the candidate worker library counts it in the same process
    # with threads but no new processes; nothing survives the run. The judge times it from
    # outside: 'G' on the worker's stdin to the end of its stdout (count written, process gone).
    binary, library = program
    worker=isolate.start(["{work}/"+binary.name, "-countType", kind, "-rankType", "ACOCORE", "-per", "TOTAL", "-r", "0", "-m", str(MAX_WEDGES), "-a", str(MAX_ARRAY), "{in}/graph.adj"],
                              tree=str(binary.parent), inputs={"graph.adj": str(graph)},
                              env=dict(clean_environment(), FS2_WORKER_LIBRARY="{work}/"+library.name),
                              path="/usr/bin:/bin", timeout_seconds=45, cpu_seconds=180, memory_bytes=8<<30,
                              syscalls="no_fork")
    seconds=None; record=b""
    try:
        before=b""
        while READY not in before and len(before) < 1<<20:
            chunk=worker.stdout.read(1<<16)  # the instance's 45 s limit ends it: never blocks forever
            if not chunk: break
            before+=chunk
        if before.endswith(READY):
            start=time.perf_counter()
            worker.stdin.write(b"G")
            while len(record) < 1<<10:
                chunk=worker.stdout.read(1<<10)
                if not chunk:
                    seconds=time.perf_counter()-start
                    break
                record+=chunk
    except BrokenPipeError:
        pass
    finally:
        result=worker.close()
    # After the record the worker only calls _exit(0); close() may race that exit with SIGKILL.
    if seconds is None or result.limit or result.returncode not in (0, -signal.SIGKILL):
        raise RuntimeError("counting executable failed")
    match=COUNT_RE.fullmatch(record)
    if not match:
        raise RuntimeError("counting executable produced invalid output")
    count=int(match.group(1))
    if count < 0 or not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("counting executable produced invalid output")
    return count, seconds

def _percentile90(xs):
    ys=sorted(xs); return ys[math.ceil(.9*len(ys))-1]

def _score(exact_fraction, baseline_speedups):
    if exact_fraction < 1.0:
        return 0.0
    speedup=math.exp(sum(math.log(x) for x in baseline_speedups)/len(baseline_speedups))
    if speedup <= 1.0:
        return 0.0
    return 100.0*math.log(speedup)/(math.log(10.0)+math.log(speedup))

def uniform_score(suite, exact_fraction, baseline_speedups):
    # The continuous score keeps its v2 definition: the speedup geomean over
    # the uniform-degree graphs. Exactness on every graph stays a validity gate.
    return _score(exact_fraction,[x for spec, x in zip(suite, baseline_speedups) if spec[0]=="uniform"])

def reference_score(suite, samples):
    """The source-locked reference's own continuous score in this judgment.

    The same curve and graphs as the candidate score, applied to the
    reference's paired speedup over the frozen baseline: median of baseline_i /
    reference_i over the 31 repetitions, which time both programs in the same
    repetition. A reference count that is not exact is a ReferenceFailure, so
    the exactness gate always holds here. ``samples`` are the per-graph
    private_timing_samples entries in suite order.
    """
    return uniform_score(suite, 1.0, [paired_speedup(s["reference"], s["baseline"]) for s in samples])

def evaluate(tree: str):
    """`tree` is the pristine starter with only the submission's allowed files overlaid (the
    shared verifier applied and scoped the patch); it is built from a private copy."""
    candidate=None; reference_root=None; baseline_root=None
    affinity=os.sched_getaffinity(0)
    try:
        # Every build and run uses the same four CPUs (inherited by the sandbox instances).
        os.sched_setaffinity(0, sorted(affinity)[:4])
        candidate=Path(tempfile.mkdtemp(prefix="butterfly_candidate_"))
        crepo=candidate/"repo"; shutil.copytree(tree,crepo,symlinks=True)
        _make_owner_writable(crepo)
        cbin=_build(crepo)
        reference_root=Path(tempfile.mkdtemp(prefix="butterfly_reference_"))
        rrepo=reference_root/"repo"; shutil.copytree(PRISTINE,rrepo)
        _make_owner_writable(rrepo)
        ap=subprocess.run(["git","apply",str((TASK_DIR/"reference.patch").resolve())],cwd=rrepo,capture_output=True,text=True,timeout=20,env=dict(clean_environment(), GIT_CEILING_DIRECTORIES=str(reference_root)))
        if ap.returncode: raise RuntimeError("reference restoration failed")
        rbin=_build(rrepo)
        baseline_root=Path(tempfile.mkdtemp(prefix="butterfly_baseline_"))
        brepo=baseline_root/"repo"; shutil.copytree(PRISTINE,brepo)
        _make_owner_writable(brepo)
        ap=subprocess.run(["git","apply",str((TASK_DIR/"baseline_naive.patch").resolve())],cwd=brepo,capture_output=True,text=True,timeout=20,env=dict(clean_environment(), GIT_CEILING_DIRECTORIES=str(baseline_root)))
        if ap.returncode: raise RuntimeError("baseline restoration failed")
        bbin=_build(brepo)
        suite=FINAL if os.environ.get("FRONTIER_SUBMISSION_ROLE")=="final" else DEV
        work=reference_root/"graphs"; work.mkdir()
        rows=[]; candidate_times=[]; reference_times=[]; baseline_times=[]; speedups=[]; baseline_speedups=[]; private_samples={}; private_counts={}; correct=0
        for i,spec in enumerate(suite):
            gid=f"graph-{i+1}"; graph=work/(gid+".adj"); _graph(graph,*spec)
            try:
                oracle,_=_run(rbin,graph,"SERIAL")
                rc,_=_run(rbin,graph,"BATCHS"); bc,_=_run(bbin,graph,"BATCHS")
            except RuntimeError as error:
                raise ReferenceFailure("reference or baseline counting failed") from error
            if rc!=oracle or bc!=oracle:
                raise ReferenceFailure("reference or baseline count is not exact")
            cc,_=_run(cbin,graph,"BATCHS")
            ct=[]; rt=[]; bt=[]; cbt=[]
            for rep in range(REPETITIONS):
                if rep%2==0:
                    cr,t=_run(cbin,graph,"BATCHS"); ct.append(t)
                    r,t=_run(rbin,graph,"BATCHS"); rt.append(t)
                else:
                    r,t=_run(rbin,graph,"BATCHS"); rt.append(t)
                    cr,t=_run(cbin,graph,"BATCHS"); ct.append(t)
                if rep%2==0:
                    b,t=_run(bbin,graph,"BATCHS"); bt.append(t)
                    cb,t=_run(cbin,graph,"BATCHS"); cbt.append(t)
                else:
                    cb,t=_run(cbin,graph,"BATCHS"); cbt.append(t)
                    b,t=_run(bbin,graph,"BATCHS"); bt.append(t)
                validate_repeated_counts(cc,rc,bc,cr,r,cb,b)
            cm=statistics.median(ct); rm=statistics.median(rt); bm=statistics.median(bt)
            exact=(cc==oracle and rc==oracle and bc==oracle); correct += int(exact)
            paired = paired_speedup(ct, rt)
            baseline_paired = paired_speedup(cbt, bt)
            candidate_times.append(cm); reference_times.append(rm); baseline_times.append(bm); speedups.append(paired); baseline_speedups.append(baseline_paired)
            rows.append({"workload":gid,"metrics":{"butterfly_count":bool(exact),"counting_seconds":bool(exact and timing_win(paired))}})
            private_samples[gid]={"candidate_reference_pair":ct,"reference":rt,"candidate_baseline_pair":cbt,"baseline":bt}
            private_counts[gid]={"oracle":oracle,"candidate":cc,"reference":rc,"baseline":bc}
        exact_fraction=correct/len(suite)
        score=uniform_score(suite,exact_fraction,baseline_speedups)
        refscore=reference_score(suite,[private_samples[f"graph-{i+1}"] for i in range(len(suite))])
        beats=int(all(v for row in rows for v in row["metrics"].values()))
        metrics={"beats_reference":beats,"beat_spec_workload_passes":rows,"reference_score":refscore,"margin":score-refscore,
                 "dev_counting_seconds_median":statistics.median(candidate_times),"dev_counting_seconds_p90":_percentile90(candidate_times),
                 "butterfly_count":correct,"counting_seconds":statistics.median(candidate_times),"reference_butterfly_count":len(suite),
                 "reference_counting_seconds":statistics.median(reference_times),
                 "baseline_counting_seconds":statistics.median(baseline_times),"baseline_speedup_geomean":math.exp(sum(math.log(x) for x in baseline_speedups)/len(baseline_speedups)),
                 "private_timing_samples":private_samples,"private_counts_per_workload":private_counts,
                 "oracle_butterfly_count":len(suite),"final_metrics":int(os.environ.get("FRONTIER_SUBMISSION_ROLE")=="final")}
        return score,score,"evaluation completed",metrics
    except (ReferenceFailure, isolate.IsolateError):
        raise
    except Exception as exc:
        return 0.0,0.0,str(exc) if str(exc) in {"submission does not build","authentic entrypoint is missing","authentic entrypoint did not build","counting executable failed","counting executable produced invalid output","nondeterministic butterfly count","reference restoration failed"} else "evaluation failed",{}
    finally:
        os.sched_setaffinity(0, affinity)
        if candidate: shutil.rmtree(candidate,ignore_errors=True)
        if reference_root: shutil.rmtree(reference_root,ignore_errors=True)
        if baseline_root: shutil.rmtree(baseline_root,ignore_errors=True)

if __name__=="__main__":
    try:
        result=evaluate(sys.argv[1]); print(result[0]); print(result[2]); print(result[3])
    except Exception:
        print(0.0); print("invalid invocation")
