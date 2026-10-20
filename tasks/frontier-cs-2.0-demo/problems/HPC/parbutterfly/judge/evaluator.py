from __future__ import annotations
import bisect, hashlib, itertools, math, os, random, re, shutil, statistics, subprocess, sys, tempfile
from trusted_build import build, bounded, clean_environment
from pathlib import Path
from submission_patch import apply_submission_patch, _make_owner_writable

TASK_DIR = Path(__file__).resolve().parent
AUTHENTIC_ENTRYPOINT = Path("apps/hyper/BipartiteButterflies.C")
AUTHENTIC_BINARY = Path("apps/hyper/BipartiteButterflies")
CONFIG = {'submission': {'allow_paths': ['apps/hyper/butterfly_count_total.h', 'apps/hyper/butterfly_count.h', 'apps/hyper/butterfly_ecount.h', 'apps/hyper/butterfly_peel.h', 'apps/hyper/butterfly_epeel.h', 'apps/hyper/butterfly_putils.h', 'apps/hyper/butterfly_utils.h', 'apps/hyper/solver_extensions/'], 'max_patch_bytes': 524288, 'max_changed_files': 20}}
REFERENCE_HASHES = {'apps/hyper/butterfly_count_total.h': '23b9a2935081c95b054022b9411c556d6b281307773e02e5e0db49061f28f2b7', 'apps/hyper/butterfly_count.h': '322fe6c6b931bd5128b0bac2c365b5fb1da95067b272535d6e05551f8b47f96a', 'apps/hyper/butterfly_ecount.h': 'f376ac752b00f6d84e103a530d73b94fa39ac4b6853a501d3358fce32e867936', 'apps/hyper/butterfly_peel.h': '2deadf7e8bd175d8b2b491878cbc96ae68e7402946f518f9c2e6798256cbe68c', 'apps/hyper/butterfly_epeel.h': 'ff478178626eae5830d3521094d4fdcc8d5c3bd53ab8bce2fa012dcbb03065cd', 'apps/hyper/butterfly_putils.h': '0213e6c55c5504cb72a9bf5b6e88bcef67b2438ad8ee93c455c074ae9c9e9c1f', 'apps/hyper/butterfly_utils.h': '1d4c609f66636e371b81b78e36b5d1dccbaf0c121136026a2c7b81c5807e7ab7'}
BASELINE_HASHES = {'apps/hyper/butterfly_count_total.h': '5fb914a15dc5ecab49f1a248892839cf4fedcd0269e192ccc7678b4e24622e41', 'apps/hyper/butterfly_count.h': '925e076a8427c35b92aa6c27fb3ac07a281b3edceccde49c5ad1ea18a3e844d8', 'apps/hyper/butterfly_ecount.h': '04e7d391e068dcf4e70eecfb5b60efca433421b382825a54717c7610db676855', 'apps/hyper/butterfly_peel.h': '58c7b0d0bfcb1e3c2d3f867015b83ffaae51af839c1a0645c17bdcd68d637ec9', 'apps/hyper/butterfly_epeel.h': '24f02adbf53cbf355ba077a81d75c9209649c0f40cf4547e1b51b648888898bf', 'apps/hyper/butterfly_putils.h': 'a1f7bd5b8e6c8f5de660932b1bddc0b1129a97a94107b0a8d9c756913985ffa9', 'apps/hyper/butterfly_utils.h': 'd6847c831df3bc40cc0b56e67fac32a843a77e192d68e7502703821f34076eae'}
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
TIME_RE = re.compile(r"^FS2_TRUSTED_SECONDS=([0-9.eE+-]+)$", re.M)
COUNT_RE = re.compile(r"number of butterflies:\s*(-?\d+)")

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
    # The retained apps/hyper/BipartiteButterflies.C CLI is the trusted parent.
    return build(repo)

def _run(program, graph: Path, kind: str) -> tuple[int,float]:
    binary, library = program
    cmd=[str(binary), "-countType", kind, "-rankType", "ACOCORE", "-per", "TOTAL", "-r", "0", "-m", str(MAX_WEDGES), "-a", str(MAX_ARRAY), str(graph)]
    env=clean_environment(); env["FS2_WORKER_LIBRARY"]=str(library)
    p=bounded(cmd, cwd=binary.parent, env=env, timeout=45)
    if p.returncode: raise RuntimeError("counting executable failed")
    counts=COUNT_RE.findall(p.stdout); times=TIME_RE.findall(p.stdout)
    if len(counts)!=1 or len(times)!=1:
        raise RuntimeError("counting executable produced invalid output")
    count=int(counts[0]); seconds=float(times[0])
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

def evaluate(solution_path: str):
    candidate=None; reference_root=None; baseline_root=None
    try:
        applied=apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not applied.ok: return 0.0,0.0,applied.message,{}
        candidate=Path(applied.workdir).parent
        crepo=Path(applied.workdir); cbin=_build(crepo)
        reference_root=Path(tempfile.mkdtemp(prefix="butterfly_reference_"))
        rrepo=reference_root/"repo"; shutil.copytree(TASK_DIR/"harbor"/"app",rrepo)
        _make_owner_writable(rrepo)
        ap=subprocess.run(["git","apply",str((TASK_DIR/"reference.patch").resolve())],cwd=rrepo,capture_output=True,text=True,timeout=20,env=dict(clean_environment(), GIT_CEILING_DIRECTORIES=str(reference_root)))
        if ap.returncode: raise RuntimeError("reference restoration failed")
        for relative, expected in REFERENCE_HASHES.items():
            if hashlib.sha256((rrepo/relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError("reference restoration failed")
        rbin=_build(rrepo)
        baseline_root=Path(tempfile.mkdtemp(prefix="butterfly_baseline_"))
        brepo=baseline_root/"repo"; shutil.copytree(TASK_DIR/"harbor"/"app",brepo)
        _make_owner_writable(brepo)
        ap=subprocess.run(["git","apply",str((TASK_DIR/"baseline_naive.patch").resolve())],cwd=brepo,capture_output=True,text=True,timeout=20,env=dict(clean_environment(), GIT_CEILING_DIRECTORIES=str(baseline_root)))
        if ap.returncode: raise RuntimeError("baseline restoration failed")
        for relative, expected in BASELINE_HASHES.items():
            if hashlib.sha256((brepo/relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError("baseline restoration failed")
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
    except ReferenceFailure:
        raise
    except Exception as exc:
        return 0.0,0.0,str(exc) if str(exc) in {"submission does not build","authentic entrypoint is missing","authentic entrypoint did not build","counting executable failed","counting executable produced invalid output","nondeterministic butterfly count","reference restoration failed"} else "evaluation failed",{}
    finally:
        if candidate: shutil.rmtree(candidate,ignore_errors=True)
        if reference_root: shutil.rmtree(reference_root,ignore_errors=True)
        if baseline_root: shutil.rmtree(baseline_root,ignore_errors=True)

if __name__=="__main__":
    try:
        result=evaluate(sys.argv[1]); print(result[0]); print(result[2]); print(result[3])
    except Exception:
        print(0.0); print("invalid invocation")
