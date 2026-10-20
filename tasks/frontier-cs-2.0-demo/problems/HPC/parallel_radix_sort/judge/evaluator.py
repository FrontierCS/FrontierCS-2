import json, math, os, pathlib, shutil, statistics, subprocess, sys, tempfile
_MODULE_DIR = pathlib.Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
from submission_patch import apply_submission_patch, _make_owner_writable

TASK_DIR=_MODULE_DIR
PRISTINE=TASK_DIR.parent / "repo"
if not PRISTINE.is_dir():
    PRISTINE=TASK_DIR.parent / "agent-base/frontier/task/repo"

class InfrastructureError(RuntimeError): pass
CONFIG={"submission":{"allow_paths":["RadixSort/edgelistgraph.h","RadixSort/frontier_solution/"],"max_patch_bytes":262144,"max_changed_files":20}}
AUX_CAP=64*1024*1024
DEV=[("d-int-uniform","int","uniform",3000000,12011),("d-pair-zipf","pair","zipf",2600000,12037),("d-int-duplicates","int","duplicates",3200000,12041),("d-pair-almost","pair","almost",2400000,12049)]
HIDDEN=[("f-int-uniform","int","uniform",3000000,92009),("f-pair-zipf","pair","zipf",2600000,92033),("f-int-duplicates","int","duplicates",3200000,92041),("f-pair-almost","pair","almost",2400000,92051)]
FINAL=DEV+HIDDEN  # Approved fresh public plus original hidden; no new shapes.
# Beat spec v3: paper-regime 16M-key inputs (64 MiB, beyond a typical L3):
# the paper's zipf_0.75 keys over [1, 1e9] and uniform 32-bit keys. They enter
# the strict verdict only; the continuous score stays over DEV/FINAL above.
LARGE_DEV=[("d-int-zipf075","int","zipf075",16000000,12059),("d-int-uniform16m","int","uniform",16000000,12071)]
LARGE_HIDDEN=[("f-int-zipf075","int","zipf075",16000000,92077),("f-int-uniform16m","int","uniform",16000000,92083)]
TRAIN_GRADED=DEV+LARGE_DEV
FINAL_GRADED=FINAL+LARGE_DEV+LARGE_HIDDEN
AUX_FRACTION=0.05  # paper Section 5: in-place sorts use < 5% auxiliary memory
GUARDS=[("int","equal",24000,7103),("pair","sorted",19999,7121),("int","duplicates",1,7127)]

def _compiler(): return str(PRISTINE / "frontier-cxx")
def _build(tree:pathlib.Path,out:pathlib.Path,source="benchmark.cpp",component=False):
    cc=_compiler()
    if not cc: raise RuntimeError("GCC 7 toolchain is unavailable")
    # component=True builds the same harness with the region-graph timer.
    timer=["-DFRONTIER_COMPONENT_TIMER"] if component else []
    cmd=[cc,"-I",str(tree/"RadixSort"),"-std=c++14",str(TASK_DIR/source),*timer,"-DCILKP","-DBITS_HACK","-O3","-flto","-march=native","-fcilkplus","-Wl,--wrap=malloc","-Wl,--wrap=calloc","-Wl,--wrap=realloc","-Wl,--wrap=free","-lcilkrts","-o",str(out)]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=90)
    if p.returncode: raise RuntimeError("build failed: "+p.stderr[-300:].replace("\n"," "))
def _authentic_smoke(tree:pathlib.Path,out:pathlib.Path,td:pathlib.Path):
    cc=_compiler(); src=tree/"RadixSort"/"example.cpp"
    cmd=[cc,"-I",str(tree/"RadixSort"),"-std=c++14",str(src),"-DCILKP","-DBITS_HACK","-O3","-flto","-march=native","-fcilkplus","-lcilkrts","-o",str(out)]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=90)
    if p.returncode: raise RuntimeError("entrypoint build failed")
    data=td/"entry.seq"; data.write_text("sequenceInt\n"+"\n".join(str((i*2654435761)&1023) for i in range(24000))+"\n")
    env={**os.environ,"CILK_NWORKERS":"4"}; p=subprocess.run(["taskset","-c","0-3",str(out),"-r","1","-c",str(data)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30,env=env)
    if p.returncode or "array is sorted!" not in p.stdout or "FAIL" in p.stdout: raise RuntimeError("authentic entrypoint failed")
def _run(binary:pathlib.Path,row,reps=21):
    wid,typ,dist,n,seed=row; env={**os.environ,"CILK_NWORKERS":"4"}
    cmd=["taskset","-c","0-3",str(binary),typ,dist,str(n),str(seed),str(reps)]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35,env=env)
    fields=p.stdout.strip().split()
    if p.returncode or len(fields)!=3 or fields[0]!="OK": raise RuntimeError(f"invalid result on {wid}")
    timing,auxiliary=int(fields[1]),int(fields[2])
    if timing<0 or (timing==0 and not wid.startswith("guard-")) or auxiliary<0:
        raise RuntimeError(f"invalid result on {wid}")
    return timing,auxiliary
def _log_speedup(cand,baseline):
    if not cand or len(cand)!=len(baseline):
        raise ValueError("incomplete continuous measurement")
    if any(not math.isfinite(x) or x<=0 for x in [*cand,*baseline]):
        raise ValueError("nonpositive or nonfinite timing")
    return math.fsum(math.log(b)-math.log(c) for c,b in zip(cand,baseline))/len(cand)

def _score(cand,baseline):
    log_s=_log_speedup(cand,baseline)
    return 100.0*log_s/(math.log(10.0)+log_s) if log_s>0 else 0.0

def reference_score(refs,baselines):
    """The source-locked reference's own continuous score in this judgment.

    The same curve and original workloads as the candidate score, applied to the
    reference's measured sort times against the same std::sort baseline times.
    The reference is always valid here: a reference over the 5% auxiliary bound
    or failing to run is an InfrastructureError.
    """
    return _score(refs,baselines)

def _input_bytes(row):
    return row[3]*(4 if row[1]=="int" else 8)

def _aux_ok(aux,row):
    return aux<=AUX_FRACTION*_input_bytes(row)

def _evaluate_patch(solution_path:str):
    result=apply_submission_patch(solution_path,TASK_DIR,CONFIG,pristine=PRISTINE)
    if not result.ok: return None,result.message,result.metrics
    return pathlib.Path(result.workdir),"",{}
def evaluate(solution_path:str, mode="train"):
  if mode not in {"train", "eval"}: raise ValueError("invalid mode")
  tree = None
  try:
    tree,msg,extra=_evaluate_patch(solution_path)
    if tree is None:return 0.0,0.0,msg,extra
    final=mode=="eval"; suite=FINAL if final else DEV
    graded=FINAL_GRADED if final else TRAIN_GRADED
    with tempfile.TemporaryDirectory() as td:
      td=pathlib.Path(td); cb=td/"candidate"; rb=td/"reference"; _build(tree,cb); _authentic_smoke(tree,td/"radixSort",td)
      cc=td/"candidate-component"; rc=td/"reference-component"; _build(tree,cc,component=True)
      ref_tree=td/"reference-tree"; shutil.copytree(PRISTINE,ref_tree); _make_owner_writable(ref_tree)
      ap=subprocess.run(["git","apply",str(TASK_DIR/"reference.patch")],cwd=ref_tree,env={**os.environ,"GIT_CEILING_DIRECTORIES":str(ref_tree.parent.resolve())},stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
      if ap.returncode: raise InfrastructureError("internal reference patch failed")
      try: _build(ref_tree,rb); _build(ref_tree,rc,component=True)
      except Exception as e: raise InfrastructureError("reference build failed") from e
      bb=td/"baseline"
      try: _build(PRISTINE,bb,"std_sort_baseline.cpp")
      except Exception as e: raise InfrastructureError("std::sort baseline build failed") from e
      for i,g in enumerate(GUARDS):
        _run(cb,(f"guard-{i}",)+g,1)
        _baseline_run(bb,(f"guard-{i}",)+g,1)
      cand=[];refs=[];aux=[];ref_aux=[];baselines=[];baseline_aux=[];cand_graph=[];ref_graph=[]
      for i,row in enumerate(graded):
        # Alternate order to distribute thermal/order bias.
        if i%2: r,ra=_reference_run(rb,row); c,a=_run(cb,row)
        else: c,a=_run(cb,row); r,ra=_reference_run(rb,row)
        # Keep original candidate/reference pairing; measure independent baseline
        # next on the same row, with the same 21-round harness and CPU settings.
        # v3 rows are not part of the continuous score and have no baseline.
        if row in suite:
            b,ba=_baseline_run(bb,row); baselines.append(b); baseline_aux.append(ba)
        # Beat spec v3: region-graph component time from the timer build, in
        # the same alternating order.
        if i%2: rg,_=_reference_run(rc,row); cg,_=_run(cc,row)
        else: cg,_=_run(cc,row); rg,_=_reference_run(rc,row)
        if not _aux_ok(ra,row): raise InfrastructureError("reference exceeds the paper auxiliary-memory bound")
        cand.append(c);refs.append(r);aux.append(a);ref_aux.append(ra);cand_graph.append(cg);ref_graph.append(rg)
      valid=max(aux,default=0)<=AUX_CAP
      passes=[{"workload":row[0],"metrics":{"sort_time_ns":bool(valid and c<r),"region_graph_ns":bool(valid and cg<rg),
               "auxiliary_bytes":bool(valid and _aux_ok(a,row))}}
              for row,c,r,cg,rg,a in zip(graded,cand,refs,cand_graph,ref_graph,aux)]
      beats=int(valid and all(v for x in passes for v in x["metrics"].values()))
      original=len(suite)
      score=_score(cand[:original],baselines) if valid else 0.0
      speedup=math.exp(_log_speedup(cand[:original],baselines))
      refscore=reference_score(refs[:original],baselines)
      metrics={"beats_reference":beats,"beat_spec_workload_passes":passes,"reference_score":refscore,"margin":score-refscore,"geomean_speedup":speedup,"log_geomean_speedup":_log_speedup(cand[:original],baselines),
       "sort_time_ns":statistics.median(cand),"reference_sort_time_ns":statistics.median(refs),
       "measurements":[{"workload":row[0],"candidate_ns":c,"reference_ns":r,"candidate_aux_bytes":a,"reference_aux_bytes":ra,
                        "candidate_region_graph_ns":cg,"reference_region_graph_ns":rg,"input_bytes":_input_bytes(row),
                        **({"baseline_ns":baselines[k],"baseline_aux_bytes":baseline_aux[k]} if k<original else {})}
                       for k,(row,c,r,a,ra,cg,rg) in enumerate(zip(graded,cand,refs,aux,ref_aux,cand_graph,ref_graph))]}
      if final: metrics.update(final_candidate_median_sort_time_ns=statistics.median(cand),final_candidate_max_auxiliary_bytes=max(aux),
                               final_candidate_median_region_graph_ns=statistics.median(cand_graph))
      else: metrics.update(dev_candidate_median_sort_time_ns=statistics.median(cand),dev_candidate_max_auxiliary_bytes=max(aux),
                           dev_candidate_median_region_graph_ns=statistics.median(cand_graph))
      return score,score,"scored complete suite",metrics
  except InfrastructureError: raise
  except Exception as e:return 0.0,0.0,str(e)[:300],{}
  finally:
    if tree is not None:
      shutil.rmtree(tree.parent, ignore_errors=True)

def _reference_run(binary, row):
    try: return _run(binary, row)
    except Exception as e: raise InfrastructureError("reference execution failed") from e

def _baseline_run(binary, row, reps=21):
    try:
        timing,aux=_run(binary,row,reps)
        if aux>AUX_CAP: raise RuntimeError("baseline auxiliary cap exceeded")
        return timing,aux
    except Exception as e: raise InfrastructureError("std::sort baseline execution failed") from e

if __name__=="__main__":
 result = evaluate(sys.argv[1], sys.argv[2])
 pathlib.Path(sys.argv[3]).write_text(json.dumps(result, sort_keys=True))
