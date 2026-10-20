#!/usr/bin/env python3
"""Run original ligra/ligra.h parallel_main -> Compute -> KCore with private output observation."""
from __future__ import annotations
import hashlib, heapq, json, math, os, re, select, shutil, signal, statistics, struct, subprocess, sys, tempfile, time
from pathlib import Path
TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
try:
    import yaml
except ImportError:
    yaml = None
from submission_patch import apply_submission_patch

REPEATS = 7
MIN_SPEEDUP = 0.03
# Beat spec v2: the paper's Table 3 reports both T1 (one thread) and parallel
# times for work-efficient k-core. runtime_seconds keeps the eight-thread
# measurement; sequential_runtime_seconds adds the same program on one thread.
PARALLEL_THREADS = 8
SEQUENTIAL_THREADS = 1
METRICS = ("runtime_seconds", "sequential_runtime_seconds")
# Exactly the headers the original HyperKCore-Efficient.C includes after
# `#define HYPER 1`, in its order; ligra.h (main) and hypergraphIO.h/IO.h (the
# loader) are in hygra.h's include closure.
FRAMEWORK_HEADERS = ("hygra.h", "index_map.h", "bucket.h", "edgeMapReduce.h")


class ReferenceVerificationError(RuntimeError):
    """The withheld reference or another trusted input failed.

    A failed verification, never a candidate score: it propagates to the
    verifier adapter, which reports evaluator_error.
    """
DEV = [("feedback-a", "feedback-a.adj"), ("feedback-b", "feedback-b.adj")]
FINAL = DEV + [("graded-a", "graded-a.adj"), ("graded-b", "graded-b.adj")]
GUARDS = [
    ("isolates", 5, [[0,1]]),
    ("overlap-duplicate", 5, [[0,1,2],[1,2,3],[0,1,2]]),
    ("singletons-cascade", 6, [[0],[0,1],[1,2],[2,3,4],[3,4]]),
]

def _config():
    if yaml: return yaml.safe_load((TASK_DIR/"config.yaml").read_text())
    return {"submission":{"allow_paths":["apps/hyper/HyperKCore-Efficient.C","apps/hyper/frontier_solution/"],"max_patch_bytes":262144,"max_changed_files":20}}

def _test_generated(nv, nh, card, seed):
    # SplitMix-style deterministic membership generation, with deliberately irregular edge sizes.
    x = seed & ((1<<64)-1); edges=[]
    def nxt():
        nonlocal x
        x=(x+0x9E3779B97F4A7C15)&((1<<64)-1); z=x
        z=((z^(z>>30))*0xBF58476D1CE4E5B9)&((1<<64)-1)
        z=((z^(z>>27))*0x94D049BB133111EB)&((1<<64)-1)
        return z^(z>>31)
    for e in range(nh):
        k = max(1, card + int(nxt()%5)-2)
        members=set()
        while len(members)<k: members.add(nxt()%nv)
        edges.append(sorted(members))
    # Ensure a modest isolate tail and a long overlapping cascade component.
    cap=max(1,nv-32)
    edges=[sorted({v%cap for v in edge}) for edge in edges]
    for i in range(31): edges[i]=[i,i+1]
    return edges

def _read_graph(path):
    """Read the exact text format emitted by upstream utils/randHypergraph."""
    with path.open() as source:
        tokens=(token for line in source for token in line.split())
        if next(tokens)!="AdjacencyHypergraph": raise RuntimeError("invalid retained workload")
        nv,mv,nh,mh=(int(next(tokens)) for _ in range(4))
        for _ in range(nv+mv): int(next(tokens))
        oh=[int(next(tokens)) for _ in range(nh)]
        eh=[int(next(tokens)) for _ in range(mh)]
        try: next(tokens); raise RuntimeError("trailing retained workload data")
        except StopIteration: pass
    if mv!=mh: raise RuntimeError("asymmetric retained workload")
    ends=oh[1:]+[mh]
    return nv,[eh[start:end] for start,end in zip(oh,ends)]

def _write_graph(path, nv, edges):
    byv=[[] for _ in range(nv)]
    for e, members in enumerate(edges):
        for v in members: byv[v].append(e)
    ev=[x for row in byv for x in row]; eh=[x for row in edges for x in row]
    ov=[]; p=0
    for row in byv: ov.append(p); p+=len(row)
    oh=[]; p=0
    for row in edges: oh.append(p); p+=len(row)
    values=["AdjacencyHypergraph",str(nv),str(len(ev)),str(len(edges)),str(len(eh))]
    values += map(str,ov); values += map(str,ev); values += map(str,oh); values += map(str,eh)
    path.write_text("\n".join(values)+"\n")

def _oracle(nv, edges):
    incident=[[] for _ in range(nv)]
    for e, members in enumerate(edges):
        for v in members: incident[v].append(e)
    deg=[len(x) for x in incident]; live_e=[True]*len(edges); live_v=[True]*nv
    heap=[(deg[v],v) for v in range(nv)]; heapq.heapify(heap); core=[0]*nv
    left=nv
    while left:
        d,v=heapq.heappop(heap)
        if not live_v[v] or d!=deg[v]: continue
        live_v[v]=False; left-=1; core[v]=d
        for e in incident[v]:
            if not live_e[e]: continue
            live_e[e]=False
            for u in edges[e]:
                if live_v[u] and deg[u]>d:
                    deg[u]=max(d,deg[u]-1); heapq.heappush(heap,(deg[u],u))
    return core

def _observed(text):
    # The public Compute body stays exact; only this private copy observes its result.
    pristine=(TASK_DIR/"harbor/app/apps/hyper/HyperKCore-Efficient.C").read_text()
    marker="template <class vertex>\nvoid Compute("
    frozen=marker+pristine.split(marker,1)[1]
    if text.count(frozen)!=1: raise RuntimeError("original Compute wiring was changed")
    needle="  auto cores = KCore(GA, num_buckets);"
    if text.count(needle)!=1: raise RuntimeError("KCore result boundary was changed")
    observation='\n  if (GA.nv && !cores.s) std::exit(2);\n  std::ostringstream frontier_observation; frontier_observation << "FRONTIER_CORES";\n  for (size_t i=0;i<GA.nv;i++) frontier_observation << " " << cores.s[i];\n  std::cerr << frontier_observation.str() << std::endl;'
    return "#include <sstream>\n"+text.replace(needle,needle+observation)

def _compile(repo, out):
    text=_observed((Path(repo)/"apps/hyper/HyperKCore-Efficient.C").read_text())
    # Baseline: the pristine editable file (the KCore stub, no solver code),
    # observed and compiled with exactly the same flags and paths.
    baseline=_observed((TASK_DIR/"harbor/app/apps/hyper/HyperKCore-Efficient.C").read_text())
    use=Path(out).with_suffix(".C"); use.write_text(text)
    base=Path(out).with_name(Path(out).name+"-baseline.C"); base.write_text(baseline)
    # The fixed framework (driver, loader, hygra) is parsed before any line of
    # the editable translation unit: the original file's prologue is
    # force-included, so macros the solver defines later cannot rewrite
    # framework or loader code, and the solver's own includes of these headers
    # are no-ops under their #pragma once guards.
    framework=[arg for header in FRAMEWORK_HEADERS for arg in ("-include",str(Path(repo)/"ligra"/header))]
    obj=Path(out).with_suffix(".o"); bobj=base.with_suffix(".o"); linkmap=Path(out).with_suffix(".map")
    def command(source,target):
        return ["g++","-std=c++14","-fopenmp","-O3","-DOPENMP","-DBYTERLE","-DHYPER=1","-include","cstdint",*framework,"-I",str(Path(repo)/"ligra"),"-I",str(Path(repo)/"apps/hyper"),"-c","-o",str(target),str(source)]
    jobs=[subprocess.Popen(command(source,target),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) for source,target in ((use,obj),(base,bobj))]
    try:
        codes=[job.wait(timeout=90) for job in jobs]
    finally:
        for job in jobs:
            if job.poll() is None: job.kill(); job.wait()
    if codes[0]: raise RuntimeError("build failed")
    if codes[1]: raise RuntimeError("symbol inspection failed")
    p=subprocess.run(["g++","-fopenmp","-O3","-o",str(out),str(obj),"-Wl,-Map="+str(linkmap)],capture_output=True,text=True,timeout=90)
    if p.returncode: raise RuntimeError("build failed")
    _reject_library_interposition(obj,bobj,linkmap,Path(out))

# C++14 [replacement.functions]: the only library functions a program may
# replace are the global allocation and deallocation functions.
REPLACEABLE=frozenset(
    ["_Znwm","_Znam","_ZnwmRKSt9nothrow_t","_ZnamRKSt9nothrow_t"]
    +[base+suffix for base in ("_ZdlPv","_ZdaPv") for suffix in ("","m","RKSt9nothrow_t")]
    # C++17 aligned forms, in case a toolchain default changes.
    +["_ZnwmSt11align_val_t","_ZnamSt11align_val_t","_ZnwmSt11align_val_tRKSt9nothrow_t","_ZnamSt11align_val_tRKSt9nothrow_t"]
    +[base+suffix for base in ("_ZdlPv","_ZdaPv") for suffix in ("St11align_val_t","mSt11align_val_t","St11align_val_tRKSt9nothrow_t")])

def _readelf(args):
    p=subprocess.run(["readelf","-W",*args],capture_output=True,text=True,timeout=60)
    if p.returncode: raise RuntimeError("symbol inspection failed")
    return p.stdout

def _defined_symbols(text):
    # (name without version, section index) for every defined global, weak
    # or unique symbol of any type in a readelf --syms/--dyn-syms listing.
    rows=[]
    for line in text.splitlines():
        f=line.split()
        if len(f)<8 or not f[0].endswith(":") or f[4] not in ("GLOBAL","WEAK","UNIQUE"): continue
        if f[6] in ("UND","Ndx"): continue
        rows.append((f[7].split("@",1)[0],f[6]))
    return rows

class _ElfObject:
    """Minimal ELF64 little-endian relocatable reader (symbols, groups, RELA)."""

    def __init__(self, path):
        data=Path(path).read_bytes(); self.data=data
        if data[:4]!=b"\x7fELF" or data[4]!=2 or data[5]!=1 or struct.unpack_from("<H",data,16)[0]!=1:
            raise RuntimeError("symbol inspection failed")
        shoff=struct.unpack_from("<Q",data,0x28)[0]
        shnum,shstrndx=struct.unpack_from("<HH",data,0x3C)
        first=struct.unpack_from("<IIQQQQIIQQ",data,shoff)
        if shnum==0: shnum=first[5]
        if shstrndx==0xFFFF: shstrndx=first[6]
        self.sections=[struct.unpack_from("<IIQQQQIIQQ",data,shoff+64*i) for i in range(shnum)]
        names=self.sections[shstrndx]
        self.section_names=[self._string(names,s[0]) for s in self.sections]
        self.symbols=[]; xindex=None
        for s in self.sections:
            if s[1]==18: xindex=s
        for s in self.sections:
            if s[1]!=2: continue
            strtab=self.sections[s[6]]
            for i in range(s[5]//24):
                name,info,other,shndx,value,size=struct.unpack_from("<IBBHQQ",data,s[4]+24*i)
                if shndx==0xFFFF and xindex: shndx=struct.unpack_from("<I",data,xindex[4]+4*i)[0]
                self.symbols.append((self._string(strtab,name),info>>4,info&15,shndx,value,size))
        self.comdat=set(); self.relocations={}
        for s in self.sections:
            words=struct.unpack_from("<%dI"%(s[5]//4),data,s[4]) if s[1]==17 else ()
            if words and words[0]&1: self.comdat.update(words[1:])
            if s[1]==4:
                self.relocations[s[7]]=[struct.unpack_from("<QQq",data,s[4]+24*i) for i in range(s[5]//24)]

    def _string(self, table, offset):
        start=table[4]+offset
        return self.data[start:self.data.index(b"\0",start)].decode("latin-1")

    def _target(self, index):
        name,_,kind,shndx,_,_=self.symbols[index]
        if kind==3: return ("section",self.section_names[shndx] if shndx<len(self.sections) else shndx)
        return ("symbol",name)

    def definition(self, symbol):
        """Bytes and relocations of the section that defines a COMDAT symbol."""
        name,bind,kind,shndx,value,size=symbol
        if shndx==0 or shndx>=0xFF00 or shndx>=len(self.sections) or shndx not in self.comdat: return None
        s=self.sections[shndx]
        body=b"" if s[1]==8 else self.data[s[4]:s[4]+s[5]]
        relocations=sorted((offset,info&0xFFFFFFFF,self._target(info>>32),addend)
                           for offset,info,addend in self.relocations.get(shndx,()))
        return (self.section_names[shndx],s[1],s[2],s[5],body,kind,value,size,tuple(relocations))

    def defined(self):
        # Global, weak and unique definitions of any type.
        return [s for s in self.symbols if s[1] in (1,2,10) and s[3]!=0]

def _reject_library_interposition(obj, baseline_obj, linkmap, binary):
    """No program symbol may also be a symbol exported by a linked library.

    A definition in the program binds the framework's and loader's calls at
    link time, so it would replace library code inside the measured process.
    Exempt are the standard's replaceable allocation functions, and a COMDAT
    definition only when the pristine editable file, compiled with the same
    flags, emits the same symbol with identical section bytes and relocations
    (that is, it comes from the fixed headers, not from candidate code).
    """
    libraries=set()
    for line in linkmap.read_text(errors="replace").splitlines():
        f=line.split()
        if len(f)==2 and f[0]=="LOAD":
            path=Path(f[1])
            try:
                with path.open("rb") as handle: magic=handle.read(18)
            except OSError: continue
            if magic[:4]==b"\x7fELF" and int.from_bytes(magic[16:18],"little")==3:
                libraries.add(path.resolve())
    if not libraries: raise RuntimeError("symbol inspection failed")
    interp=_readelf(["-l",str(binary)])
    for match in re.finditer(r"interpreter: ([^\]]+)\]",interp): libraries.add(Path(match.group(1)).resolve())
    exported=set()
    for library in sorted(libraries):
        exported.update(name for name,_ in _defined_symbols(_readelf(["--dyn-syms",str(library)])))
    candidate=_ElfObject(obj); baseline=_ElfObject(baseline_obj)
    pristine={}
    for symbol in baseline.defined():
        pristine.setdefault(symbol[0],baseline.definition(symbol))
    for symbol in candidate.defined():
        name=symbol[0].split("@",1)[0]
        if name not in exported or name in REPLACEABLE: continue
        reference=pristine.get(symbol[0])
        if reference is not None and candidate.definition(symbol)==reference: continue
        raise RuntimeError("program defines a library symbol")

def _run(binary, graph, expected, role, threads=PARALLEL_THREADS):
    # Correctness and time belong to this same execution; child-reported metrics are ignored.
    import resource
    limit=max(65536,len(expected)*16+8192)
    def output_limit(): resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
    env=os.environ.copy(); env.update(OMP_NUM_THREADS=str(threads),OMP_DYNAMIC="FALSE")
    with tempfile.TemporaryFile() as output:
        started=time.perf_counter_ns()
        p=subprocess.Popen([str(binary),"-s","-rounds","0","-nb","128",str(graph)],stdout=subprocess.DEVNULL,stderr=output,env=env,preexec_fn=output_limit,start_new_session=True)
        try:
            # pidfd readiness avoids Python's timeout-wait polling, which rounds short runs up.
            descriptor=os.pidfd_open(p.pid)
            try:
                if not select.select([descriptor],[],[],40)[0]:
                    raise RuntimeError(f"{role} executable timed out")
                p.wait()
            finally:
                os.close(descriptor)
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGKILL)
                p.wait()
        elapsed=(time.perf_counter_ns()-started)/1e9
        if p.returncode: raise RuntimeError(f"{role} executable exited with status {p.returncode}")
        output.seek(0); raw=output.read(limit+1)
    if len(raw)>limit: raise RuntimeError("core output exceeded its size limit")
    rows=[line for line in raw.decode("ascii").splitlines() if line.startswith("FRONTIER_CORES")]
    if len(rows)!=1: raise RuntimeError("core result was not uniquely observable")
    observed=[int(x) for x in rows[0].split()[1:]]
    if observed!=expected: raise RuntimeError(f"{role} returned incorrect core numbers")
    if not math.isfinite(elapsed) or elapsed<=0: raise RuntimeError("invalid verifier elapsed time")
    return elapsed

def _run_reference(binary, graph, expected, threads=PARALLEL_THREADS):
    # A failed or incorrect reference never sets the bar and is never scored as
    # a candidate failure.
    try:
        return _run(binary,graph,expected,"reference",threads)
    except Exception as error:
        raise ReferenceVerificationError("reference execution failed") from error

def _score_time(t): return 100.0*math.exp(-t/0.25)

def _temp(prefix):
    base=Path(os.environ.get("FRONTIER_EVAL_SCRATCH",Path.cwd()))
    base.mkdir(parents=True,exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix,dir=base))

def _failure_metrics(suite):
    # Global invalidity fails every active workload; no elapsed time was measured.
    return {
        "beat_spec_workload_passes": [
            {"workload": name, "metrics": {metric: False for metric in METRICS}}
            for name, _ in suite
        ],
        "beats_reference": 0,
    }

def evaluate(solution_path: str):
    applied=None; reference=None; work=None
    suite=FINAL if os.environ.get("FRONTIER_SUBMISSION_ROLE")=="final" else DEV
    try:
        applied=apply_submission_patch(solution_path,TASK_DIR,_config())
        if not applied.ok: return 0.0,0.0,applied.message,_failure_metrics(suite)
        reference=apply_submission_patch(TASK_DIR/"reference.patch",TASK_DIR,_config())
        if not reference.ok: raise ReferenceVerificationError("private reference could not be restored")
        work=_temp("frontier_eval_"); cb=work/"candidate"; rb=work/"reference"
        _compile(Path(applied.workdir),cb)
        try:
            _compile(Path(reference.workdir),rb)
        except Exception as error:
            raise ReferenceVerificationError("private reference did not build") from error
        fast=os.environ.get("FRONTIER_TEST_FAST")=="1"
        cases=[]; workload_hashes={}
        try:
            for index,(name,filename) in enumerate(suite):
                if fast:
                    nv=2500; edges=_test_generated(nv,4000,10+4*index,31001+sum(filename.encode()))
                    path=work/(name+".adj"); _write_graph(path,nv,edges)
                else:
                    path=TASK_DIR/"workloads"/filename
                    nv,edges=_read_graph(path)
                expected=_oracle(nv,edges)
                cases.append((name,path,expected))
                workload_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            if len(set(workload_hashes.values()))!=len(cases): raise RuntimeError("duplicate performance workloads")
        except Exception as error:
            raise ReferenceVerificationError("trusted performance workloads are unusable") from error
        for name,nv,edges in GUARDS:
            path=work/("guard-"+name+".adj"); _write_graph(path,nv,edges)
            expected=_oracle(nv,edges)
            _run_reference(rb,path,expected); _run(cb,path,expected,"candidate")
        repeats=3 if fast else REPEATS
        ct={name:[] for name,_,_ in cases}; rt={name:[] for name,_,_ in cases}
        for rep in range(repeats):
            for name,path,expected in cases:
                order=((cb,ct[name],"candidate"),(rb,rt[name],"reference")) if rep%2==0 else ((rb,rt[name],"reference"),(cb,ct[name],"candidate"))
                for binary,bag,role in order:
                    bag.append(_run(binary,path,expected,role) if role=="candidate" else _run_reference(binary,path,expected))
        cm={name:statistics.median(values) for name,values in ct.items()}
        rm={name:statistics.median(values) for name,values in rt.items()}
        # Sequential (one-thread) pass, after and separate from the unchanged
        # eight-thread pass. A candidate failure here fails only this metric;
        # validity and the continuous score keep their eight-thread definition.
        st={name:[] for name,_,_ in cases}; srt={name:[] for name,_,_ in cases}
        sequential_failure={}
        for rep in range(repeats):
            for name,path,expected in cases:
                if name in sequential_failure: continue
                order=(("candidate","reference") if rep%2==0 else ("reference","candidate"))
                for role in order:
                    if role=="reference":
                        srt[name].append(_run_reference(rb,path,expected,SEQUENTIAL_THREADS))
                        continue
                    try:
                        st[name].append(_run(cb,path,expected,"candidate",SEQUENTIAL_THREADS))
                    except ReferenceVerificationError:
                        raise
                    except Exception as error:
                        sequential_failure[name]=type(error).__name__
                        break
        scm={name:(None if name in sequential_failure else statistics.median(values)) for name,values in st.items()}
        srm={name:(statistics.median(values) if values else None) for name,values in srt.items()}
        rows=[{"workload":name,"metrics":{
            "runtime_seconds":bool(cm[name] <= rm[name]*(1.0-MIN_SPEEDUP)),
            "sequential_runtime_seconds":bool(scm[name] is not None and srm[name] is not None
                                              and scm[name] <= srm[name]*(1.0-MIN_SPEEDUP))}} for name in cm]
        med=statistics.median(cm.values()); refmed=statistics.median(rm.values())
        smed=None if sequential_failure else statistics.median(scm.values())
        score=_score_time(med); refscore=_score_time(refmed)
        metrics={"beat_spec_workload_passes":rows,"beats_reference":int(all(value for row in rows for value in row["metrics"].values())),"reference_score":refscore,"margin":score-refscore,"candidate_runtime_seconds_median":med,"candidate_sequential_runtime_seconds_median":smed,"runtime_seconds_per_workload":cm,"reference_runtime_seconds_per_workload":rm,"candidate_runtime_samples":ct,"reference_runtime_samples":rt,"sequential_runtime_seconds_per_workload":scm,"reference_sequential_runtime_seconds_per_workload":srm,"candidate_sequential_runtime_samples":st,"reference_sequential_runtime_samples":srt,"candidate_sequential_failures":sequential_failure,"required_speedup_fraction":MIN_SPEEDUP,"candidate_core_numbers":"validated on every measured execution","reference_core_numbers":"oracle-validated on every measured execution","final_runtime_seconds":med,"workload_sha256":workload_hashes}
        return score,score,"valid exact result",metrics
    except ReferenceVerificationError:
        # Failed verification, not a candidate score.
        raise
    except Exception as error:
        return 0.0,0.0,str(error)[:120] or "evaluation failed",_failure_metrics(suite)
    finally:
        for result in (applied,reference):
            if result and result.workdir: shutil.rmtree(Path(result.workdir).parent,ignore_errors=True)
        if work: shutil.rmtree(work,ignore_errors=True)

if __name__=="__main__":
    result=evaluate(sys.argv[1] if len(sys.argv)>1 else str(TASK_DIR/"reference.patch"))
    print(json.dumps({"score":result[0],"score_unbounded":result[1],"message":result[2],"metrics":result[3]},sort_keys=True))
