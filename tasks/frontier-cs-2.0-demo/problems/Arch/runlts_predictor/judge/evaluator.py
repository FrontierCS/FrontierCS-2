"""Private original-trace evaluator. Authentic entrypoint: lib/cbp.cc -> ./cbp.

No synthetic fallback, no sample scores, no framework CSV reference numbers.
Missing assets or failed reference execution are evaluator errors, not valid zeroes.

The reference is a fixed source run by a deterministic simulator, so its per-trace
counters are read from reference_counts.json instead of being simulated again on
every call. That table is accepted only when it matches the reference restored for
this evaluation (see reference_identity); anything else is an evaluator error.
RUNLTS_REFERENCE_AUDIT=live simulates the reference as before and requires every
counter to equal the table. A submission that restores the reference exactly is
itself a live reference run and is checked against the table the same way.

Trace simulations run in separate processes, up to trace_workers() at a time
(run_ordered). Every counter is an integer produced by the deterministic simulator
from one trace, so the degree of concurrency cannot change a result; results and
failures are taken in manifest order, exactly as the sequential loop produced them.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import yaml
import contextlib
import io
import candidate_failure
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from submission_patch import apply_submission_patch
from source_policy import audit, InvalidSource, CAP_BITS, FIXED_BITS

ROOT=Path(__file__).resolve().parent
CODE_PATH=Path('.')
CONFIG=yaml.safe_load((ROOT/'config.yaml').read_text())
METRICS=('br_mis_pki','cyc_wp_pki','logical_storage_bits')
# The two reference comparisons. logical_storage_bits is one fixed cap applied to
# the whole submission, so it is true on every workload of any graded result.
PREDICTION_METRICS=('br_mis_pki','cyc_wp_pki')
BUILD_TIMEOUT=180
# Watchdog for one simulation, not a metric. The longest reference trace took 660 s
# with four simulations side by side on the authoring host (about 775 s scaled to
# the recorded native run). It was 1,200 s while simulations ran alone; it is doubled
# so that a simulation which finished inside 1,200 s alone still finishes when
# TRACE_WORKERS simulations share caches, memory bandwidth and sibling hardware
# threads.
TRACE_TIMEOUT=2400
TOTAL_TIMEOUT=CONFIG['runtime']['timeout_seconds']
DATA=ROOT/'private_data'/'traces'
# Address-space limit of each build or simulator process: the envelope of one
# simulation (runtime.environment), unchanged by running several side by side.
# A native qualification must confirm this envelope before release. This is not
# a claim of a measured full-run maximum.
MEMORY_BYTES=CONFIG['runtime']['environment']['memory_mb']*1024*1024
# Simulations run side by side: the verifier's CPU allocation (environment.cpus).
# RUNLTS_TRACE_WORKERS may only lower it, and it never exceeds the CPUs available.
TRACE_WORKERS=CONFIG['environment']['cpus']
TRACE_WORKERS_ENV='RUNLTS_TRACE_WORKERS'
CANCEL_POLL_SECONDS=0.2
BUILD_ARGV=['make','-j1','OPT=-O3 -D_GLIBCXX_ASSERTIONS']
TRACE_ARGV=['./cbp']
# Fixed part of the child environment; HOME and TMPDIR are the build tree.
COMMAND_ENV={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','MAKEFLAGS':'-j1'}
PINNED_REFERENCE=ROOT/'reference_counts.json'
# SHA-256 of reference_counts.json. Regenerate both together.
PINNED_REFERENCE_SHA256='ff6c50e82ea4e78ee193e2a90809a90c6af60aaf0d09da3ab430de9be18ab4f1'
PINNED_REFERENCE_SCHEMA='runlts.pinned-reference-counts/v1'
REFERENCE_AUDIT_ENV='RUNLTS_REFERENCE_AUDIT'
PINNED_FIELDS=('instr','misp_br','cyc_wp','full_instr','printed_br','printed_wp')
_RUNTIME_TELEMETRY=ContextVar('runlts_private_runtime_telemetry',default=None)

def runtime_snapshot(telemetry):
    traces=list(telemetry['traces'])
    memory={'scope':'Verifier cgroup high-water mark, including page cache and wrapper processes',
            'status':'unavailable'}
    try:
        memory['bytes']=int(Path('/sys/fs/cgroup/memory.peak').read_text().strip())
        memory['status']='available'
    except (OSError,ValueError):pass
    return {'schema':'runlts.private-runtime-telemetry/v1','submission_role':telemetry['submission_role'],
            'elapsed_seconds':time.monotonic()-telemetry['started'],'traces':traces,
            'longest_trace':max(traces,key=lambda row:row['elapsed_seconds']) if traces else None,
            'verifier_cgroup_memory_peak':memory,
            'child_process_rss_peak':{'bytes':int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)*1024,
                'scope':'Linux RUSAGE_CHILDREN lifetime maximum child-process RSS; excludes cgroup page cache'}}

def available_cpus():
    """CPUs this process may use: its affinity mask, capped by the cgroup quota."""
    try:count=len(os.sched_getaffinity(0))
    except (AttributeError,OSError):count=os.cpu_count() or 1
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':count=min(count,int(quota)//int(period))
    except (OSError,ValueError,ZeroDivisionError):pass
    return max(1,count)

def trace_workers():
    value=os.environ.get(TRACE_WORKERS_ENV,'')
    if value and not re.fullmatch(r'[1-9][0-9]{0,3}',value):raise EvaluationError('unknown trace worker count')
    return max(1,min(TRACE_WORKERS,int(value) if value else TRACE_WORKERS,available_cpus()))

def oom_kills():
    """Processes the kernel has killed in the verifier cgroup for memory, if known."""
    try:
        for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines():
            name,_,count=line.partition(' ')
            if name=='oom_kill':return int(count)
    except (OSError,ValueError):pass
    return None

class Cancelled(BaseException):
    """A simulation stopped because an earlier one in manifest order failed."""

# The cancellation flag of the job the current thread is running, if any.
_JOB=threading.local()

def run_ordered(jobs,workers):
    """Run zero-argument jobs at most `workers` at a time; results in job order.

    The outcome is the sequential loop's: if jobs fail, the exception of the first
    failing job in list order is raised. When a job fails, every later job is
    cancelled (its simulator process group is killed, or it never starts), while
    earlier jobs run to completion because one of them may be the first failure.
    Nothing is left running on return. Telemetry rows of the jobs up to and
    including the first failure are recorded in job order."""
    count=len(jobs);outcomes=[None]*count
    flags=[threading.Event() for _ in jobs];lock=threading.Lock()
    def run(index):
        rows=[];token=_RUNTIME_TELEMETRY.set({'traces':rows});_JOB.cancel=flags[index]
        try:
            if flags[index].is_set():raise Cancelled()
            outcomes[index]=(True,jobs[index](),rows)
        except BaseException as error:
            outcomes[index]=(False,error,rows)
            if not isinstance(error,Cancelled):
                with lock:
                    for later in flags[index+1:]:later.set()
        finally:
            _JOB.cancel=None;_RUNTIME_TELEMETRY.reset(token)
    if workers<=1 or count<=1:
        for index in range(count):run(index)
    else:
        pool=ThreadPoolExecutor(max_workers=min(workers,count),thread_name_prefix='runlts-trace')
        try:
            for index in range(count):pool.submit(run,index)
        finally:
            try:pool.shutdown(wait=True)
            except BaseException:
                # Interrupted while waiting: stop every simulation, then reap them.
                for flag in flags:flag.set()
                pool.shutdown(wait=True)
                raise
    telemetry=_RUNTIME_TELEMETRY.get();results=[]
    for ok,value,rows in outcomes:
        if not ok and isinstance(value,Cancelled):
            raise EvaluationError('trace simulation cancelled without a failure')
        if telemetry is not None:telemetry['traces'].extend(rows)
        if not ok:raise value
        results.append(value)
    return results

def timed_trace(role,tree,row,deadline):
    started=time.monotonic();completed=False
    try:
        result=run_trace(tree,row,deadline);completed=True
        return result
    finally:
        telemetry=_RUNTIME_TELEMETRY.get()
        if telemetry is not None:
            telemetry['traces'].append({'role':role,'workload':row['id'],
                'elapsed_seconds':time.monotonic()-started,'completed':completed})

class EvaluationError(RuntimeError):pass
class CandidateError(RuntimeError):
    """`category`, when set, is a key of candidate_failure.CATEGORIES."""
    def __init__(self,message,category=None):
        super().__init__(message);self.category=category

# Every rejection reason source_policy.py can report, bucketed into the fixed
# solver-visible categories. The reason text itself never leaves the judge in
# development feedback; test_evaluator.py checks this table against the source.
SOURCE_POLICY_CATEGORIES={
    'retained predictor scaffolding changed':'outside_editable_regions',
    'non-register predictor code changed':'outside_editable_regions',
    'unexpected source after retained implementation':'outside_editable_regions',
    'retained harness changed':'outside_editable_regions',
    'source symlinks are not allowed':'outside_editable_regions',
    'unexpected file in predictor tree':'outside_editable_regions',
    'source splicing, raw strings and alternative directives are unsupported':'lexical_form',
    'line comment crosses an editable boundary':'lexical_form',
    'block comment crosses an editable boundary':'lexical_form',
    'literal crosses an editable boundary':'lexical_form',
    'source splicing and alternative preprocessing tokens are unsupported':'lexical_form',
    'escaped source tokens are unsupported':'lexical_form',
    'only local utility includes are permitted in added code':'preprocessor_directive',
    'utility include escapes the source surface':'preprocessor_directive',
    'compiler intrinsics and new templates are outside the source contract':'forbidden_identifier',
    'unsupported side effect or dynamic state in register implementation':'forbidden_identifier',
    'C++ source failed semantic audit':'semantic_audit',
    'new templates and lambdas are unsupported':'forbidden_construct',
    'new pointer/reference-returning functions are unsupported':'forbidden_construct',
    'compiler builtin calls are unsupported':'forbidden_construct',
    'unresolved or indirect call in register code':'forbidden_construct',
    'unsupported external call in register code':'forbidden_construct',
    'unsafe cast or dynamic allocation':'forbidden_construct',
    'raw pointers are not permitted in new state/code':'forbidden_construct',
    'new mutable reference aliases are unsupported':'forbidden_construct',
    'pointer/reference casts are unsupported':'forbidden_construct',
    'use checked std::array indexing in new code':'forbidden_construct',
    'address manipulation is outside the register contribution':'forbidden_construct',
    'register code references protected predictor state':'protected_state_access',
    'retained prediction inputs/state are read-only':'protected_state_access',
    'unresolved array extent':'state_declaration',
    'state inheritance is unsupported':'state_declaration',
    'use explicit fixed state instead of unions':'state_declaration',
    'opaque state declaration':'state_declaration',
    'persistent state needs fixed integers, bit-fields, or fixed arrays':'state_declaration',
    'full logical predictor storage exceeds 192 KiB':'storage_cap',
}

def in_stage(error,category):
    """Name the failing candidate stage unless the raise site already did."""
    if error.category is None:error.category=category
    return error

def candidate_category(error):
    """The fixed category of a candidate-side failure, or None if it has none."""
    if isinstance(error,InvalidSource):category=SOURCE_POLICY_CATEGORIES.get(str(error))
    else:category=getattr(error,'category',None)
    return category if category in candidate_failure.CATEGORIES else None

class CommandFailure(CandidateError):
    """Keep bounded runtime evidence private; candidate feedback stays generic."""
    def __init__(self, kind, *, elapsed_seconds, timeout_seconds, returncode, output_tail):
        super().__init__('build or trace timeout' if kind=='timeout' else 'build or simulator execution failed')
        self.diagnostic={'kind':kind,'elapsed_seconds':round(elapsed_seconds,3),
                         'timeout_seconds':timeout_seconds,'returncode':returncode,
                         'output_tail':output_tail[-4096:]}

def private_exception_chain(error):
    """Bounded causal metadata; no messages, paths, source lines or locals."""
    chain=[];seen=set()
    while error is not None and id(error) not in seen and len(chain)<8:
        seen.add(id(error))
        item={'error_type':type(error).__name__[:128]}
        if isinstance(error,OSError) and type(error.errno) is int:
            item['errno']=error.errno
        frames=[];tb=error.__traceback__
        while tb is not None:
            frames.append({'function':tb.tb_frame.f_code.co_name[:128],'line':tb.tb_lineno})
            frames=frames[-8:];tb=tb.tb_next
        if frames:item['frames']=frames
        chain.append(item)
        error=error.__cause__ or (error.__context__ if not error.__suppress_context__ else None)
    return {'exceptions':chain,'truncated':error is not None}

def reference_failure(phase,error):
    failure=EvaluationError('private reference '+phase+' failed')
    failure.private_execution_evidence={'execution_valid':False,'reference_phase':phase,
        'error_type':type(error).__name__,'exception_chain':private_exception_chain(error)}
    if isinstance(error,CommandFailure):
        failure.private_execution_evidence['command_failure']=error.diagnostic
    elif isinstance(error,CandidateError):
        # Parser messages are fixed task strings; never retain arbitrary OSError paths.
        failure.private_execution_evidence['parser_failure']=str(error)[:256]
    return failure

@dataclass(frozen=True)
class Measurement:
    instr:int
    misp_br:int
    cyc_wp:int
    full_instr:int
    printed_br:Decimal
    printed_wp:Decimal

def parse_output(text:str, expected:dict):
    """Unique original simulator sections; never select a predictor lookalike row."""
    lines=text.splitlines()
    def section(phrase):
        indices=[i for i,s in enumerate(lines) if 'DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS' in s and phrase in s]
        if len(indices)!=1:raise CandidateError('missing or duplicate simulator accounting','output_format')
        i=indices[0]
        if i+2>=len(lines):raise CandidateError('truncated simulator accounting','output_format')
        if lines[i+1].split()!=['Instr','Cycles','IPC','NumBr','MispBr','BrPerCyc','MispBrPerCyc','MR','MPKI','CycWP','CycWPAvg','CycWPPKI']:
            raise CandidateError('invalid simulator accounting header','output_format')
        t=lines[i+2].split()
        if len(t)!=12:raise CandidateError('invalid simulator accounting row','output_format')
        if any(not re.fullmatch(r'\d+',t[k]) for k in (0,1,3,4,9)):
            raise CandidateError('noninteger simulator counters','output_format')
        if any(not re.fullmatch(r'\d+\.\d{4}',t[k]) for k in (8,11)):
            raise CandidateError('invalid printed simulator rates','output_format')
        n,b,w=int(t[0]),int(t[4]),int(t[9])
        if n<=0 or b>int(t[3]) or int(t[3])>n:raise CandidateError('invalid simulator counter range','output_format')
        br,wp=Decimal(t[8]),Decimal(t[11])
        # Check display consistency only. Strict comparisons use integer counters.
        if abs(br-Decimal(1000*b)/n)>Decimal('0.0000500001') or abs(wp-Decimal(1000*w)/n)>Decimal('0.0000500001'):
            raise CandidateError('inconsistent simulator rates','output_format')
        return n,b,w,br,wp
    n,b,w,br,wp=section('(50 Perc instructions)')
    full=section('(Full Simulation i.e. Counts Not Reset When Warmup Ends)')[0]
    if full!=expected['full_instr'] or n!=expected['measured_instr'] or not (full//2<n<=full):
        raise CandidateError('incomplete or substituted trace execution','instruction_count_mismatch')
    return Measurement(n,b,w,full,br,wp)

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def tree_sha256(tree):
    """One digest of every path and content below a source tree, before any build."""
    h=hashlib.sha256();tree=Path(tree)
    for p in sorted(tree.rglob('*'),key=lambda q:q.relative_to(tree).as_posix()):
        if p.is_dir() and not p.is_symlink():continue
        kind=sha256_file(p) if p.is_file() and not p.is_symlink() else 'not-a-regular-file'
        h.update((p.relative_to(tree).as_posix()+'\0'+kind+'\n').encode())
    return h.hexdigest()

def reference_identity(reference_tree):
    """Everything that determines the reference's counters, as this evaluation has it.

    reference_tree_sha256 covers the restored reference source together with the
    retained simulator, interfaces and Makefiles; the document hashes are the
    identities the task already pins. Trace bytes are bound per row, and manifest()
    has already checked them against workload_manifest.json."""
    return {'reference_patch_sha256':sha256_file(ROOT/'reference.patch'),
            'reference_core_sha256':sha256_file(Path(reference_tree)/'my_cond_branch_predictor.cc'),
            'reference_tree_sha256':tree_sha256(reference_tree),
            'harness_integrity_sha256':sha256_file(ROOT/'integrity.json'),
            'source_contract_sha256':sha256_file(ROOT/'source_contract.json'),
            'workload_manifest_sha256':sha256_file(ROOT/'workload_manifest.json'),
            'invocation':{'build_argv':list(BUILD_ARGV),'trace_argv':[*TRACE_ARGV,'<trace.gz>'],
                          'environment':dict(COMMAND_ENV)}}

def pinned_failure(reason,**evidence):
    """The table, not the candidate, is at fault: always an evaluator error."""
    failure=EvaluationError('pinned reference '+reason)
    failure.private_execution_evidence={'execution_valid':False,'reference_phase':'pinned_table','reason':reason,**evidence}
    return failure

def pinned_measurement(row):
    """A table row as a Measurement, under the same sanity rules as simulator output."""
    if any(type(row.get(k)) is not int for k in ('instr','misp_br','cyc_wp','full_instr')):raise ValueError('counter')
    if any(not isinstance(row.get(k),str) or not re.fullmatch(r'\d+\.\d{4}',row[k]) for k in ('printed_br_mis_pki','printed_cyc_wp_pki')):raise ValueError('rate')
    n,b,w,full=row['instr'],row['misp_br'],row['cyc_wp'],row['full_instr']
    br,wp=Decimal(row['printed_br_mis_pki']),Decimal(row['printed_cyc_wp_pki'])
    if n<=0 or b<0 or w<0 or b>n or not (full//2<n<=full):raise ValueError('range')
    if abs(br-Decimal(1000*b)/n)>Decimal('0.0000500001') or abs(wp-Decimal(1000*w)/n)>Decimal('0.0000500001'):raise ValueError('rate consistency')
    return Measurement(n,b,w,full,br,wp)

def pinned_reference(suite,reference_tree):
    """Reference measurements for `suite` from the pinned table, or EvaluationError."""
    try:raw=PINNED_REFERENCE.read_bytes()
    except OSError:raise pinned_failure('table missing') from None
    if hashlib.sha256(raw).hexdigest()!=PINNED_REFERENCE_SHA256:raise pinned_failure('table checksum mismatch')
    try:doc=json.loads(raw)
    except ValueError:raise pinned_failure('table unreadable') from None
    rows=doc.get('rows') if isinstance(doc,dict) else None
    if not isinstance(doc,dict) or doc.get('schema')!=PINNED_REFERENCE_SCHEMA or not isinstance(rows,list) \
            or any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in rows):
        raise pinned_failure('table schema mismatch')
    current=reference_identity(reference_tree);recorded=doc.get('identity')
    if recorded!=current:
        names=sorted(k for k in current if not isinstance(recorded,dict) or recorded.get(k)!=current[k])
        raise pinned_failure('identity mismatch',mismatched=names or ['unexpected identity fields'])
    by_id={r['id']:r for r in rows}
    if len(rows)!=105 or len(by_id)!=105:raise pinned_failure('table coverage invalid')
    reference={}
    for row in suite:
        pinned=by_id.get(row['id'])
        if pinned is None:raise pinned_failure('row missing',workload=row['id'])
        try:measured=pinned_measurement(pinned)
        except ValueError:raise pinned_failure('row invalid',workload=row['id']) from None
        # The row must describe this exact trace and its complete execution.
        if pinned.get('path')!=row['path'] or pinned.get('trace_sha256')!=row['sha256'] \
                or measured.full_instr!=row['full_instr'] or measured.instr!=row['measured_instr']:
            raise pinned_failure('row does not match the workload manifest',workload=row['id'])
        reference[row['id']]=measured
    return reference

def require_pinned_equal(row,live,pinned,source):
    """A live reference execution must reproduce the table exactly."""
    fields={k:{'pinned':str(getattr(pinned,k)),'live':str(getattr(live,k))} for k in PINNED_FIELDS if getattr(pinned,k)!=getattr(live,k)}
    if fields:
        failure=EvaluationError('pinned reference counts differ from live reference execution')
        failure.private_execution_evidence={'execution_valid':False,'reference_phase':'pinned_audit',
            'audit_source':source,'workload':row['id'],'fields':fields}
        raise failure

def reference_audit_requested():
    value=os.environ.get(REFERENCE_AUDIT_ENV,'')
    if value not in ('','live'):raise EvaluationError('unknown reference audit mode')
    return value=='live'

def manifest(final:bool):
    doc=json.loads((ROOT/'workload_manifest.json').read_text()); rows=doc['workloads']
    if len(rows)!=105 or len({r['id'] for r in rows})!=105 or len({r['path'] for r in rows})!=105:
        raise EvaluationError('full105 workload manifest is invalid')
    if sum(r['development'] is True for r in rows)!=24:raise EvaluationError('development partition is invalid')
    identity=hashlib.sha256(('\n'.join(sorted(r['path'] for r in rows))+'\n').encode()).hexdigest()
    if identity!='71505618b468179e5d03664d7d414e5da4d54347a3fe4fecd1b9bac75bab1386':raise EvaluationError('original full105 identities changed')
    expected={'compress':8,'fp':14,'infra':16,'int':37,'media':4,'web':26}
    for category,count in expected.items():
        group=[r for r in rows if r['category']==category]
        if len(group)!=count:raise EvaluationError('workload category coverage is invalid')
        ordered=sorted(group,key=lambda r:(hashlib.sha256(r['path'].encode()).hexdigest(),r['path']))
        wanted={r['id'] for r in ordered[:math.ceil(count/5)]}
        if wanted!={r['id'] for r in group if r['development']}:raise EvaluationError('frozen partition changed')
    for r in rows:
        p=Path(r['path'])
        if p.is_absolute() or '..' in p.parts or len(p.parts)!=2 or p.suffix!='.gz':raise EvaluationError('invalid workload path')
        if r['id']!=r['path'][:-3] or p.parts[0]!=r['category']:raise EvaluationError('workload identity mismatch')
    # Provisioning is intentionally all-or-nothing, including before train.
    if doc.get('status')!='checksum_pinned':raise EvaluationError('full105 trace assets and checksums are not provisioned')
    for r in rows:
        if not isinstance(r.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',r['sha256']):raise EvaluationError('trace checksum pin missing')
        if any(type(r.get(k)) is not int or r[k]<=0 for k in ('compressed_bytes','full_instr','measured_instr')):raise EvaluationError('trace completeness pin missing')
        p=DATA/r['path']
        if p.is_symlink() or not p.is_file():raise EvaluationError('required trace asset missing')
        if p.stat().st_size!=r['compressed_bytes']:raise EvaluationError('trace size mismatch')
        h=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(1024*1024):h.update(chunk)
        if h.hexdigest()!=r['sha256']:raise EvaluationError('trace checksum mismatch')
    return rows if final else [r for r in rows if r['development']]

def arithmetic_means(rows):
    if not rows:raise EvaluationError('empty suite')
    n=Decimal(len(rows))
    return sum((r.printed_br for r in rows),Decimal(0))/n, sum((r.printed_wp for r in rows),Decimal(0))/n

def score_of(rows):
    # Absolute, reference-independent, smooth and strictly decreasing in both means.
    # Units 1 BrMisPKI and 100 CycWpPKI are challenge constants, not paper claims.
    b,w=arithmetic_means(rows)
    return 100.0/math.sqrt((1.0+float(b))*(1.0+float(w)/100.0))

def grade(workloads,candidate,reference,final,*,logical_storage_bits):
    # Trusted static allocation, never candidate-printed size or process RSS.
    if type(logical_storage_bits) is not int or logical_storage_bits < FIXED_BITS:
        raise EvaluationError('missing or invalid logical storage audit')
    if logical_storage_bits > CAP_BITS:
        raise CandidateError('full logical predictor storage exceeds 192 KiB','storage_cap')
    ids=[r['id'] for r in workloads]
    if len(ids)!=len(set(ids)) or not ids:raise EvaluationError('invalid measurement coverage')
    if set(candidate)!=set(ids) or set(reference)!=set(ids):raise EvaluationError('incomplete paired measurement coverage')
    if final and len(ids)!=105:raise EvaluationError('final evaluation requires all 105 workloads')
    decisions=[]; paired=[]
    for wid in ids:
        c,r=candidate[wid],reference[wid]
        if not (c.instr==r.instr>0) or c.full_instr!=r.full_instr:raise CandidateError('paired instruction counts differ','instruction_count_mismatch')
        checks={'br_mis_pki':c.misp_br*r.instr<r.misp_br*c.instr,'cyc_wp_pki':c.cyc_wp*r.instr<r.cyc_wp*c.instr,
                'logical_storage_bits':logical_storage_bits<=CAP_BITS}
        decisions.append({'workload':wid,'metrics':checks})
        paired.append({'workload':wid,'instr':c.instr,'misp_br':c.misp_br,'cyc_wp':c.cyc_wp,'br_mis_pki':float(Fraction(1000*c.misp_br,c.instr)),'cyc_wp_pki':float(Fraction(1000*c.cyc_wp,c.instr)), 'reference_instr':r.instr,'reference_misp_br':r.misp_br,'reference_cyc_wp':r.cyc_wp})
    crows=[candidate[i] for i in ids]; rrows=[reference[i] for i in ids]
    score,rs=score_of(crows),score_of(rrows)
    b,w=arithmetic_means(crows); rb,rw=arithmetic_means(rrows)
    metrics={'beats_reference':int(all(v for row in decisions for v in row['metrics'].values())),
             'beat_spec_workload_passes':decisions,'reference_score':rs,'margin':score-rs,
             'paired_measurements':paired,'evaluation_valid':True,'suite':'final' if final else 'development',
             'workload_count':len(ids), 'logical_storage_bits':logical_storage_bits,
             'logical_storage_accounting':{'fixed_non_sr_bits':FIXED_BITS,
                 'sr_bits':logical_storage_bits-FIXED_BITS,'total_bits':logical_storage_bits,'cap_bits':CAP_BITS},
             'final_beats_reference': int(final and all(v for row in decisions for v in row['metrics'].values()))}
    if not final:
        # Development only: workloads clearing both reference comparisons together.
        cleared=sum(all(row['metrics'][name] for name in PREDICTION_METRICS) for row in decisions)
        metrics['all_metrics_progress']={'passed':cleared,'failed':len(ids)-cleared,'total':len(ids),
            'pass_percent':round(100.0*cleared/len(ids),2),'fail_percent':round(100.0*(len(ids)-cleared)/len(ids),2)}
        # Development only: the candidate's own means per application category, from
        # exact per-trace rates. No reference value or comparison outcome.
        groups={}
        for row in workloads:groups.setdefault(row.get('category','uncategorized'),[]).append(candidate[row['id']])
        metrics['category_means']={category:{'traces':len(rows),
            'br_mis_pki_mean':float(sum(Fraction(1000*m.misp_br,m.instr) for m in rows)/len(rows)),
            'cyc_wp_pki_mean':float(sum(Fraction(1000*m.cyc_wp,m.instr) for m in rows)/len(rows))}
            for category,rows in sorted(groups.items())}
    prefix='final' if final else 'train'
    metrics[prefix+'_br_mis_pki_mean']=float(b);metrics[prefix+'_cyc_wp_pki_mean']=float(w)
    metrics['reference_'+prefix+'_br_mis_pki_mean']=float(rb);metrics['reference_'+prefix+'_cyc_wp_pki_mean']=float(rw)
    return score,score,'evaluation complete',metrics

def limited_argv(argv):
    # An exec-side launcher avoids running Python callbacks after fork from a
    # trace worker thread. Apply the same hard and soft limits before exec.
    return ['/usr/bin/prlimit',f'--as={MEMORY_BYTES}:{MEMORY_BYTES}',
            '--fsize=16777216:16777216','--core=0:0','--',*argv]

def run_command(argv,cwd,timeout,deadline):
    remaining=min(timeout,deadline-time.monotonic())
    if remaining<=0:raise CandidateError('evaluation timeout')
    # Do not forward model/provider environment to compiler or candidate binary.
    env={**COMMAND_ENV,'HOME':str(cwd),'TMPDIR':str(cwd)}
    cancel=getattr(_JOB,'cancel',None)
    # An unnamed file: nothing is left in the tree when a simulation is killed, and
    # simulations sharing a build tree as working directory never share an output.
    with tempfile.TemporaryFile(dir=cwd) as out:
        proc=subprocess.Popen(limited_argv(argv),cwd=cwd,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.monotonic();timed_out=False;cancelled=False
        try:
            while True:
                left=started+remaining-time.monotonic()
                cancelled=cancel is not None and cancel.is_set()
                if cancelled or left<=0:
                    timed_out=not cancelled;break
                try:proc.wait(timeout=left if cancel is None else min(left,CANCEL_POLL_SECONDS));break
                except subprocess.TimeoutExpired:pass
        finally:
            # Also on an interrupt: never leave the process group running.
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
        if cancelled:raise Cancelled()
        if timed_out or proc.returncode:
            out.seek(0,os.SEEK_END);size=out.tell();out.seek(max(0,size-4096))
            tail=out.read(4096).decode('utf-8',errors='replace')
            raise CommandFailure('timeout' if timed_out else 'process_exit',
                elapsed_seconds=time.monotonic()-started,timeout_seconds=remaining,
                returncode=proc.returncode,output_tail=tail)
        out.seek(0);return out.read(16*1024*1024).decode('utf-8',errors='replace')

def build(tree,deadline):
    # The retained Makefiles compile lib/cbp.cc and link cbp. Bounds assertions
    # add no prediction inputs, simulator options, or logical state.
    run_command(list(BUILD_ARGV),tree,BUILD_TIMEOUT,deadline)

def run_trace(tree,row,deadline):
    # Fresh process through the original binary, Makefiles and default simulator.
    if not (tree/'lib/cbp.cc').is_file():raise CandidateError('retained simulator entrypoint missing','outside_editable_regions')
    text=run_command([*TRACE_ARGV,str(DATA/row['path'])],tree,TRACE_TIMEOUT,deadline)
    return parse_output(text,row)

def _evaluate(solution_path):
    deadline=time.monotonic()+TOTAL_TIMEOUT
    # All temporary trees use task-local disk, never the author's RAM-backed /tmp.
    scratch=ROOT/'scratch';scratch.mkdir(exist_ok=True)
    tempfile.tempdir=str(scratch)
    patched=None;ref=None
    try:
        patched=apply_submission_patch(solution_path,ROOT,CONFIG)
        if not patched.ok:raise CandidateError('invalid patch or patch outside permitted surface','patch_policy')
        try:
            storage=json.loads(run_command([sys.executable,str(ROOT/'source_policy.py'),str(patched.workdir)],patched.workdir,60,deadline))
        except CandidateError:raise CandidateError('source audit failed or exceeded its resource limit','source_audit_incomplete') from None
        if storage.get('infrastructure_error'):raise EvaluationError('source audit infrastructure unavailable')
        if not storage.get('valid'):
            reason=storage.get('reason','invalid register implementation')
            raise CandidateError(reason,SOURCE_POLICY_CATEGORIES.get(reason) if isinstance(reason,str) else None)
        final=os.environ.get('FRONTIER_SUBMISSION_ROLE')=='final'
        audit=reference_audit_requested()
        workers=trace_workers()
        suite=manifest(final)
        # The reference is always restored: the table is only valid for this source.
        ref=apply_submission_patch(ROOT/'reference.patch',ROOT,CONFIG)
        if not ref.ok:raise EvaluationError('private reference restoration failed')
        pinned=pinned_reference(suite,ref.workdir)
        # Source trees are compared before either build writes into them.
        candidate_is_reference=tree_sha256(patched.workdir)==tree_sha256(ref.workdir)
        if audit:
            try:build(ref.workdir,deadline)
            except Exception as error:raise reference_failure('build',error) from error
        try:build(patched.workdir,deadline)
        except CandidateError as error:
            if candidate_is_reference:raise reference_failure('build',error) from error
            raise in_stage(error,'build_failed')
        # One job per simulation, in the order the sequential loop ran them: for
        # each workload the live reference (audit only), then the candidate. Each
        # job classifies its own failure, so run_ordered raises what that loop did.
        def reference_job(row):
            try:live=timed_trace('reference',ref.workdir,row,deadline)
            except Exception as error:raise reference_failure('execution',error) from error
            require_pinned_equal(row,live,pinned[row['id']],'live_reference')
        def candidate_job(row):
            try:measured=timed_trace('candidate',patched.workdir,row,deadline)
            except CandidateError as error:
                if candidate_is_reference:raise reference_failure('execution',error) from error
                # A simulator killed because the verifier ran out of memory says
                # nothing about the candidate.
                killed=getattr(error,'diagnostic',{}).get('returncode')==-signal.SIGKILL and error.diagnostic.get('kind')=='process_exit'
                if killed and oom_before is not None and (oom_kills() or 0)>oom_before:
                    failure=EvaluationError('verifier memory exhausted during trace simulation')
                    failure.private_execution_evidence={'execution_valid':False,'workload':row['id'],'command_failure':error.diagnostic}
                    raise failure from error
                raise in_stage(error,'simulator_run_failed')
            if candidate_is_reference:require_pinned_equal(row,measured,pinned[row['id']],'reference_submission')
            return measured
        oom_before=oom_kills();jobs=[];owners=[]
        for row in suite:
            if audit:jobs.append(lambda row=row:reference_job(row));owners.append(None)
            jobs.append(lambda row=row:candidate_job(row));owners.append(row['id'])
        measured=run_ordered(jobs,workers)
        candidate={wid:value for wid,value in zip(owners,measured) if wid is not None}
        reference={row['id']:pinned[row['id']] for row in suite}
        result=grade(suite,candidate,reference,final,logical_storage_bits=storage.get('logical_bits'))
        result[3]['logical_storage_audit_method']=storage.get('method','unspecified')
        checked=len(suite) if audit or candidate_is_reference else 0
        result[3]['pinned_reference']={'table_sha256':PINNED_REFERENCE_SHA256,
            'live_check':'live_reference' if audit else 'reference_submission' if candidate_is_reference else 'none',
            'workloads_equal_to_live_execution':checked}
        result[3]['trace_simulation']={'workers':workers,'configured_workers':TRACE_WORKERS,'simulations':len(jobs)}
        return result
    finally:
        for result in (patched,ref):
            if result and result.cleanup_root:shutil.rmtree(result.cleanup_root,ignore_errors=True)

def evaluate(solution_path:str)->tuple[float,float,str,dict]:
    telemetry={'started':time.monotonic(),'submission_role':os.environ.get('FRONTIER_SUBMISSION_ROLE','train'),
               'traces':[]}
    token=_RUNTIME_TELEMETRY.set(telemetry)
    try:
        result=_evaluate_with_failure_mapping(solution_path)
        result[3]['private_runtime_telemetry']=runtime_snapshot(telemetry)
        return result
    except Exception as error:
        evidence=dict(getattr(error,'private_execution_evidence',{}))
        evidence['private_runtime_telemetry']=runtime_snapshot(telemetry)
        error.private_execution_evidence=evidence
        raise
    finally:_RUNTIME_TELEMETRY.reset(token)

def _evaluate_with_failure_mapping(solution_path):
    try:return _evaluate(solution_path)
    except (CandidateError,InvalidSource) as e:
        # Final keeps the private scored zero. A development rejection returns only
        # a fixed category and sentence (candidate_failure.py), which the verifier
        # re-derives from the same table before any solver sees it.
        category=candidate_category(e)
        if os.environ.get('FRONTIER_SUBMISSION_ROLE')=='final' or category is None:return 0.0,0.0,str(e),{}
        visible=candidate_failure.diagnostic(category)
        return 0.0,0.0,visible['message'],{'candidate_diagnostic':visible}
    except EvaluationError:
        # Native adapter writes status=failed and bounded private exception evidence.
        # Missing assets/reference execution must never become a passed score-zero.
        raise
    except Exception as error:
        raise EvaluationError('internal evaluator infrastructure failure') from error

def prepare():
    try:
        rows=manifest(True)
        return {'ready':True,'final_workloads':len(rows)}
    except EvaluationError:return {'ready':False,'reason':'full105 data provisioning incomplete'}

if __name__=='__main__':
    result=evaluate(sys.argv[1] if len(sys.argv)==2 else '')
    print(json.dumps({'score':result[0],'score_unbounded':result[1],'message':result[2],'metrics':result[3]},allow_nan=False))
    sys.exit(2 if result[3].get('evaluator_error') else 0)
