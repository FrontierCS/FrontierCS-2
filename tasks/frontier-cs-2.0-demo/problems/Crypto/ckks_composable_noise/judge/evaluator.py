"""Scaled CKKS composition measurement: accuracy bands and estimator time.

Every reported number is printed by a trusted derived experiment main. No file
of package estimator is overlaid, so submitted code cannot emit an observation.
An uncalibrated release fails closed before launching a submitted program.

The shared verifier applies the patch and enforces the editable surface
(task.toml [submission]: estimator/ only, .go files, no _test.go, the fixed set
denied); check_tree() runs the judge-owned Go AST policy on the judged tree
against the pristine starter; evaluate(tree) builds and measures that tree. Every submitted
program (the guards and each measured main) is built by the judge's toolchain and runs in the
shared isolate launcher (run_measured).

Scoring (formal_scoring.py). Per invocation, the two discrepancies are |predicted - actual| bits
of AVGLog2Prec.L2 and STDLog2Prec.L2; a_w, s_w are their means over the invocations of
configuration w (3 pairs in final over 29 configurations, 1 pair in train over the 6 base
ones). raw_score = score_unbounded = 100 / (1 + E), E = sum_w (a_w + s_w) / (2 W) in bits.
Metrics per workload: avg_/std_log2_prec_l2_abs_discrepancy_bits (lower better; passes when
candidate - reference <= the frozen band of formal_calibration.json) and estimator_seconds
(lower better; passes when the candidate's trusted clock is strictly lower in every pair).
beats_reference = every metric passes on every configuration. A rejected or failed run is a
scored zero (failed_result).
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
from go_entrypoint import registered_go_entrypoint
import isolate  # the shared launcher, beside edit_scope in the Verifier base
# config.yaml is generated from task.toml (JSON); environment and go_memory_limit come from
# its [judge.config], timeout_seconds from quota.eval_timeout_seconds.
CONFIG = json.loads((TASK_DIR / 'config.yaml').read_text())
RELEASE = json.loads((TASK_DIR / 'judge/release.json').read_text())
from workload_configurations import validate_manifest, prepare_overlay, verify_parameters
# The starter repo/, copied by the Verifier Dockerfile; the submission is overlaid on a copy.
PRISTINE = TASK_DIR / 'pristine'
WORKLOADS = validate_manifest(json.loads((TASK_DIR/'judge/workload_manifest.json').read_text()), RELEASE)
ENTRYPOINT = 'estimator/experiments/mul_relin_rescale/mul_relin_rescale.go'
METRICS = ('avg_log2_prec_l2_abs_discrepancy_bits', 'std_log2_prec_l2_abs_discrepancy_bits')
TIME_METRIC = 'estimator_seconds'
# The train metrics task.toml [feedback] shows the Agent (the two discrepancy means are built
# from METRICS in summarize/failed_result).
FEEDBACK = ('candidate_avg_log2_prec_l2_abs_discrepancy_mean_bits',
            'candidate_std_log2_prec_l2_abs_discrepancy_mean_bits', 'candidate_estimator_seconds_mean')
PAIRS = RELEASE['interleaved_pairs']
DEV_SUITE = tuple(RELEASE['development'])
# Development feedback measures a cheaper subset with one pair; it gates nothing.
DEV_FEEDBACK = tuple(RELEASE['development_feedback']['workloads'])
DEV_PAIRS = RELEASE['development_feedback']['interleaved_pairs']
if not DEV_FEEDBACK or len(set(DEV_FEEDBACK)) != len(DEV_FEEDBACK) or not set(DEV_FEEDBACK) <= set(DEV_SUITE) or type(DEV_PAIRS) is not int or DEV_PAIRS < 1:
    raise ValueError('invalid development feedback manifest')
FINAL_SUITE = tuple(RELEASE['final'])
SCORE_BITS = 1.0

class EvaluationError(Exception):
    def __init__(self, message, receipt=None):
        super().__init__(message)
        self.receipt = receipt or {}

def final_role():
    return os.environ.get('FRONTIER_SUBMISSION_ROLE') == 'final'

def suite():
    return FINAL_SUITE if final_role() else DEV_FEEDBACK

def pairs():
    return PAIRS if final_role() else DEV_PAIRS

def scratch_root():
    # Native adapter supplies TMPDIR under its writable workspace. No /tmp fallback.
    p = Path(os.environ.get('TMPDIR', str(TASK_DIR.parent / 'scratch/runtime'))).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def go_env():
    """The judge's Go toolchain settings: offline, read-only modules, caches under the scratch root."""
    env = {k: os.environ[k] for k in ('PATH', 'GOMODCACHE') if k in os.environ}
    p = scratch_root()
    for k, subdir in [('GOPATH','go-path'), ('GOCACHE','go-cache'), ('GOTMPDIR','go-tmp')]:
        env[k] = str(p / subdir)
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    env.update(TMPDIR=str(p), GOTOOLCHAIN='local', GOENV='off', GOWORK='off', GOTELEMETRY='off',
               GOFLAGS='-mod=readonly', CGO_ENABLED='0', GOPROXY='off', GOSUMDB='off')
    return env

def program_env():
    """The Go runtime settings of a measured program."""
    cpus = min(CONFIG['environment']['cpus'], len(os.sched_getaffinity(0))) if hasattr(os, 'sched_getaffinity') else 1
    # One scheduler thread: the estimator clock sees its own collection work.
    cpus = 1 if RELEASE.get('measured_gomaxprocs') == 1 else cpus
    return {'GOMAXPROCS': str(cpus), 'GOMEMLIMIT': CONFIG['runtime']['go_memory_limit']}

def run(args, cwd, deadline):
    return run_measured(args, cwd, deadline)[0]

def run_measured(args, cwd, deadline):
    """`go run -p=1 [flags] <main>` in `cwd`, split in two: the trusted toolchain builds the
    program (CGO_ENABLED=0 and no generators, so a build runs no submitted code), and the
    program runs under the shared isolate launcher under /usr/bin/time: it reads only the
    system and its binary, writes only its own private directories, gets a clean
    environment with the Go runtime settings, no network, and is bounded by memory_mb, 16 MiB
    files, 1 MiB per output stream and the evaluation deadline."""
    command = [str(x) for x in args]
    if command[:2] != ['go', 'run']:
        raise ValueError('run_measured runs `go run` commands')
    receipt = {'command':command, 'cwd':str(cwd), 'launch_attempts':1, 'main_completed':0,
               'exit_code':None, 'peak_rss_kib':None,
               'rss_scope':'GNU time maximum RSS of the program in its isolate instance, compilation excluded; not summed concurrent RSS'}
    if deadline - time.monotonic() <= 0:
        raise EvaluationError('evaluation timeout', {'status':'not_started_deadline'})
    with tempfile.TemporaryDirectory(prefix='ckks-process-', dir=scratch_root()) as d:
        (Path(d)/'program').mkdir()
        binary = Path(d)/'program'/'main'
        started = time.monotonic()
        try:
            built = subprocess.run(['go', 'build', *command[2:-1], '-o', str(binary), command[-1]],
                                   cwd=cwd, env=go_env(), capture_output=True,
                                   timeout=max(0.001, deadline-time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            receipt.update(status='timeout', elapsed_seconds=time.monotonic()-started)
            raise EvaluationError('evaluation timeout', receipt) from exc
        if built.returncode:
            receipt.update(status='failed', exit_code=built.returncode, elapsed_seconds=time.monotonic()-started,
                           stderr=built.stderr[-4000:].decode('utf-8','replace'))
            raise EvaluationError('build, source policy, or execution failed', receipt)
        remain = deadline - time.monotonic()
        if remain <= 0:
            receipt.update(status='timeout', elapsed_seconds=time.monotonic()-started)
            raise EvaluationError('evaluation timeout', receipt)
        # The instance works on a copy of the program's directory (the binary alone).
        result = isolate.run(['/usr/bin/time', '-f', '%e %M', '-o', '{out}/usage', '{work}/main'],
                                  timeout_seconds=remain, tree=str(binary.parent),
                                  env=program_env(), outputs=['usage'],
                                  memory_bytes=CONFIG['environment']['memory_mb']*1024**2,
                                  file_bytes=16*1024**2, output_bytes=2**20)
        receipt.update(elapsed_seconds=time.monotonic()-started, main_completed=int(result.ok),
                       exit_code=result.returncode,
                       status='timeout' if result.timed_out else ('ok' if result.ok else 'failed'))
        match = re.search(r'^([\d.]+) (\d+)$', (result.outputs['usage'] or b'').decode('utf-8','replace'), re.M)
        if match:
            receipt.update(time_elapsed_seconds=float(match[1]), peak_rss_kib=int(match[2]))
        output = result.stdout.decode('utf-8','replace')
        # Only private evidence carries child output.
        receipt.update(stdout=output, stderr=result.stderr.decode('utf-8','replace')[-4000:])
        for label, data in (('stdout',result.stdout),('stderr',result.stderr)):
            receipt[label+'_sha256']=hashlib.sha256(data).hexdigest()
            receipt[label+'_bytes']=len(data)
            receipt[label+'_excerpt_truncated']=result.limit == 'output'
        if result.limit == 'output':
            raise EvaluationError('excessive program output', receipt)
        if not result.ok:
            raise EvaluationError('evaluation timeout' if result.timed_out else 'build, source policy, or execution failed', receipt)
        return output, receipt

def source_identity(tree):
    files = {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(tree.rglob('*')) if p.is_file() and '.git' not in p.parts}
    return {'sha256':hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), 'files':files}

def observation_prefix():
    """Unpredictable per-invocation prefix; it exists only in the derived main."""
    return 'CKKS_OBSERVATION_' + secrets.token_hex(16) + ' '

def invoke_original(tree, name, overlay, deadline):
    entry = ENTRYPOINT if name == 'mul_relin_rescale' else f'estimator/experiments/{name}/{name}.go'
    results=[]
    def launch_go(args, cwd, end):
        # The importer supplies the original source name; observation flags only.
        command=[*args[:3], '-overlay', str(overlay), *args[3:]]
        result=run_measured(command,cwd,end);results.append(result);return result[0]
    with registered_go_entrypoint(tree, entry, launch_go, deadline) as launch:
        if name == 'mul_relin_rescale':
            runpy.run_path('/app/estimator/experiments/mul_relin_rescale/mul_relin_rescale.go', init_globals=launch)
        else:
            runpy.run_path('/app/'+entry, init_globals=launch)
    if len(results)!=1: raise EvaluationError('original entrypoint did not execute once')
    return results[0]

def parse_observations(output, prefix, name):
    """Accept exactly the derived main's lines; anything else on stdout is ignored."""
    events=[json.loads(line[len(prefix):]) for line in output.splitlines() if line.startswith(prefix)]
    if [v.get('kind') for v in events]!=['parameters','final','final','estimator_ns'] or any(set(v)!={'kind','value'} for v in events):
        raise EvaluationError('missing, duplicate or reordered trusted observations')
    final=[events[1]['value'],events[2]['value']]
    n=2 if name=='coeffs_to_slots' else 1
    for row in final:
        if not isinstance(row,list) or len(row)!=3 or row[2]!=n or any(type(v) not in (int,float) or not math.isfinite(v) for v in row) or row[1]<0:
            raise EvaluationError('non-finite or invalid precision observations')
    elapsed=events[3]['value']
    if type(elapsed) is not int or elapsed<=0:
        raise EvaluationError('invalid estimator clock observation')
    params=[events[0]['value']]
    json.dumps(params,allow_nan=False)
    a,s=final[0][:2]; b,t=final[1][:2]
    return {'predicted_avg':a,'predicted_std':s,'actual_avg':b,'actual_std':t,
            METRICS[0]:abs(a-b),METRICS[1]:abs(s-t),'estimator_ns':elapsed,
            'resolved_parameters':params}

def measure_one(tree, name, dest, deadline, identity):
    """One complete fresh invocation of one configuration. Failures are recorded."""
    configuration=WORKLOADS[name];circuit=configuration['circuit']
    entry=configuration['main_path']
    record={'workload':name,'source_sha256':identity['sha256'],
            'entrypoint_sha256':identity['files'][entry], 'source_invocations':0,
            'launch_attempts':0,'main_completed':0,
            'randomness':RELEASE['randomization'],'status':'pending'}
    row=None
    try:
        prefix=observation_prefix()
        overlay,receipt=prepare_overlay(tree,dest,prefix,configuration)
        record['configuration_binding']=receipt
        output,receipt=invoke_original(tree,circuit,overlay,deadline)
        record.update(receipt);record['source_invocations']=1
        row=parse_observations(output,prefix,circuit)
        verify_parameters(row['resolved_parameters'],configuration)
        record.update(row)
        record.pop('stdout',None)
    except isolate.IsolateError:
        raise
    except EvaluationError as exc:
        row=None;record.update(exc.receipt);record['source_invocations']=int(record.get('main_completed',0));record['status']='invalid';record['reason']=str(exc)
    except Exception as exc:
        row=None;record['status']='invalid';record['reason']='observation decoding failed';record['exception_type']=type(exc).__name__
    if record.get('status')!='ok' and 'stdout' in record:
        record.setdefault('stdout_sha256',hashlib.sha256(record['stdout'].encode()).hexdigest())
        # The prefix never leaves the evaluator; a failed row keeps only a digest.
        record.pop('stdout')
    return row,record

def measure_suite(tree, names, dest, deadline, *, coverage=False):
    """One invocation of every configuration (the calibration batch unit)."""
    if len(names) != len(set(names)) or any(name not in WORKLOADS for name in names):
        raise EvaluationError('unknown or duplicate configured workload')
    identity=source_identity(tree);rows={};records=[]
    for name in names:
        row,record=measure_one(tree,name,Path(dest)/name,deadline,identity)
        if row is not None: rows[name]=row
        records.append(record) # Failed runs and deadline-exhausted rows are never dropped.
    return rows,{'source_identity':identity,'observations':records,
                 'instrumentation':'trusted derived mains; package estimator not overlaid'}

def measure_pairs(candidate, reference, names, dest, deadline, pairs):
    """Interleave candidate and reference invocations of each configuration."""
    if len(names) != len(set(names)) or any(name not in WORKLOADS for name in names):
        raise EvaluationError('unknown or duplicate configured workload')
    identities={'candidate':source_identity(candidate),'reference':source_identity(reference)}
    trees={'candidate':candidate,'reference':reference}
    rows={side:{} for side in trees};records={side:[] for side in trees}
    for name in names:
        for index in range(pairs):
            # Alternate which side runs first so neither owns a position effect.
            for side in (('candidate','reference') if index%2==0 else ('reference','candidate')):
                row,record=measure_one(trees[side],name,Path(dest)/side/name/str(index),deadline,identities[side])
                record['pair']=index;records[side].append(record)
                if row is not None: rows[side].setdefault(name,[]).append(row)
    complete={side:{name:value for name,value in rows[side].items() if len(value)==pairs} for side in trees}
    return complete,{side:{'source_identity':identities[side],'observations':records[side],
                           'instrumentation':'trusted derived mains; package estimator not overlaid'} for side in trees}

def mean_rows(rows):
    """Per configuration: mean discrepancy over its invocations, and its clocks."""
    return {name:{**{k:sum(r[k] for r in value)/len(value) for k in METRICS},
                  'estimator_ns':[r['estimator_ns'] for r in value]} for name,value in rows.items()}

# When check_tree started: the evaluation deadline counts from here, as it counted from the
# start of the old single evaluate() that also ran the source policy.
STARTED = None

def check_tree(tree, pristine, changes):
    """Shared-verifier hook: the judge-owned Go AST policy (imports, package-level work,
    goroutines, channels, the protected dft.go declarations) over the judged tree, compared
    with the pristine starter. A violation is the shared `content_policy` rejection; a policy
    that cannot be built is the judge's own fault and raises."""
    global STARTED
    STARTED = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='ckks-policy-', dir=scratch_root()) as d:
        policy = Path(d)/'source-policy'
        subprocess.run(['go','build','-p=1','-o',str(policy),str(TASK_DIR/'judge/source_policy.go')],
                       cwd=TASK_DIR, env=go_env(), check=True, capture_output=True, timeout=300)
        # The judge's own parser over the submitted source (it runs none of it).
        verdict = subprocess.run([str(policy), str(pristine), str(tree)], cwd=TASK_DIR, env=go_env(),
                                 capture_output=True, timeout=120)
    if verdict.returncode == 0:
        return None
    return 'content_policy', sorted(c.path for c in changes if c.status != 'D') or ['estimator/']

def guards(tree, dest, deadline):
    code=(TASK_DIR/'judge/contracts.go.txt').read_text()+'\nfunc main(){frontierContracts()}\n'
    guard=dest/'contracts.go';guard.write_text(code)
    run(['go','run','-p=1',guard],tree/'estimator',deadline)

def accuracy(rows):
    error=sum(row[k] for row in rows for k in METRICS)/(2*len(rows))
    return 100.0/(1.0+error/SCORE_BITS)

def restore_reference(scratch):
    """The reference tree: a copy of the pristine starter with judge/reference.patch and then
    the private comment restoration applied. Both patches are the trusted evaluator's own;
    a submission cannot select either or change its path."""
    tree = Path(scratch)/'reference-tree'  # measure_pairs writes its overlays under scratch/reference
    shutil.copytree(PRISTINE, tree, symlinks=True)
    for path in (tree, *tree.rglob('*')):
        if not path.is_symlink(): path.chmod(path.stat().st_mode | 0o200)
    env = {**go_env(), 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
           'GIT_CEILING_DIRECTORIES': str(Path(scratch).resolve())}
    try:
        subprocess.run(['git','apply','--whitespace=nowarn',str(TASK_DIR/'reference.patch')],
                       cwd=tree, env=env, check=True, capture_output=True, timeout=30)
    except Exception as exc:
        raise EvaluationError('reference setup failed') from exc
    try:
        subprocess.run(['git','apply','--whitespace=nowarn',
                        str(TASK_DIR/'private-comment-restoration.patch')],
                       cwd=tree, env=env, check=True, capture_output=True, timeout=30)
        return tree
    except Exception as exc:
        raise EvaluationError('private reference comment restoration failed') from exc

def formal_record():
    from formal_scoring import read_calibration
    if RELEASE.get('mode') != 'formal' or not isinstance(RELEASE.get('formal_calibration'),dict):
        raise EvaluationError('formal calibration is not configured')
    declared=RELEASE['formal_calibration']
    return read_calibration(TASK_DIR/declared['file'],expected_sha256=declared['sha256'],
        final_workloads=FINAL_SUITE,expected_binding=declared['measurement_binding'])

def failed_result(names, evidence, message, *, calibration_invalid=False):
    """Never recurse through calibration when reporting a calibration error."""
    metrics = {
        'status': 'calibration_unavailable' if calibration_invalid else 'invalid_execution',
        'beats_reference': 0, 'continuous_accuracy_score': 0.0,
        'reference_score': 0.0, 'margin': 0.0,
        'calibration_pending': calibration_invalid,
        'release': RELEASE['release'], 'workload_count': len(names),
        'verification_mode': 'final' if os.environ.get('FRONTIER_SUBMISSION_ROLE') == 'final' else 'development',
        'beat_spec_workload_passes': [
            {'workload': w, 'metrics': {k: False for k in (*METRICS, TIME_METRIC)}} for w in names],
        'operator_evidence': pack_evidence(evidence),
    }
    for k in METRICS:
        metrics[k] = {}; metrics['reference_'+k] = {}
        metrics['candidate_'+k.replace('_bits','_mean_bits')] = 0.0
    metrics['candidate_estimator_seconds_mean'] = 0.0
    return 0.0, 0.0, message, metrics

def pack_evidence(evidence):
    if final_role():
        # Full 29-by-3 receipts belong in the adapter's authenticated member,
        # rather than the bounded outward metrics object.
        from ckks_evidence_transport import packed
        try:
            return packed(evidence)
        except ValueError as error:
            raise EvaluationError('private evidence exceeds transport limits') from error
    # Native adapter caps JSON depth at six and individual strings at 4096.
    # Chunk a lossless canonical JSON document; hashes detect truncation.
    body=json.dumps(evidence,sort_keys=True,allow_nan=False,separators=(',',':'))
    return {'encoding':'json-chunks-v1','sha256':hashlib.sha256(body.encode()).hexdigest(),
            'chunks':[body[i:i+1800] for i in range(0,len(body),1800)]}

def unpack_evidence(value):
    if value.get('encoding') != 'json-chunks-v1':
        from ckks_evidence_transport import unpack
        try:
            return unpack(value)
        except (ValueError, TypeError, KeyError) as error:
            raise EvaluationError('private evidence integrity failed') from error
    body=''.join(value['chunks'])
    if value['encoding']!='json-chunks-v1' or hashlib.sha256(body.encode()).hexdigest()!=value['sha256']:
        raise EvaluationError('private evidence integrity failed')
    return json.loads(body)

def summarize(candidate, reference, names, evidence):
    from formal_scoring import compare
    mode='final' if os.environ.get('FRONTIER_SUBMISSION_ROLE')=='final' else 'development'
    calibration=formal_record()
    complete=set(candidate)==set(reference)==set(names)
    if complete:
        result=compare(candidate,reference,list(names),calibration,mode=mode,pairs=pairs(),development=DEV_FEEDBACK)
        score=result.pop('score');refscore=result['reference_score']
    else:
        score=refscore=0.0
        result={'beats_reference':0,'margin':0.0,'reference_score':0.0,
                'beat_spec_workload_passes':[{'workload':w,'metrics':{k:False for k in (*METRICS,TIME_METRIC)}} for w in names]}
    evidence.update(verification_mode=mode,workload_manifest=list(names),interleaved_pairs=pairs(),
                    calibration_sha256=RELEASE['formal_calibration']['sha256'])
    metrics={**result,'continuous_accuracy_score':score,'calibration_pending':False,
             'status':'valid_formal_measurement' if complete else 'invalid_execution',
             'verification_mode':mode,'workload_count':len(names),'operator_evidence':pack_evidence(evidence),
             'calibration_sha256':RELEASE['formal_calibration']['sha256']}
    for k in METRICS:
        metrics[k]={w:r[k] for w,r in candidate.items()}
        metrics['reference_'+k]={w:r[k] for w,r in reference.items()}
        metrics['candidate_'+k.replace('_bits','_mean_bits')]=sum(r[k] for r in candidate.values())/len(candidate) if complete else 0.0
    metrics[TIME_METRIC]={w:[v/1e9 for v in r['estimator_ns']] for w,r in candidate.items()}
    metrics['reference_'+TIME_METRIC]={w:[v/1e9 for v in r['estimator_ns']] for w,r in reference.items()}
    clocks=[v/1e9 for r in candidate.values() for v in r['estimator_ns']]
    metrics['candidate_estimator_seconds_mean']=sum(clocks)/len(clocks) if complete else 0.0
    if len(json.dumps(metrics,sort_keys=True,allow_nan=False).encode())>500*1024:
        raise EvaluationError('formal evidence exceeds native bound')
    return score,score,'fresh scaled CKKS accuracy and estimator-time measurement' if complete else 'invalid execution; see private evidence',metrics

def evaluate(tree: str) -> tuple[float,float,str,dict]:
    """`tree` is the pristine starter with the submission's allowed files overlaid.

    Only the formal release is judged here. The calibration release that produced
    judge/formal_calibration.json (collection over exact reference payloads) is recorded under
    docs/task-authoring/ckks_composable_noise/provenance/; any other release mode fails closed
    below as an unavailable calibration."""
    names=suite();evidence={}
    started=STARTED if STARTED is not None else time.monotonic()
    try:
        formal_record()
    except Exception as exc:
        evidence['failure']={'reason':'current release calibration unavailable', 'type':type(exc).__name__}
        return failed_result(names, evidence, 'current release calibration unavailable', calibration_invalid=True)
    try:
        deadline=started+CONFIG['runtime']['timeout_seconds']-10 # reserve serialization/cleanup time
        tempfile.tempdir=str(scratch_root())
        candidate=Path(tree)
        with tempfile.TemporaryDirectory(prefix='ckks-evaluation-', dir=scratch_root()) as d:
            d=Path(d)
            reference=restore_reference(d)
            version=subprocess.run(['go','version'],cwd=TASK_DIR,env=go_env(),capture_output=True,text=True,
                                   check=True,timeout=max(0.001,deadline-time.monotonic())).stdout.strip()
            if RELEASE['toolchain'] not in version: raise EvaluationError('incorrect Go toolchain')
            evidence['toolchain']=version
            guards(candidate,d,deadline)
            # Fresh interleaved reference comparison. Never a frozen point reference.
            rows,detail=measure_pairs(candidate,reference,names,d,deadline,pairs())
            evidence.update(detail)
            return summarize(mean_rows(rows['candidate']),mean_rows(rows['reference']),names,evidence)
    except EvaluationError as exc:
        evidence['failure']={'reason':str(exc),**exc.receipt}
        return failed_result(names, evidence, str(exc))
    except isolate.IsolateError:
        raise  # the boundary could not be built: a failed verification, not a score
    except Exception as exc:
        evidence['failure']={'reason':'invalid submission or evaluation failure','type':type(exc).__name__}
        return failed_result(names, evidence, 'invalid submission or evaluation failure')

if __name__=='__main__':
    result=evaluate(sys.argv[1] if len(sys.argv)==2 else '')
    print(json.dumps(dict(zip(('score','score_unbounded','message','metrics'),result)),allow_nan=False))
