"""Scaled task contract checks; synthetic calibration fixtures are not measurements."""
import deployment_test_guard  # noqa: F401  # isort: skip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import pytest
REPOSITORY=Path(__file__).resolve().parents[4]
TASK=REPOSITORY/'tasks/frontier-cs-2.0-demo/problems/Crypto/ckks_composable_noise'
sys.path.insert(0,str(REPOSITORY/'scripts'))
import task_manifest  # noqa: E402
# The judge as the Verifier image holds it: judge/ plus the generated config.yaml and pristine/.
sys.path.insert(0,str(task_manifest.judge_bundle(TASK)))
# Generic judge module names are shared with other tasks; load this task's own
# whatever the interpreter imported first.
for _name in ('evaluator','formal_scoring','workload_configurations',
              'go_entrypoint','ckks_evidence_transport'):
    sys.modules.pop(_name,None)
import evaluator as e
PROVENANCE=REPOSITORY/'docs/task-authoring/ckks_composable_noise/provenance'

def patched_paths(text):
    return {line.split(' b/',1)[1] for line in text.splitlines() if line.startswith('diff --git a/')}

@pytest.fixture(autouse=True)
def formal_mode(monkeypatch):
    monkeypatch.setitem(e.RELEASE, 'mode', 'formal')

GO=pytest.mark.skipif(shutil.which('go') is None,reason='needs the Go toolchain of the verifier image')
FAST,SLOW=[1,1,1],[2,2,2]
def measured(value,clocks,names=None):
    return {w:{**{k:value for k in e.METRICS},'estimator_ns':list(clocks)} for w in (names or e.FINAL_SUITE)}

@pytest.fixture(scope='module')
def restored(tmp_path_factory):
    # Reference tree with the private comment restoration applied.
    return e.restore_reference(tmp_path_factory.mktemp('reference'))

def test_reference_restoration_and_retained_bytes(restored):
    provenance=json.loads((PROVENANCE/'source_provenance.json').read_text())
    pins=dict(provenance['authorized_parameter_derivation']['files'])
    for name,digest in pins.items():
        assert hashlib.sha256((restored/name).read_bytes()).hexdigest()==digest
    assert patched_paths((e.TASK_DIR/'reference.patch').read_text())=={'estimator/operations.go','estimator/noise.go'}

def test_configuration_entrypoints_and_split(monkeypatch):
    assert len(e.DEV_SUITE)==21 and len(e.FINAL_SUITE)==29
    assert set(e.DEV_SUITE)<set(e.FINAL_SUITE)
    assert {e.WORKLOADS[w]['circuit'] for w in set(e.FINAL_SUITE)-set(e.DEV_SUITE)}=={'iterative_step','mod1','bootstrapping'}
    for role in ('agent','train','development','arbitrary',''):
        monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE',role);assert e.suite()==e.DEV_FEEDBACK and e.pairs()==1
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE','final');assert e.suite()==e.FINAL_SUITE and e.pairs()==3
    # Development feedback is a declared subset of the calibrated development rows.
    assert e.DEV_FEEDBACK==e.DEV_SUITE[:6] and all(e.WORKLOADS[w]['original_row'] for w in e.DEV_FEEDBACK)

def test_invalid_never_crashes(tmp_path):
    # Patches are applied by the shared verifier; an unusable tree is still a scored zero.
    for tree in (tmp_path, tmp_path/'missing'):
        a,b,msg,m=e.evaluate(str(tree));assert a==b==0 and msg

def test_formal_and_submitted_mode_fail_closed(monkeypatch):
    monkeypatch.setitem(e.RELEASE, 'formal_calibration', None)
    monkeypatch.setenv('CKKS_MODE','development')
    monkeypatch.setenv('FRONTIER_CALIBRATION','approved')
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE','final')
    # No calibration exists for this release: nothing may be measured or passed.
    assert e.RELEASE['mode']=='formal' and e.RELEASE['formal_calibration'] is None and e.suite()==e.FINAL_SUITE
    monkeypatch.setattr(e,'measure_pairs',lambda *a,**k:pytest.fail('must not start measurement'))
    monkeypatch.setattr(e,'guards',lambda *a,**k:pytest.fail('must not build the submission'))
    for path in ('/absent',str(e.PRISTINE)):
        result=e.evaluate(path)
        assert result[0]==0 and result[3]['status']=='calibration_unavailable'
        assert result[3]['beats_reference']==0 and result[3]['calibration_pending'] is True

@pytest.fixture
def synthetic_calibration(monkeypatch):
    cells=[w+'/'+m for w in e.FINAL_SUITE for m in e.METRICS]
    fixture={'development_workloads':list(e.DEV_SUITE),'frozen':{'cells':cells,'bands':dict.fromkeys(cells,.25)}}
    monkeypatch.setattr(e,'formal_record',lambda:fixture)
    monkeypatch.setitem(e.RELEASE,'formal_calibration',{'sha256':'unit-test-only'})
    return fixture

def test_score_independent_of_verdict_and_orders_discrepancy(monkeypatch, synthetic_calibration):
    monkeypatch.setenv("FRONTIER_SUBMISSION_ROLE","final")
    exact=e.summarize(measured(0,FAST),measured(0,SLOW),e.FINAL_SUITE,{})
    partial=e.summarize(measured(0.1,FAST),measured(0,SLOW),e.FINAL_SUITE,{})
    weak=e.summarize(measured(1,FAST),measured(0,SLOW),e.FINAL_SUITE,{})
    assert 0<weak[0]<partial[0]<exact[0]==100
    assert exact[3]['beats_reference']==1 and weak[3]['beats_reference']==0
    assert all(not x[3]['calibration_pending'] for x in (weak,partial,exact))
    # The continuous score ignores the clock entirely.
    assert e.summarize(measured(0.1,SLOW),measured(0,FAST),e.FINAL_SUITE,{})[0]==partial[0]

def test_metrics_complete_and_failure_not_omitted(synthetic_calibration):
    names={m['name'] for m in json.loads((REPOSITORY/'docs/task-authoring/ckks_composable_noise/provenance/beat_spec.json').read_text())['metrics']}  # archived design record
    r=e.summarize({}, {},e.FINAL_SUITE,{'failed':'fixture'})
    assert r[0]==0 and e.unpack_evidence(r[3]['operator_evidence'])['failed']=='fixture'
    rows=r[3]['beat_spec_workload_passes']
    assert len(rows)==29 and len({x['workload'] for x in rows})==29
    assert all(set(x['metrics'])==names and all(type(v) is bool for v in x['metrics'].values()) for x in rows)
    assert r[3]['beats_reference']==int(all(v for row in rows for v in row['metrics'].values()))
    assert {'reference_score','margin','calibration_pending'}<=r[3].keys()

def test_only_the_experiment_main_is_overlaid(restored,tmp_path):
    for name in e.FINAL_SUITE:
        row=e.WORKLOADS[name];prefix=e.observation_prefix()
        overlay,receipt=e.prepare_overlay(restored,tmp_path/name,prefix,row)
        replacements=json.loads(overlay.read_text())['Replace']
        assert list(replacements)==[str(restored/row['main_path'])]
        derived=Path(replacements[str(restored/row['main_path'])]).read_text()
        assert derived.count(prefix)==1 and '%.2f' not in derived.split('func main() {')[1].split('frontierReport(')[0]
        assert prefix.startswith('CKKS_OBSERVATION_') and prefix not in json.dumps(receipt)

def observation_lines(prefix,final=([40.0,2.0,1],[39.5,2.5,1]),clock=123456,parameters=None):
    events=[{'kind':'parameters','value':parameters or {'LogN':12}},{'kind':'final','value':list(final[0])},
            {'kind':'final','value':list(final[1])},{'kind':'estimator_ns','value':clock}]
    return [prefix+json.dumps(v) for v in events]

def test_missing_nonfinite_duplicate_observations_rejected():
    for s in ('','P '+json.dumps({'kind':'final','value':[float('nan'),1,1]})):
        with pytest.raises(e.EvaluationError):e.parse_observations(s,'P ','mul_relin_rescale')
    good=observation_lines('P ')
    row=e.parse_observations('\n'.join(good),'P ','mul_relin_rescale')
    assert row['estimator_ns']==123456 and row[e.METRICS[0]]==0.5 and row[e.METRICS[1]]==0.5
    bad=[good[:3],good+good[3:],good[:1]+good[2:]+good[1:2],[good[0],good[2],good[1],good[3]][::-1],
         observation_lines('P ',clock=0),observation_lines('P ',clock=1.5),observation_lines('P ',clock=True),
         observation_lines('P ',final=([40.0,2.0,2],[39.5,2.5,1])),observation_lines('P ',final=([40.0,-1.0,1],[39.5,2.5,1])),
         good[:3]+['P '+json.dumps({'kind':'estimator_ns','value':5,'extra':1})]]
    for lines in bad:
        with pytest.raises(e.EvaluationError):e.parse_observations('\n'.join(lines),'P ','mul_relin_rescale')

def test_forged_observation_lines_are_ignored_or_rejected():
    """Output printed by submitted code never has this invocation's prefix."""
    prefix=e.observation_prefix();other=e.observation_prefix()
    assert prefix!=other and len(prefix)==len('CKKS_OBSERVATION_')+33
    honest=observation_lines(prefix)
    forged=observation_lines(other,final=([60.0,0.0,1],[60.0,0.0,1]),clock=1)
    forged+=observation_lines('CKKS_OBSERVATION_PLACEHOLDER ',clock=1)+observation_lines('CKKS_OBSERVATION_ ',clock=1)
    row=e.parse_observations('\n'.join(forged[:4]+honest[:2]+forged[4:]+honest[2:]),prefix,'mul_relin_rescale')
    assert row['estimator_ns']==123456 and row['predicted_avg']==40.0
    # Guessing the prefix cannot help either: a second set of lines is a rejection.
    with pytest.raises(e.EvaluationError):
        e.parse_observations('\n'.join(honest+observation_lines(prefix,clock=1)),prefix,'mul_relin_rescale')
    # A failed row never returns raw child output to the evidence, so no prefix leaks.
    def boom(*a,**k):raise e.EvaluationError('x',{'stdout':prefix+'{}','status':'failed'})
    import pytest as _p
    mp=_p.MonkeyPatch()
    try:
        mp.setattr(e,'invoke_original',boom)
        identity=e.source_identity(e.PRISTINE)
        row,record=e.measure_one(e.PRISTINE,'mul_relin_rescale',Path(e.scratch_root())/'forged-row',10,identity)
    finally:mp.undo()
    assert row is None and 'stdout' not in record and 'CKKS_OBSERVATION_' not in json.dumps(record)

def test_no_credentials_and_writable_nonroot_caches(monkeypatch):
    monkeypatch.setenv('EXAMPLE_PROVIDER_KEY','not-a-real-key');monkeypatch.delenv('HOME',raising=False)
    # The toolchain's environment (go_env) and the measured program's (program_env, the only
    # variables an isolate instance gets) carry no credentials.
    env=e.go_env()
    assert 'EXAMPLE_PROVIDER_KEY' not in env and 'HOME' not in env and env['GOPROXY']=='off'
    assert env['CGO_ENABLED']=='0'
    assert all(Path(env[k]).is_dir() for k in ('GOCACHE','GOTMPDIR','GOPATH'))
    assert set(e.program_env())=={'GOMAXPROCS','GOMEMLIMIT'}

def test_packaging_stages_helpers(tmp_path):
    repo=tmp_path/'repo';shutil.copytree(e.PRISTINE,repo)
    def git(*args):return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True)
    git('init','-q');git('add','-A');git('-c','user.name=Fixture','-c','user.email=fixture@invalid','commit','-qm','base')
    git('apply',str(Path(__file__).resolve().parent/'valid_helper.patch'))
    subprocess.run(['sh','make_submission.sh'],cwd=repo,check=True,capture_output=True)
    assert 'estimator/solver_extensions/identity/identity.go' in patched_paths((repo/'solution.patch').read_text())
    (repo/'LICENSE').write_text('invalid')
    assert subprocess.run(['sh','make_submission.sh'],cwd=repo,capture_output=True).returncode!=0

def test_go_adapter_actual_dispatch_and_failure(monkeypatch,tmp_path):
    commands=[]
    def execute(args,cwd,deadline):
        commands.append(([str(x) for x in args],str(cwd)))
        return '',{'status':'ok'}
    monkeypatch.setattr(e,'run_measured',execute)
    tree=e.PRISTINE;before=(tree/e.ENTRYPOINT).read_bytes()
    overlay=tmp_path/'overlay.json';overlay.write_text('{}')
    e.invoke_original(tree,'mul_relin_rescale',overlay,123)
    assert commands==[(['go','run','-p=1','-overlay',str(overlay),'mul_relin_rescale.go'],str((tree/e.ENTRYPOINT).parent))]
    assert (tree/e.ENTRYPOINT).read_bytes()==before
    def fail(*args):raise e.EvaluationError('compiler failed')
    monkeypatch.setattr(e,'run_measured',fail)
    with pytest.raises(e.EvaluationError,match='compiler failed'):e.invoke_original(tree,'mul_relin_rescale',overlay,123)
    assert '/app/'+e.ENTRYPOINT not in sys.path_importer_cache

def test_go_adapter_requires_invocation_and_rejects_escape():
    from go_entrypoint import registered_go_entrypoint
    tree=e.PRISTINE
    with pytest.raises(ValueError):
        with registered_go_entrypoint(tree,'../escape.go',e.run,123):pass
    with pytest.raises(RuntimeError,match='was not executed'):
        with registered_go_entrypoint(tree,e.ENTRYPOINT,e.run,123):pass

@GO
def test_source_policy_accepts_numerical_helpers_rejects_tampering(tmp_path):
    tree=tmp_path/'tree';shutil.copytree(e.PRISTINE,tree)
    policy=tmp_path/'policy';deadline=time.monotonic()+120
    e.run(['go','build','-p=1','-o',policy,e.TASK_DIR/'judge/source_policy.go'],e.TASK_DIR,deadline)
    helper=tree/'estimator/tables';helper.mkdir(parents=True)
    (helper/'tables.go').write_text('package tables\nimport "math/bits"\nvar Weights = [...]float64{1,2,3}\nfunc Width(x uint)int{return bits.Len(x)}\n')
    e.run([policy,e.PRISTINE,tree],tree,deadline)
    p=tree/'estimator/noise.go';original=p.read_text()
    # Editable files may change freely; only what they import and schedule is policed.
    p.write_text(original.replace('sigma *= math.Sqrt','sigma *= 1 * math.Sqrt'))
    e.run([policy,e.PRISTINE,tree],tree,deadline)
    for bad in (original.replace('"math"','"math"\n"os"'),original.replace('"math"','"math"\n"sync"'),
                original.replace('time.Now().UnixNano()','time.Since(time.Now()).Nanoseconds()')):
        assert bad!=original;p.write_text(bad)
        with pytest.raises(e.EvaluationError):e.run([policy,e.PRISTINE,tree],tree,deadline)
    p.write_text(original)
    (helper/'tables.go').write_text('package tables\nfunc init(){}\n')
    with pytest.raises(e.EvaluationError):e.run([policy,e.PRISTINE,tree],tree,deadline)

@GO
def test_check_tree_reports_policy_violations_as_content_policy(tmp_path):
    from types import SimpleNamespace
    tree=tmp_path/'tree';shutil.copytree(e.PRISTINE,tree)
    change=[SimpleNamespace(status='M',path='estimator/noise.go')]
    assert e.check_tree(str(tree),str(e.PRISTINE),change) is None
    p=tree/'estimator/noise.go';p.write_text(p.read_text().replace('"math"','"math"\n"os"'))
    assert e.check_tree(str(tree),str(e.PRISTINE),change)==('content_policy',['estimator/noise.go'])

@GO
def test_all_original_and_measured_mains_compile_only(restored,tmp_path):
    deadline=time.monotonic()+900
    e.run(['go','build','-p=1','./...'],e.PRISTINE/'estimator',deadline)
    for name in e.FINAL_SUITE:
        row=e.WORKLOADS[name]
        overlay,_=e.prepare_overlay(restored,tmp_path/name,e.observation_prefix(),row)
        e.run(['go','build','-p=1','-overlay',overlay,'-o',tmp_path/'bin','./experiments/'+row['circuit']],restored/'estimator',deadline)

def test_absolute_importlib_stdlib_loading(tmp_path):
    code='import importlib.util,sys;s=importlib.util.spec_from_file_location("e",sys.argv[1]);m=importlib.util.module_from_spec(s);sys.modules["e"]=m;s.loader.exec_module(m);assert m.evaluate("missing")[0]==0'
    subprocess.run([sys.executable,'-c',code,str(e.TASK_DIR/'evaluator.py')],cwd=tmp_path,check=True,capture_output=True)

def test_private_evidence_chunks_are_lossless():
    value={'parameters':{'nested':{'deep':{'deeper':{'primes':[2**60+7]*1000}}}},'failures':['invalid']}
    packed=e.pack_evidence(value)
    assert all(len(c)<=1800 for c in packed['chunks'])
    assert e.unpack_evidence(packed)==value
    packed['chunks'][0]+='x'
    with pytest.raises(e.EvaluationError):e.unpack_evidence(packed)

def test_authored_modes_allow_nonroot_reading():
    for p in e.TASK_DIR.rglob('*'):
        if '__pycache__' in p.parts:continue
        if p.is_dir():assert p.stat().st_mode & 0o005 == 0o005,str(p)
        elif p.is_file():assert p.stat().st_mode & 0o004,str(p)


def test_measured_main_insertions_are_exactly_reversible(restored):
    """The derived main is the fixed main plus inserted statements, nothing else."""
    import workload_configurations as c
    for tree in (e.PRISTINE,restored):
        for name in e.FINAL_SUITE:
            row=e.WORKLOADS[name];source=(tree/row['main_path']).read_text()
            configured=c.configured_main(source,row);derived=c.timed_main(configured,row)
            kept=[];skip=0
            for line in derived.split('\n'):
                text=line.strip()
                if 'frontierReport := func' in text:skip=6;continue
                if skip:skip-=1;continue
                if 'frontier' in text:continue
                kept.append(line)
            assert '\n'.join(kept)==configured,name
            assert c.measured_main(source,row,'CKKS_OBSERVATION_'+'0'*32+' ').count('CKKS_OBSERVATION_'+'0'*32+' ')==1
            with pytest.raises(ValueError):c.measured_main(source,row,'CKKS_OBSERVATION_guess ')

def test_launch_errors_keep_command_and_do_not_claim_completed_main(monkeypatch,tmp_path):
    # run_measured runs only `go run` commands; anything else is a judge bug.
    with pytest.raises(ValueError):
        e.run_measured(['missing-tool'],tmp_path,time.monotonic()+10)
    # A build that fails keeps the command and claims no completed main.
    monkeypatch.setattr(e.subprocess,'run',lambda *a,**k: subprocess.CompletedProcess(a[0],2,b'',b'no toolchain'))
    with pytest.raises(e.EvaluationError) as failure:
        e.run_measured(['go','run','-p=1','main.go'],tmp_path,time.monotonic()+10)
    assert failure.value.receipt['launch_attempts']==1
    assert failure.value.receipt['main_completed']==0
    assert failure.value.receipt['command']==['go','run','-p=1','main.go']


def test_outer_failure_retains_private_diagnostics(monkeypatch, synthetic_calibration):
    real_run=e.subprocess.run
    def toolchain(args,*rest,**kwargs):  # the judge asks the Go toolchain for its version
        if list(args)[:2]==['go','version']:
            return subprocess.CompletedProcess(args,0,'go version go1.23.12 linux/amd64\n','')
        return real_run(args,*rest,**kwargs)
    monkeypatch.setattr(e.subprocess,'run',toolchain)
    monkeypatch.setattr(e,'guards',lambda *args: (_ for _ in ()).throw(OSError()))
    result=e.evaluate(str(e.PRISTINE))
    assert result[0]==0
    assert e.unpack_evidence(result[3]['operator_evidence'])['failure']['type']=='OSError'


def test_oversize_evidence_fails_explicitly_without_native_rejection(monkeypatch, synthetic_calibration):
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE','final')
    rows=measured(0.1,FAST)
    with pytest.raises(e.EvaluationError,match='evidence exceeds'):
        e.summarize(rows,rows,e.FINAL_SUITE,{'oversize':'x'*(25*1024*1024)})


def test_full_suite_transport_preserves_all_scientific_observations(monkeypatch, synthetic_calibration):
    monkeypatch.setenv("FRONTIER_SUBMISSION_ROLE","final")
    # Shape fixture only: expands one measured primitive row; no composition claim.
    sys.path.insert(0,str(task_manifest.SHARED/'verifier'))
    native=pytest.importorskip('fcs2_verifier',reason='the shared verifier adapter')
    observations=[{'workload':w,'raw':[[32.0,1.8],[31.9,1.9]],'parameters':{'Q':[2**60]*25,'boot':{'nested':{'scales':[45,55]}}}} for w in e.FINAL_SUITE]
    rows=measured(0.1,FAST)
    evidence={'candidate':observations,'reference':observations}
    result=e.summarize(rows,rows,e.FINAL_SUITE,evidence)
    # The shared verifier keeps the packed private evidence in the evidence archive's metrics.
    bounded=native.run_evaluator(lambda _:result,Path('fixture-tree'),'eval',max_metrics_bytes=e.CONFIG['evaluation']['max_metrics_bytes'])
    assert {k:v for k,v in e.unpack_evidence(bounded[3]['operator_evidence']).items() if k in evidence}==evidence
    assert len(bounded[3]['beat_spec_workload_passes'])==29


def test_actual_evaluator_routes_fresh_complete_modes(monkeypatch,tmp_path,synthetic_calibration):
    from types import SimpleNamespace
    calls=[];patches=[]
    def patch(scratch):
        patches.append(str(scratch));return tmp_path/'reference'
    monkeypatch.setattr(e,'restore_reference',patch)
    monkeypatch.setattr(e,'guards',lambda *args:None)
    real_run=e.subprocess.run
    def toolchain(args,*rest,**kwargs):  # the judge asks the Go toolchain for its version
        if list(args)[:2]==['go','version']:
            return subprocess.CompletedProcess(args,0,'go version go1.23.12 linux/amd64\n','')
        return real_run(args,*rest,**kwargs)
    monkeypatch.setattr(e.subprocess,'run',toolchain)
    clocks={'candidate':100,'reference':200}
    def measure(candidate,reference,names,d,deadline,pairs):
        calls.append((candidate,reference,tuple(names),pairs))
        row=lambda side:{**{m:0.0 for m in e.METRICS},'estimator_ns':clocks[side]}
        return ({side:{w:[row(side)]*pairs for w in names} for side in clocks},
                {side:{'fixture_only':True} for side in clocks})
    monkeypatch.setattr(e,'measure_pairs',measure)
    for role,names,count in [('train',e.DEV_FEEDBACK,1),('final',e.FINAL_SUITE,3)]:
        for candidate_clock,expected in ((100,1),(200,0),(300,0)):
            clocks['candidate']=candidate_clock
            calls.clear();patches.clear();monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE',role)
            result=e.evaluate(str(tmp_path/'candidate'))
            assert result[0]==100 and result[3]['beats_reference']==expected
            assert [c[2:] for c in calls]==[(names,count)]
            assert calls[0][0]!=calls[0][1] and len(patches)==1
            assert result[3]['workload_count']==len(names)
            assert result[3]['candidate_estimator_seconds_mean']==candidate_clock/1e9
            detail=e.unpack_evidence(result[3]['operator_evidence'])
            assert detail['workload_manifest']==list(names) and detail['interleaved_pairs']==count
            assert detail['verification_mode']==('final' if role=='final' else 'development')

def test_final_each_cell_failure_blocks_compatibility_but_keeps_score(monkeypatch, synthetic_calibration):
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE','final')
    bands=e.formal_record()['frozen']['bands']
    for w in e.FINAL_SUITE:
        for m in e.METRICS:
            candidate=measured(0.0,FAST);reference=measured(0.0,SLOW)
            candidate[w][m]=bands[w+'/'+m]+1.0
            result=e.summarize(candidate,reference,e.FINAL_SUITE,{})
            assert 0<result[0]<100 and result[3]['beats_reference']==0
    with pytest.raises(ValueError,match='manifest'):
        short=measured(0.0,FAST,e.DEV_SUITE)
        e.summarize(short,short,e.DEV_SUITE,{})
