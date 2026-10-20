"""Adversarial submissions, run through the real evaluator.

These compile and execute code, so they belong to the measured-evidence pass,
not the source-phase unit tests:

    python3 -m pytest -q -p no:cacheprovider tests/maintenance/edlib_exact_alignment/test_adversarial.py

Each fixture in adversarial/ is the reference body with one attack wrapped
around it. Every fixture is evaluated on one 10,000-base and one 30,000-base
workload with the unchanged evaluator code path; only the workload list, the
short-case correctness sweep and repeated compilation are trimmed. For the
two attacks that depend on files, a control without the guard shows that the
attack itself works. That control really does create files and System V IPC
objects, so run this file only inside a disposable verifier container.
"""
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
JUDGE_ROOT = REPOSITORY_ROOT / 'tasks/frontier-cs-2.0-demo/problems/Bioinformatics/edlib_exact_alignment/judge'
sys.path.insert(0, str(JUDGE_ROOT))
# Generic judge module names are shared with other tasks' judges; load this
# task's own whatever the interpreter imported first.
for _name in ('evaluator', 'native_cmake', 'cmake_entrypoint', 'core_measurement', 'elf_surface',
              'candidate_failure', 'submission_patch', 'judge', 'judge.workloads'):
    sys.modules.pop(_name, None)

import evaluator as e
import native_cmake

FIXTURES=Path(__file__).resolve().parent/'adversarial'
WORKLOADS=('public-10000-99-NW','public-30000-90-HW')

@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(e,'rows',lambda final=False:[w for w in REAL_ROWS(final) if w['id'] in WORKLOADS])
    monkeypatch.setattr(e,'correctness',lambda *a,**k:None)
    monkeypatch.setattr(e,'load_qualification',lambda:{'execution_admitted':True,'reference_qualification_complete':True})
    receipts=[]
    monkeypatch.setattr(e,'PROCESS_RECEIPTS',receipts)
    return receipts

REAL_ROWS=e.rows

def assert_wrapper_storage(metrics):
    """A fixture wraps the reference body: a few frames and one static counter more, never less."""
    candidate=metrics['core_peak_live_bytes_candidate_by_workload'];reference=metrics['core_peak_live_bytes_reference_by_workload']
    assert all(reference[w]<=candidate[w]<=reference[w]+8192 for w in reference),(candidate,reference)

def stray_files():
    found=[]
    for directory in ('/tmp','/dev/shm','/var/tmp'):
        if os.path.isdir(directory):
            found+=[str(p) for p in Path(directory).glob('edlib-fx-*')]+[str(p) for p in Path(directory).glob('edlib-guard-state*')]
    return found

def outcome(name):
    try:
        payload = e.ROOT/'reference.patch' if name == '../reference' else FIXTURES/(name+'.patch')
        score,_,message,metrics=e._evaluate(payload,False)
    except e.IncompleteEvaluation as failure:
        return 'rejected',failure.private_execution_evidence['message']
    # A static rejection of a development submission is a scored zero with one fixed category.
    if set(metrics)=={'candidate_diagnostic'}:
        assert score==0 and message==metrics['candidate_diagnostic']['message']
        return 'rejected',metrics['candidate_diagnostic']['category']
    return 'completed',metrics

def test_reference_completes_and_is_never_better_than_itself_on_storage(small):
    kind,metrics=outcome('../reference')
    assert kind=='completed'
    assert metrics['core_peak_live_bytes_candidate_by_workload']==metrics['core_peak_live_bytes_reference_by_workload']
    assert all(set(row['metrics'])=={'cpu_time_seconds','core_peak_live_bytes'} for row in metrics['beat_spec_workload_passes'])
    assert all(row['metrics']['core_peak_live_bytes'] for row in metrics['beat_spec_workload_passes'])
    assert metrics['peak_rss_bytes_gated'] is False and len(metrics['peak_rss_bytes_candidate_by_workload'])==len(WORKLOADS)
    # Every native execution ran with the fixed environment, in the read-only directory.
    assert not stray_files()

def test_state_cache_cannot_store_or_reuse_a_path(small):
    """(a) Writes the path to a file in one process and returns it from the next."""
    kind,metrics=outcome('state_cache')
    # A reused path announces itself on stderr, which the exact output parser rejects;
    # completing therefore means no process ever found state from an earlier one.
    assert kind=='completed',metrics
    assert not stray_files()
    assert_wrapper_storage(metrics)

def test_mode_detection_has_nothing_to_detect(small):
    """(b) Returns a non-path whenever the command line contains the silent flag."""
    kind,metrics=outcome('mode_detect')
    assert kind=='completed',metrics
    argv=[r['argv'] for r in small if r['argv'][0].endswith('edlib-aligner')]
    path=['-p','-f','CIG_EXT','-r','1','-k','-1'];location=['-l','-r','1','-k','-1']
    assert argv and all('-s' not in a and a[1]=='-m' and a[3:-2] in (path,location) for a in argv)
    # Timed, untimed and instrumented path executions are one command line; the
    # only other command line is the location-only one, in which no submitted code runs.
    assert any(a[3:-2]==path for a in argv) and any(a[3:-2]==location for a in argv)

def test_wrong_path_in_the_printed_mode_is_rejected(small):
    """(b) The mirror image: skips the work in the only mode that is now timed."""
    assert outcome('wrong_path_when_printed')==('rejected','path is missing or nonoptimal')

def test_screen_evasion_reaches_no_writable_object(small):
    """(c) (fopen)(...), a function pointer and asm-labelled raw syscalls."""
    kind,metrics=outcome('screen_evasion')
    assert kind=='completed',metrics       # any channel that opened would have printed
    assert not stray_files()

def test_forged_small_time_gains_nothing(small):
    """Correct path, 50 ms of extra real cost, then a complete output with a 1 ns
    timer line and exit before the CLI prints its own."""
    kind,metrics=outcome('forged_time_small')
    assert kind=='completed',metrics
    candidate=metrics['cpu_time_seconds_candidate_by_workload'];reference=metrics['cpu_time_seconds_reference_by_workload']
    # The compared time is the verifier's: the real cost is all there.
    assert all(candidate[w]>reference[w]+0.045 for w in candidate)
    assert not any(row['metrics']['cpu_time_seconds'] for row in metrics['beat_spec_workload_passes'])
    assert metrics['beats_reference']==0 and metrics['margin']<0
    # The forged line was read, as a diagnostic only.
    assert all(x==1e-9 for w in candidate for x in metrics['self_reported_seconds_replicates_by_workload'][w]['C'])

def test_forged_large_time_is_rejected(small):
    assert outcome('forged_time_large')==('rejected','self-reported duration exceeds verifier-measured elapsed time')

def test_extra_output_is_rejected(small):
    assert outcome('extra_output')==('rejected','CLI output malformed')

def test_compared_time_is_the_verifiers_and_bounds_the_self_report(small):
    kind,metrics=outcome('../reference')
    assert kind=='completed'
    for w,lanes in metrics['cpu_time_seconds_replicates_by_workload'].items():
        for lane in 'CR':
            own=metrics['self_reported_seconds_replicates_by_workload'][w][lane]
            assert len(lanes[lane])==len(own)==e.TIMING_ROUNDS and all(0<a<b for a,b in zip(own,lanes[lane]))
            located=metrics['location_seconds_replicates_by_workload'][w][lane]
            assert len(located)==e.TIMING_ROUNDS and all(x>0 for x in located)
            compared=metrics['cpu_time_seconds_candidate_by_workload' if lane=='C' else 'cpu_time_seconds_reference_by_workload'][w]
            assert compared==min(lanes[lane])-min(located)
    native=[r for r in small if 'elapsed_seconds' in r and r['argv'][0].endswith('edlib-aligner')]
    assert native and all(0<r['elapsed_seconds']<=r['wall_seconds'] for r in native)

@pytest.mark.parametrize('execution',range(1+2*e.TIMING_ROUNDS+5))
def test_every_candidate_execution_is_validated(small,monkeypatch,execution):
    """Corrupt the path printed by exactly one candidate CLI execution.

    Execution 0 (untimed), the alternating timed path and location-only runs
    of every round, and the five instrumented ones of the first workload must
    each invalidate the evaluation. A location-only run is corrupted in its
    printed distance.
    """
    real=native_cmake.run_bounded;seen=[0]
    def tampering(args,cwd,deadline,**options):
        out,rss,elapsed=real(args,cwd,deadline,**options)
        candidate=isinstance(args[0],native_cmake.NativeExecutable) and tampering.candidate in str(args[0].source)
        if candidate:
            if seen[0]==execution:
                if 'Cigar:\n' in out:
                    head,tail=out.split('Cigar:\n');cigar,rest=tail.split('\n',1)
                    out=head+'Cigar:\n1X'+cigar+'\n'+rest
                else:
                    head,tail=out.split('\n#0: ');distance,rest=tail.split(' ',1)
                    out=head+'\n#0: '+str(int(distance)+1)+' '+rest
            seen[0]+=1
        return out,rss,elapsed
    monkeypatch.setattr(e,'run_bounded',tampering)
    applied=e.apply_submission_patch
    def remember(path,root,config):
        result=applied(path,root,config)
        if 'candidate' not in tampering.__dict__: tampering.candidate=str(result.workdir)
        return result
    monkeypatch.setattr(e,'apply_submission_patch',remember)
    with pytest.raises(e.IncompleteEvaluation) as failure:
        e._evaluate(e.ROOT/'reference.patch',False)
    assert failure.value.private_execution_evidence['message'] in ('path is missing or nonoptimal','path overrun','invalid path operation','CLI distance or location disagreement')
    assert seen[0]==execution+1

def _plain(binary,q,t,cwd):
    return subprocess.run([str(binary),'-m','NW','-p','-f','CIG_EXT','-r','1','-k','-1',str(q),str(t)],cwd=cwd,
                          capture_output=True,text=True)

def test_controls_show_the_attacks_work_without_the_guard(tmp_path,monkeypatch):
    """Without the guard the cache is reused and write channels open; with it, neither."""
    import time
    deadline=time.monotonic()+600
    row=[w for w in REAL_ROWS(False) if w['id']==WORKLOADS[0]][0]
    q,t,qp,tp=e.materialize(row,tmp_path)
    built={}
    for name in ('state_cache','screen_evasion'):
        result=e.apply_submission_patch(FIXTURES/(name+'.patch'),e.ROOT,e.CONFIG)
        assert result.ok
        built[name]=e.build(result.workdir,deadline)[0].artifact
    free=tmp_path/'unguarded';free.mkdir()
    first=_plain(built['state_cache'],qp,tp,free);second=_plain(built['state_cache'],qp,tp,free)
    assert 'state reused' not in first.stderr and 'state reused' in second.stderr
    assert list(free.glob('edlib-fx-*'))
    for stray in stray_files(): os.unlink(stray)
    opened=_plain(built['screen_evasion'],qp,tp,free).stderr
    assert 'channel open: (fopen)' in opened and 'channel open: raw openat' in opened and 'channel open: mkdir' in opened
    for stray in list(free.iterdir())+[Path(p) for p in stray_files()]:
        stray.rmdir() if stray.is_dir() else stray.unlink()
    # The same binaries under the guard: twice each, no reuse, nothing opened, nothing left behind.
    monkeypatch.setattr(e,'PROCESS_RECEIPTS',None)
    work=tmp_path/'guarded';work.mkdir()
    e.prepare_native_guard(work,deadline)
    try:
        for name in built:
            for _ in range(2):
                out,_,_=e.run_native([built[name],'-m','NW','-p','-f','CIG_EXT','-r','1','-k','-1',qp,tp],deadline)
                assert 'EDLIB_FIXTURE' not in out
                e.parse_cli_run(out,'NW',q,t)
        assert not stray_files() and not any(e.NATIVE_CWD.iterdir())
    finally:
        e.NATIVE_GUARD=e.NATIVE_CWD=None

def test_guard_self_test_reports_every_probe_without_the_guard(tmp_path):
    """The self-test is meaningful: unguarded, the same binary reports open channels."""
    import time
    guard,selftest=native_cmake.build_native_guard(tmp_path,time.monotonic()+120)
    canary=tmp_path/'canary';canary.write_text('x\n')
    free=tmp_path/'cwd';free.mkdir()
    out=subprocess.run([str(selftest),str(canary)],cwd=free,capture_output=True,text=True).stdout
    assert out!=e.GUARD_SELFTEST_EXPECTED
    for probe in ('fopen_write_cwd','fstat','shmget','socket_unix','kill_parent_probe'): assert ' '+probe+' ' in out+' '

# ---- the location-only run and the storage metric ----

def test_location_slowdown_outside_the_function_is_rejected(small):
    """The body closes its function and adds a constructor that burns time when -l is given."""
    assert outcome('location_slowdown_scope')==('rejected','body_scope')

def test_location_slowdown_through_the_constructor_table_is_rejected(small):
    """The same constructor without leaving the function, through a section attribute."""
    assert outcome('location_slowdown_section')==('rejected','launch_surface')

def _wall(binary,flags,q,t,cwd):
    import time
    best=None
    for _ in range(5):
        started=time.monotonic()
        done=subprocess.run([str(binary),'-m','NW',*flags,'-r','1','-k','-1',str(q),str(t)],cwd=cwd,capture_output=True)
        elapsed=time.monotonic()-started
        assert done.returncode==0
        best=elapsed if best is None else min(best,elapsed)
    return best

def test_control_the_constructor_table_slowdown_works_when_unchecked(tmp_path):
    """Built without the launch-surface check, the fixture inflates only its own location-only run."""
    import time
    deadline=time.monotonic()+600
    row=[w for w in REAL_ROWS(False) if w['id']==WORKLOADS[0]][0]
    q,t,qp,tp=e.materialize(row,tmp_path)
    built={}
    for name,payload in (('fixture',FIXTURES/'location_slowdown_section.patch'),('reference',e.ROOT/'reference.patch')):
        result=e.apply_submission_patch(payload,e.ROOT,e.CONFIG)
        assert result.ok
        built[name]=e.build(result.workdir,deadline)[0]
    with pytest.raises(e.Invalid): e.check_launch_surface(built['fixture'],built['reference'])
    slow=_wall(built['fixture'].artifact,['-l'],qp,tp,tmp_path)-_wall(built['reference'].artifact,['-l'],qp,tp,tmp_path)
    same=_wall(built['fixture'].artifact,['-p','-f','CIG_EXT'],qp,tp,tmp_path)-_wall(built['reference'].artifact,['-p','-f','CIG_EXT'],qp,tp,tmp_path)
    assert slow>0.002 and abs(same)<slow/2

@pytest.mark.parametrize('name',['static_storage','stack_storage'])
def test_storage_moved_off_the_heap_is_still_counted(small,name):
    """The work arrays of one code path, as static or as automatic arrays of fixed size."""
    kind,metrics=outcome(name)
    assert kind=='completed',metrics
    candidate=metrics['core_peak_live_bytes_candidate_by_workload'];reference=metrics['core_peak_live_bytes_reference_by_workload']
    used=WORKLOADS[0]                      # the workload that takes that code path
    assert candidate[used]>reference[used]
    assert not [row for row in metrics['beat_spec_workload_passes'] if row['workload']==used][0]['metrics']['core_peak_live_bytes']
    if name=='static_storage':
        static=metrics['static_storage_bytes']
        assert 0<=static['candidate']-static['reference']-2*1027*(17+32+2)<=64      # the three arrays, plus alignment
        assert all(candidate[w]>reference[w] for w in reference)
    assert metrics['beats_reference']==0

def test_mapped_storage_is_counted_like_heap(small):
    kind,metrics=outcome('mapped_storage')
    assert kind=='completed',metrics
    candidate=metrics['core_peak_live_bytes_candidate_by_workload'];reference=metrics['core_peak_live_bytes_reference_by_workload']
    # The mapped bytes replace the same number of heap bytes; only a few call frames differ.
    assert all(abs(candidate[w]-reference[w])<=2048 for w in reference),(candidate,reference)

def test_storage_of_the_reference_is_deterministic(small):
    kind,metrics=outcome('../reference')
    assert kind=='completed'
    for lanes in metrics['core_peak_live_bytes_replicates_by_workload'].values():
        assert len(set(lanes['C']+lanes['R']))==1
