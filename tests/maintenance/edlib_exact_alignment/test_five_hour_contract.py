import deployment_test_guard  # noqa: F401
import sys
from pathlib import Path
TASK = Path(__file__).resolve().parents[3] / "tasks/frontier-cs-2.0-demo/problems/Bioinformatics/edlib_exact_alignment"
sys.path.insert(0, str(TASK / "judge"))
"""Affected resource and workload contracts; not scientific measurements."""
import json
from pathlib import Path
import evaluator as e

def test_changed_workload_full_membership():
    public=e.rows(False); final=e.rows(True)
    expected={(n,s,m) for n in (10000,30000) for s in (60,90,99) for m in ('NW','SHW','HW')}
    assert len(public)==18 and len(final)==36
    assert {x['id'] for x in public} < {x['id'] for x in final}
    for split in ('public','heldout'):
        selected=[x for x in final if x['split']==split]
        assert len(selected)==18
        assert {(x['length'],x['similarity'],x['mode']) for x in selected}==expected
    visible=json.loads((e.ROOT/'harbor/app/challenge_data/public_workloads.json').read_text())
    assert visible['workloads']==public
    assert all(x['split']=='public' for x in visible['workloads'])

def test_supported_budget_projection():
    from autobench.pipeline.fcs2 import Budgets
    b=Budgets.derive(e.CONFIG,agent_work_seconds=14400)
    assert b.wall_time_seconds==18000
    assert b.max_train_calls==20
    assert b.max_train_wall_seconds==12000
    assert b.eval_timeout_seconds==600
    assert b.max_train_calls*b.eval_timeout_seconds==b.max_train_wall_seconds

def test_changed_fixture_hashes():
    from judge.workloads import generate,digest
    for row in e.rows(True):
        q,t=generate(row)
        assert (digest(q),digest(t))==(row['query_sha256'],row['target_sha256'])

def test_measurement_never_sets_qualification(monkeypatch):
    monkeypatch.setattr(e,'load_qualification',lambda:{'execution_admitted':False,'reference_qualification_complete':False})
    before=(e.ROOT/'qualification.json').read_bytes()
    def refuse(*args,**kwargs):
        raise e.Invalid('diagnostic deliberate stop before build')
    monkeypatch.setattr(e,'apply_submission_patch',refuse)
    result=e.measure_reference(final=True)
    assert result[0]==0
    assert result[3]['reference_qualification_complete'] is False
    assert result[3]['beats_reference']==0
    assert len(result[3]['beat_spec_workload_passes'])==36
    assert (e.ROOT/'qualification.json').read_bytes()==before
    assert 'operator_exception' in result[3]
    # The submission entrypoint cannot select the operator-only measurement path.
    result=e.evaluate('unread.patch')
    assert 'qualification pending' in result[2]
    assert 'operator_exception' not in result[3]

def test_runtime_admission_does_not_award_qualification(monkeypatch):
    monkeypatch.setattr(e,'load_qualification',lambda:{'execution_admitted':True,'reference_qualification_complete':False})
    def refuse(*args,**kwargs):raise e.Invalid('entered admitted evaluator')
    monkeypatch.setattr(e,'apply_submission_patch',refuse)
    result=e.evaluate('unused.patch')
    assert result[2]=='entered admitted evaluator'
    assert result[3]['reference_qualification_complete'] is False
    assert result[3]['beats_reference']==0
