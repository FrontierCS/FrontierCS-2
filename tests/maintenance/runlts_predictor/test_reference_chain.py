"""Failure-path fixtures only: no compilation, trace execution or task score."""
import deployment_test_guard  # noqa: F401  # isort: skip

import errno
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
from tests.maintenance.runlts_predictor.runtime_imports import MODULES
e = MODULES["evaluator"]
adapter = MODULES["fcs2_verifier"]


@pytest.fixture
def reference_fault(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'ROOT', tmp_path)
    trees = [tmp_path/'candidate', tmp_path/'reference']
    for tree in trees:
        tree.mkdir()
    outputs = iter(SimpleNamespace(ok=True, workdir=t, cleanup_root=t) for t in trees)
    monkeypatch.setattr(e, 'apply_submission_patch', lambda *a: next(outputs))
    monkeypatch.setattr(e, 'run_command', lambda *a: '{"valid":true,"logical_bits":1}')
    monkeypatch.setattr(e, 'manifest', lambda *a: [{'id':'fixture-only'}])
    # These faults are in the live reference, which only the audit mode executes.
    monkeypatch.setattr(e, 'pinned_reference', lambda *a: {'fixture-only': None})
    monkeypatch.setenv(e.REFERENCE_AUDIT_ENV, 'live')
    monkeypatch.setattr(e, 'build', lambda *a: None)
    monkeypatch.setattr(e, 'grade', lambda *a: pytest.fail('invalid execution reached grading'))
    return trees


def install_fault(monkeypatch, phase, kind):
    if kind == 'oserror':
        cause = PermissionError(errno.EACCES, 'PRIVATE_MESSAGE', '/PRIVATE_PATH')
    elif kind == 'parser':
        cause = e.CandidateError('incomplete or substituted trace execution')
    else:
        cause = e.CommandFailure('process_exit', elapsed_seconds=.1,
            timeout_seconds=1200, returncode=7, output_tail='PRIVATE_COMMAND_OUTPUT')
    cause.__cause__ = ValueError('PRIVATE_NESTED_CAUSE')
    def fail(*args):
        local_secret = 'PRIVATE_LOCAL_VARIABLE'
        raise cause
    monkeypatch.setattr(e, 'build' if phase == 'build' else 'run_trace', fail)
    return cause


@pytest.mark.parametrize('phase', ['build','execution'])
@pytest.mark.parametrize('kind', ['oserror','parser','command'])
def test_original_reference_exception_chain_survives(reference_fault, monkeypatch, phase, kind):
    cause = install_fault(monkeypatch, phase, kind)
    with pytest.raises(e.EvaluationError) as caught:
        e.evaluate('fixture.patch')
    failure = caught.value
    assert failure.__cause__ is cause
    assert str(failure) == f'private reference {phase} failed'
    evidence = failure.private_execution_evidence
    chain = evidence['exception_chain']['exceptions']
    assert [x['error_type'] for x in chain] == [type(cause).__name__, 'ValueError']
    assert chain[0]['frames'][-1]['function'] == 'fail'
    if kind == 'oserror':
        assert chain[0]['errno'] == errno.EACCES
    if kind == 'command':
        assert evidence['command_failure']['returncode'] == 7
    assert 'PRIVATE' not in json.dumps(evidence['exception_chain'])
    assert not any(p.exists() for p in reference_fault), 'temporary source trees must be cleaned'


@pytest.mark.parametrize('phase', ['build','execution'])
def test_actual_adapter_keeps_private_cause_out_of_public_result(reference_fault, monkeypatch, tmp_path, capsys, phase):
    zstandard = pytest.importorskip('zstandard')
    install_fault(monkeypatch, phase, 'command')
    payload=tmp_path/'payload.tar';payload.write_bytes(b'fixture-only')
    result_path=tmp_path/'result.json';evidence_path=tmp_path/'evidence.tar.zst'
    scratch=tmp_path/'adapter-scratch';scratch.mkdir()
    monkeypatch.setenv('FRONTIER_VERIFIER_SCRATCH', str(scratch))
    for key,value in {'FRONTIER_RESULT_PATH':result_path,'FRONTIER_EVIDENCE_PATH':evidence_path,
                      'FRONTIER_PAYLOAD_PATH':payload,'FRONTIER_DATASET_PATH':tmp_path/'dataset.json'}.items():
        monkeypatch.setenv(key,str(value))
    monkeypatch.setenv('FRONTIER_VERIFICATION_MODE','train')
    monkeypatch.setattr(adapter,'extract_payload',lambda *a: None)
    monkeypatch.setattr(adapter,'entrypoint_path',lambda *a: tmp_path/'solution.patch')
    monkeypatch.setattr(adapter,'stage_task',lambda *a: tmp_path)
    monkeypatch.setattr(adapter,'load_evaluator',lambda *a: e.evaluate)
    assert adapter.main(tmp_path,'payload/solution.patch') == 0
    result=json.loads(result_path.read_text())
    assert result['status']=='failed' and 'raw_score' not in result
    assert result['metrics']=={} and 'private_metrics' not in result
    out=capsys.readouterr()
    assert 'PRIVATE' not in result_path.read_text()+out.out+out.err
    raw=zstandard.ZstdDecompressor().decompress(evidence_path.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        document=json.loads(archive.extractfile('metrics.json').read())
    assert document['execution_valid'] is False
    private=document['metrics']['private_execution']
    assert private['reference_phase']==phase
    assert private['command_failure']['output_tail']=='PRIVATE_COMMAND_OUTPUT'
    assert private['exception_chain']['exceptions'][1]['error_type']=='ValueError'
    assert not list(scratch.iterdir())


def test_private_cause_metadata_is_bounded_without_messages_paths_or_locals():
    error=PermissionError(errno.ENOENT,'PRIVATE_MESSAGE','/PRIVATE_PATH')
    for _ in range(20):
        parent=RuntimeError('PRIVATE_CAUSE');parent.__cause__=error;error=parent
    report=e.private_exception_chain(error)
    assert len(report['exceptions'])==8 and report['truncated'] is True
    assert 'PRIVATE' not in json.dumps(report)
    cycle=RuntimeError('PRIVATE_CYCLE');cycle.__cause__=cycle
    report=e.private_exception_chain(cycle)
    assert len(report['exceptions'])==1 and report['truncated'] is True
