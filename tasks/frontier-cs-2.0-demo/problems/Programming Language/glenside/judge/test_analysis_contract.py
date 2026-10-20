from pathlib import Path

import pytest

import evaluator


def test_contract_compares_candidate_facts(monkeypatch):
    expected = [{'case': str(index), 'fact': index} for index in range(34)]
    actual = [*expected[:-1], {'case': '33', 'fact': 'incorrect'}]
    calls = []

    def invoke(binary, payload):
        calls.append((binary, payload))
        return expected if binary == Path('trusted') else actual

    monkeypatch.setattr(evaluator, '_invoke', invoke)
    with pytest.raises(ValueError, match='^candidate analysis contract failed$'):
        evaluator._check_analysis_contract(Path('candidate'), Path('trusted'))
    assert [binary for binary, _ in calls] == [Path('trusted'), Path('candidate')]
    actual[:] = expected
    evaluator._check_analysis_contract(Path('candidate'), Path('trusted'))


def test_contract_failure_hides_private_details(monkeypatch):
    def invoke(binary, payload):
        if binary == Path('trusted'):
            return [{'case': str(index)} for index in range(34)]
        raise RuntimeError('private expected fact')

    monkeypatch.setattr(evaluator, '_invoke', invoke)
    with pytest.raises(RuntimeError, match='^candidate analysis invocation failed$'):
        evaluator._check_analysis_contract(Path('candidate'), Path('trusted'))
