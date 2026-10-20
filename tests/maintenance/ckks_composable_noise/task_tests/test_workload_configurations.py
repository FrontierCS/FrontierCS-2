"""Source/configuration checks only. Fixtures are not reference measurements."""
import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]/'tasks/frontier-cs-2.0-demo/problems/Crypto/ckks_composable_noise/judge'))
# Generic judge module names are shared with other tasks; load this task's own
# whatever the interpreter imported first.
for _name in ('evaluator', 'formal_scoring', 'workload_configurations', 'fixed_surface', 'submission_patch',
              'go_entrypoint', 'child_environment', 'calibration', 'ckks_evidence_transport'):
    sys.modules.pop(_name, None)
import evaluator as e
import workload_configurations as c


def literal(row):
    return {'LogN': 12, 'LogDefaultScale': row['log_scale'],
            'Xs': {'Type': 'Ternary', **({'H': 192} if row['secret'] == 'h192' else {'P': 2/3})},
            'Q': [2**v+1 for v in row['log_q']], 'P': [2**v+1 for v in row['log_p']]}


def test_table1_axes_and_predecessor_membership():
    rows = c.configurations(); dev, final = c.manifests()
    assert len(rows) == 29 and final[:9] == list(c.ORIGINAL_FINAL)
    assert dev[:6] == list(c.ORIGINAL_DEV) and len(dev) == 21
    for circuit in c.ORIGINAL_FINAL[1:]:
        expected = {(s, h) for s in ((45,) if circuit in ('mod1', 'bootstrapping') else (45, 55))
                    for h in ('dense', 'h192')}
        assert {(r['log_scale'], r['secret']) for r in rows if r['circuit'] == circuit} == expected
    assert len([r for r in rows if r['paper_set']]) == 28
    assert 2*37*len(final) == 2146 and 5*2*(len(dev)+len(final)) == 500


@pytest.mark.parametrize('row', c.configurations(), ids=lambda r: r['id'])
def test_effective_main_changes_only_declared_parameters(row):
    source = (e.TASK_DIR/'harbor/app'/row['main_path']).read_text()
    derived = c.configured_main(source, row)
    if row['original_row']:
        assert derived == source
    else:
        assert derived != source
        assert f"LogScale := {row['log_scale']}" in derived
        assert derived.count('"github.com/tuneinsight/lattigo/v6/ring"') == 1
    def core(body):
        body = re.sub(r'(?m)^\s*(?://)?Xs:[^\n]+', '', body)
        body = re.sub(r'LogScale := \d+', '', body)
        body = re.sub(r'LogQ:\s*\[\]int\{[^}]+\}', '', body)
        body = body.replace('"github.com/tuneinsight/lattigo/v6/ring"', '')
        return re.sub(r'\s+', '', body)
    assert core(derived) == core(source)
    assert 'LogN := 12' in derived
    observed = literal(row)
    values = [{'ResidualParameters': observed}] if row['circuit'] == 'bootstrapping' else [observed]
    assert c.verify_parameters(values, row)


@pytest.mark.parametrize('circuit,scale,secret', [
    ('mod1', 55, 'dense'), ('bootstrapping', 55, 'h192'),
    ('power_basis', 54, 'dense'), ('power_basis', True, 'dense'),
    ('power_basis', 45, 'h32'), ('mul_relin_rescale', 45, 'h192'), ('missing', 45, 'dense')])
def test_unapproved_configuration_excluded(circuit, scale, secret):
    with pytest.raises(ValueError): c.configuration(circuit, scale, secret)


@pytest.mark.parametrize('mutation', [
    lambda d: d['rows'].pop(), lambda d: d['rows'].append(d['rows'][0]),
    lambda d: d['rows'].reverse(), lambda d: d['rows'][10].update(log_n=16),
    lambda d: d['rows'][10].update(log_scale=55 if d['rows'][10]['log_scale']==45 else 45)])
def test_manifest_rejects_missing_duplicate_reordered_or_changed_rows(mutation):
    document = json.loads((e.TASK_DIR/'judge/workload_manifest.json').read_text())
    mutation(document)
    with pytest.raises(ValueError): c.validate_manifest(document, e.RELEASE)


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(LogN=16), lambda p: p.update(LogDefaultScale=55),
    lambda p: p.update(Xs={'Type':'Ternary','H':32}),
    lambda p: p['Q'].pop(), lambda p: p['P'].__setitem__(0, 2**59+1)])
def test_resolved_runtime_configuration_must_match(mutation):
    row = c.configuration('coeffs_to_slots', 45, 'h192'); p = literal(row); mutation(p)
    with pytest.raises(ValueError): c.verify_parameters([p], row)


def test_changed_main_and_forged_row_rejected():
    row = c.configuration('poly_eval', 45, 'h192')
    source = (e.TASK_DIR/'harbor/app'/row['main_path']).read_text()
    with pytest.raises(ValueError, match='source changed'): c.configured_main(source+'\n', row)
    with pytest.raises(ValueError, match='row'): c.configured_main(source, {**row, 'log_n': 16})


def test_evaluator_routes_added_row_to_base_circuit_and_checks_actual_parameters(tmp_path, monkeypatch):
    row = c.configuration('coeffs_to_slots', 45, 'h192'); calls=[]
    def invoke(tree, circuit, overlay, deadline):
        calls.append(circuit)
        replacements=json.loads(overlay.read_text())['Replace']
        assert str(tree/row['main_path']) in replacements
        assert 'LogScale := 45' in Path(replacements[str(tree/row['main_path'])]).read_text()
        return 'synthetic', {'status':'ok', 'main_completed':1, 'launch_attempts':1}
    def parse(output, prefix, circuit):
        assert circuit == 'coeffs_to_slots'  # preserves two-output Stats aggregation
        return {**{metric:0.0 for metric in e.METRICS}, 'resolved_parameters':[literal(row)]}
    monkeypatch.setattr(e,'invoke_original',invoke);monkeypatch.setattr(e,'parse_observations',parse)
    tree=e.TASK_DIR/'harbor/app'
    result, receipt=e.measure_suite(tree,[row['id']],tmp_path/'valid',10)
    assert list(result)==[row['id']] and calls==['coeffs_to_slots']
    assert receipt['observations'][0]['configuration_binding']['configuration']==row
    original=parse
    def wrong(*args):
        value=original(*args);value['resolved_parameters'][0]['LogDefaultScale']=55;return value
    monkeypatch.setattr(e,'parse_observations',wrong)
    result, receipt=e.measure_suite(tree,[row['id']],tmp_path/'invalid',10)
    assert result=={} and receipt['observations'][0]['status']=='invalid'
    assert receipt['observations'][0]['main_completed']==1


def test_unknown_suite_rejected_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(e,'observation_prefix',lambda *a,**k:pytest.fail('must not start measurement'))
    for names in [('missing',), (e.FINAL_SUITE[0], e.FINAL_SUITE[0])]:
        with pytest.raises(e.EvaluationError):e.measure_suite(tmp_path,names,tmp_path/'output',10)


def test_public_manifest_and_clock_remain_exact():
    task=e.TASK_DIR.parent
    assert (task/'agent-base/frontier/task/workload_manifest.json').read_bytes()==(e.TASK_DIR/'judge/workload_manifest.json').read_bytes()
    assert e.CONFIG['evaluation']['answering_timeout_seconds']==18000
    assert e.CONFIG['evaluation']['max_dev_submissions']==100
    assert e.CONFIG['evaluation']['max_train_wall_seconds']==3600
    assert e.CONFIG['runtime']['timeout_seconds']==600
    spec=json.loads((e.TASK_DIR/'beat_spec.json').read_text())
    assert {m['name'] for m in spec['metrics']}==set(e.METRICS)
    assert all(m['better']=='lower' and 'all58cells' in m['tolerance'] for m in spec['metrics'])
    assert all(x['reveals_reference'].startswith('no') for x in spec['solver_feedback'])
    assert e.RELEASE['formal_calibration'] is None
