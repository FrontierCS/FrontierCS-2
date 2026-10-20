"""Train feedback: what crosses the judge boundary, and nothing else. No compilation."""
import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import json
from pathlib import Path
import sys

import pytest

TASK = (Path(__file__).resolve().parents[3] / 'tasks/frontier-cs-2.0-demo'
        / 'problems/Bioinformatics/edlib_exact_alignment')
sys.path.insert(0, str(TASK / 'judge'))
# Generic judge module names are shared with other tasks' judges; load this
# task's own whatever the interpreter imported first.
for _name in ('evaluator', 'native_cmake', 'cmake_entrypoint', 'core_measurement', 'elf_surface',
              'candidate_failure', 'submission_patch', 'judge', 'judge.workloads'):
    sys.modules.pop(_name, None)
import evaluator as e

# Loaded by path: the verifier directory must not shadow other tasks' modules.
_spec = importlib.util.spec_from_file_location('edlib_beat_spec_feedback', TASK / 'verifier/beat_spec_feedback.py')
feedback = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feedback)


def evaluator_metrics():
    """A private train metrics dict of the shape evaluator._evaluate returns."""
    workloads = e.rows(False)
    time_c = {w['id']: 0.001 + i * 1e-5 for i, w in enumerate(workloads)}
    time_r = {w['id']: 0.00115 for w in workloads}
    storage_c = {w['id']: 40000 + 100 * i for i, w in enumerate(workloads)}
    storage_r = {w['id']: 41000 for w in workloads}
    decisions = e.audit(workloads, True, True, time_c, time_r, storage_c, storage_r)
    return {'beats_reference': int(all(all(x['metrics'].values()) for x in decisions)),
            'beat_spec_workload_passes': decisions, 'reference_score': 60.0, 'margin': 1.0,
            'cpu_time_seconds_candidate_by_workload': time_c, 'cpu_time_seconds_reference_by_workload': time_r,
            'core_peak_live_bytes_candidate_by_workload': storage_c, 'core_peak_live_bytes_reference_by_workload': storage_r,
            'location_seconds_replicates_by_workload': {}, 'path_seconds_median_candidate_by_workload': time_c,
            'static_storage_bytes': {'candidate': 8, 'reference': 8}, 'timing_rounds': e.TIMING_ROUNDS,
            'development_cpu_time_seconds_median': 0.01, 'development_peak_rss_bytes_max': 1}


def test_declarations_agree_and_nothing_beyond_progress_is_approved():
    spec = json.loads((TASK / 'judge/beat_spec.json').read_text())
    policy = json.loads((TASK / 'policy.json').read_text())
    assert [x['name'] for x in spec['metrics']] == ['cpu_time_seconds', 'core_peak_live_bytes']
    # The one declared field is the categorical rejection, never a measurement.
    assert [x['name'] for x in spec['solver_feedback']] == ['candidate_diagnostic']
    assert policy['metadata']['evaluation']['solver_visible_metrics'] == ['candidate_diagnostic']
    assert e.CONFIG['evaluation']['solver_visible_metrics'] == ['candidate_diagnostic']
    statement = ' '.join((TASK / 'agent-base/frontier/task/repo/README.md').read_text().split())
    for part in ('`beat_spec_progress`', '- `cpu_time_seconds`:', '- `core_peak_live_bytes`:',
                 'the number of workloads on which the metric clears, the number on which it does not, the total and the two percentages'):
        assert part in statement, part


def test_train_feedback_is_the_per_metric_counts_under_the_two_metric_names():
    metrics = evaluator_metrics()
    visible = feedback.public_metrics(metrics, beat_spec_path=TASK / 'judge/beat_spec.json',
                                      configured_visible=['candidate_diagnostic'])
    assert set(visible) == {'beat_spec_progress'}
    progress = visible['beat_spec_progress']
    assert set(progress) == {'cpu_time_seconds', 'core_peak_live_bytes'}
    for column in progress.values():
        assert set(column) == {'passed', 'failed', 'total', 'pass_percent', 'fail_percent'}
        assert column['total'] == 30 and column['passed'] + column['failed'] == 30
    assert progress['cpu_time_seconds']['passed'] == 15 and progress['core_peak_live_bytes']['passed'] == 11
    text = json.dumps(visible)
    for row in metrics['beat_spec_workload_passes']:
        assert row['workload'] not in text
    assert 'reference' not in text and '0.00115' not in text and '41000' not in text


def test_a_candidate_difference_that_is_not_positive_counts_as_not_cleared():
    metrics = evaluator_metrics()
    workloads = e.rows(False)
    time_c = dict(metrics['cpu_time_seconds_candidate_by_workload'])
    first = workloads[0]['id']
    time_c[first] = -0.0001
    metrics['beat_spec_workload_passes'] = e.audit(
        workloads, True, True, time_c, metrics['cpu_time_seconds_reference_by_workload'],
        metrics['core_peak_live_bytes_candidate_by_workload'], metrics['core_peak_live_bytes_reference_by_workload'])
    visible = feedback.public_metrics(metrics, beat_spec_path=TASK / 'judge/beat_spec.json')
    assert visible['beat_spec_progress']['cpu_time_seconds']['passed'] == 14


def test_a_name_that_is_not_declared_fails_closed():
    with pytest.raises(ValueError):
        feedback.public_metrics(evaluator_metrics(), beat_spec_path=TASK / 'judge/beat_spec.json',
                                configured_visible=['cpu_time_seconds_candidate_by_workload'])


def test_the_verifier_admits_only_the_pinned_rejection_table():
    """The shipped adapter loads judge/candidate-diagnostic.json and rebuilds the object from its table."""
    saved_path, saved_module = list(sys.path), sys.modules.get('beat_spec_feedback')
    sys.path.insert(0, str(TASK / 'verifier'))
    try:
        spec = importlib.util.spec_from_file_location('edlib_fcs2_verifier', TASK / 'verifier/fcs2_verifier.py')
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
    finally:
        sys.path[:] = saved_path
        if saved_module is None:
            sys.modules.pop('beat_spec_feedback', None)
        else:
            sys.modules['beat_spec_feedback'] = saved_module
    admit = verifier.load_candidate_diagnostic(TASK / 'judge')
    assert admit is not None
    import candidate_failure
    for category in candidate_failure.CATEGORIES:
        visible = admit(candidate_failure.diagnostic(category))
        assert set(visible) == {'schema', 'category', 'message'} and visible['category'] == category
        assert visible['message'].startswith('Rejected: ') and '\n' not in visible['message']
    with pytest.raises(ValueError):
        admit({'schema': candidate_failure.SCHEMA, 'category': 'outside_after_body', 'message': 'Rejected: }'})
