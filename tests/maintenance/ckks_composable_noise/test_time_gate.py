"""Accuracy-and-time gate: parity, fixed surface, interface, forged output.

    python3 -m pytest -q -p no:cacheprovider tests/maintenance/ckks_composable_noise/test_time_gate.py

Pure-Python tests run everywhere. Tests marked GO compile and run Go programs;
they need go1.23.12 on PATH and the pinned module cache in GOMODCACHE (both are
in the verifier image) and are skipped without them. Each fixture in
adversarial/ is the reference estimator with one violation added; regenerate
them with make_fixtures.py and the frozen measured mains with
freeze_measured_mains.py, both run from the task's judge directory.
"""
import deployment_test_guard  # noqa: F401  # isort: skip

import json
from pathlib import Path
import shutil
import sys
import time

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
JUDGE_ROOT = REPOSITORY_ROOT/'tasks/frontier-cs-2.0-demo/problems/Crypto/ckks_composable_noise/judge'
# Judge modules have generic names shared with other tasks' judges. Drop them
# from the import cache before loading this task's own, whatever other task's
# maintenance tests the interpreter imported first, and again afterwards so
# tests imported later load theirs.
JUDGE_MODULES = ('evaluator', 'formal_scoring', 'workload_configurations', 'fixed_surface', 'submission_patch',
                 'go_entrypoint', 'child_environment', 'calibration', 'ckks_evidence_transport')
for _name in JUDGE_MODULES:
    sys.modules.pop(_name, None)
sys.path.insert(0, str(JUDGE_ROOT))
import evaluator as e
import formal_scoring as scoring
import workload_configurations as c
from fixed_surface import fixed_surface_violations
from submission_patch import apply_submission_patch, validate_patch

for _name in JUDGE_MODULES:
    sys.modules.pop(_name, None)


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

GO = pytest.mark.skipif(shutil.which('go') is None, reason='needs the Go toolchain of the verifier image')
ADVERSARIAL = Path(__file__).resolve().parent/'adversarial'
PRISTINE = e.TASK_DIR/'harbor/app'
FIXED_SURFACE = {'fixed_statistics': 'estimator/stats.go', 'fixed_test_vector': 'estimator/utils.go',
                 'fixed_experiment': 'estimator/experiments/mul_relin_rescale/mul_relin_rescale.go',
                 'fixed_experiment_added': 'estimator/experiments/extra/extra.go',
                 'fixed_module': 'estimator/go.mod'}
POLICY = ('deferred_goroutine', 'deferred_timer', 'deferred_finalizer', 'precompute_init',
          'precompute_package_initializer', 'fixed_clear_target', 'forbidden_import')
BUILD = ('forged_observation_hook', 'interface_break')
ACCEPTED = ('control_helper_package', 'forged_observation_print')


@pytest.fixture
def calibration(monkeypatch):
    cells = [w+'/'+m for w in e.FINAL_SUITE for m in e.METRICS]
    fixture = {'development_workloads': list(e.DEV_SUITE), 'frozen': {'cells': cells, 'bands': dict.fromkeys(cells, .25)}}
    monkeypatch.setattr(e, 'formal_record', lambda: fixture)
    monkeypatch.setitem(e.RELEASE, 'formal_calibration', {'sha256': 'unit-test-only'})
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE', 'final')
    return fixture


def rows(value, clocks):
    return {w: {**{k: value for k in e.METRICS}, 'estimator_ns': list(clocks)} for w in e.FINAL_SUITE}


def applied(name):
    path = ADVERSARIAL/(name+'.patch') if not str(name).endswith('.patch') else Path(name)
    result = apply_submission_patch(path, e.TASK_DIR, e.CONFIG)
    assert result.ok, result.message
    return result


def test_fixture_inventory_is_complete():
    names = {p.stem for p in ADVERSARIAL.glob('*.patch')}
    assert names == set(FIXED_SURFACE) | set(POLICY) | set(BUILD) | set(ACCEPTED)


def test_reference_parity_does_not_pass(calibration):
    """The reference against itself: accurate, but never strictly faster in every pair."""
    equal = e.summarize(rows(0.0, [5, 5, 5]), rows(0.0, [5, 5, 5]), e.FINAL_SUITE, {})
    assert equal[0] == 100 and equal[3]['beats_reference'] == 0
    assert all(row['metrics'] == {e.METRICS[0]: True, e.METRICS[1]: True, e.TIME_METRIC: False}
               for row in equal[3]['beat_spec_workload_passes'])

    # Ordinary timing noise: ahead in some pairs, behind in others.
    noisy = e.summarize(rows(0.0, [4, 6, 5]), rows(0.0, [5, 5, 6]), e.FINAL_SUITE, {})
    assert noisy[3]['beats_reference'] == 0
    faster = e.summarize(rows(0.0, [4, 4, 4]), rows(0.0, [5, 5, 5]), e.FINAL_SUITE, {})
    assert faster[3]['beats_reference'] == 1
    assert faster[3]['estimator_seconds'][e.FINAL_SUITE[0]] == [4e-9]*3
    assert faster[3]['reference_estimator_seconds'][e.FINAL_SUITE[0]] == [5e-9]*3


def test_full_final_evidence_member_roundtrip_and_unchanged_train(monkeypatch):
    import hashlib
    monkeypatch.syspath_prepend(str(JUDGE_ROOT.parent/'verifier'))
    native = load(JUDGE_ROOT.parent/'verifier/fcs2_verifier.py', 'ckks_native_transport')
    from ckks_evidence_transport import unpack
    # A full final has 87 pairs and retains raw observations from both sides.
    value = {'pairs': [{'index': i, 'candidate': 'raw-' + str(i)*8000,
                        'reference': 'raw-' + str(i)*8000} for i in range(87)]}
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE', 'final')
    members = {}
    result = native.run_evaluator(lambda _: (0.0, 0.0, 'transport fixture',
        {'operator_evidence': e.pack_evidence(value)}), Path('fixture.patch'), 'eval',
        evidence_members=members)
    descriptor = result[3]['operator_evidence']
    assert descriptor['member'] == 'ckks/calibration.json.xz'
    assert hashlib.sha256(members[descriptor['member']]).hexdigest() == descriptor['compressed_sha256']
    assert unpack(descriptor, members[descriptor['member']]) == value
    assert len(json.dumps(result[3]).encode()) < native.MAX_METRICS_BYTES
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE', 'train')
    small = {'observations': [1, 2, 3]}
    raw = json.dumps(small, sort_keys=True, allow_nan=False, separators=(',', ':'))
    assert e.pack_evidence(small) == {'encoding': 'json-chunks-v1',
        'sha256': hashlib.sha256(raw.encode()).hexdigest(), 'chunks': [raw]}


def test_one_lost_pair_in_one_configuration_blocks_the_pass(calibration):
    for workload in e.FINAL_SUITE:
        for pair in range(e.PAIRS):
            candidate = rows(0.0, [4, 4, 4]); candidate[workload]['estimator_ns'][pair] = 5
            result = e.summarize(candidate, rows(0.0, [5, 5, 5]), e.FINAL_SUITE, {})
            assert result[0] == 100 and result[3]['beats_reference'] == 0
            failed = [r['workload'] for r in result[3]['beat_spec_workload_passes'] if not r['metrics'][e.TIME_METRIC]]
            assert failed == [workload]


def test_faster_but_less_accurate_does_not_pass(calibration):
    result = e.summarize(rows(1.0, [1, 1, 1]), rows(0.0, [5, 5, 5]), e.FINAL_SUITE, {})
    assert 0 < result[0] < 100 and result[3]['beats_reference'] == 0
    assert all(row['metrics'][e.TIME_METRIC] for row in result[3]['beat_spec_workload_passes'])


def test_development_feedback_is_one_pair_on_six_base_rows_and_gates_nothing(calibration, monkeypatch):
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE', 'train')
    assert e.suite() == e.DEV_FEEDBACK and e.pairs() == 1 and len(e.DEV_FEEDBACK) == 6
    assert set(e.DEV_FEEDBACK) < set(e.DEV_SUITE) and len(e.DEV_SUITE) == 21
    one = lambda value, clock: {w: {**{k: value for k in e.METRICS}, 'estimator_ns': [clock]} for w in e.DEV_FEEDBACK}
    faster = e.summarize(one(0.0, 4), one(0.0, 5), e.DEV_FEEDBACK, {})
    assert faster[3]['beats_reference'] == 1 and faster[3]['workload_count'] == 6
    assert faster[3]['verification_mode'] == 'development'
    parity = e.summarize(one(0.0, 5), one(0.0, 5), e.DEV_FEEDBACK, {})
    assert parity[0] == 100 and parity[3]['beats_reference'] == 0
    # Three clocks per row are a final-shaped observation, not a development one.
    with pytest.raises(ValueError):
        three = lambda clock: {w: {**{k: 0.0 for k in e.METRICS}, 'estimator_ns': [clock]*3} for w in e.DEV_FEEDBACK}
        e.summarize(three(4), three(5), e.DEV_FEEDBACK, {})
    # The full calibrated development list is no longer what a development call measures.
    wide = {w: {**{k: 0.0 for k in e.METRICS}, 'estimator_ns': [4]} for w in e.DEV_SUITE}
    with pytest.raises(ValueError, match='manifest'):
        e.summarize(wide, wide, e.DEV_SUITE, {})
    for bad in (['missing'], [e.FINAL_SUITE[6]], list(e.DEV_FEEDBACK)*2, []):
        with pytest.raises(ValueError):
            scoring.compare(one(0.0, 4), one(0.0, 5), list(e.DEV_FEEDBACK), calibration, mode='development', pairs=1, development=bad)
    # Final is untouched by the development protocol.
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE', 'final')
    assert e.suite() == e.FINAL_SUITE and e.pairs() == 3 == e.PAIRS


def test_clock_shape_is_validated():
    for bad in ([1, 1], [1, 1, 1, 1], [1, 0, 1], [1, 1.0, 1], [1, True, 1], [1, -1, 1]):
        with pytest.raises(ValueError):
            scoring.faster_in_every_pair(bad, [2, 2, 2], 3)
        with pytest.raises(ValueError):
            scoring.faster_in_every_pair([1, 1, 1], bad, 3)
    assert scoring.faster_in_every_pair([1, 1, 1], [2, 2, 2], 3)
    assert not scoring.faster_in_every_pair([2, 1, 1], [2, 2, 2], 3)


def test_mean_rows_and_incomplete_pairs(calibration):
    one = {**{k: 0.3 for k in e.METRICS}, 'estimator_ns': 7}
    two = {**{k: 0.1 for k in e.METRICS}, 'estimator_ns': 9}
    mean = e.mean_rows({'w': [one, two, two]})
    assert mean['w']['estimator_ns'] == [7, 9, 9]
    assert all(abs(mean['w'][k]-(0.5/3)) < 1e-12 for k in e.METRICS)
    # A configuration with a failed invocation is absent, and the suite is invalid.
    short = rows(0.0, [4, 4, 4]); short.pop(e.FINAL_SUITE[3])
    result = e.summarize(short, rows(0.0, [5, 5, 5]), e.FINAL_SUITE, {})
    assert result[0] == 0 and result[3]['beats_reference'] == 0 and result[3]['status'] == 'invalid_execution'


def test_pairs_are_interleaved_per_configuration(monkeypatch, tmp_path):
    order = []
    def one(tree, name, dest, deadline, identity):
        order.append((name, tree))
        return {**{k: 0.0 for k in e.METRICS}, 'estimator_ns': 1}, {'workload': name, 'status': 'ok'}
    monkeypatch.setattr(e, 'measure_one', one)
    monkeypatch.setattr(e, 'source_identity', lambda tree: {'sha256': str(tree), 'files': {}})
    names = e.FINAL_SUITE[:2]
    complete, detail = e.measure_pairs('C', 'R', names, tmp_path, 10, 3)
    assert order == [(n, t) for n in names for t in ('C', 'R', 'R', 'C', 'C', 'R')]
    assert all(len(complete[side][n]) == 3 for side in complete for n in names)
    assert [r['pair'] for r in detail['candidate']['observations']] == [0, 1, 2, 0, 1, 2]


def test_feedback_is_aggregate_and_reveals_no_reference(calibration):
    feedback = load(e.TASK_DIR.parent/'verifier/beat_spec_feedback.py', 'ckks_beat_spec_feedback')
    result = e.summarize(rows(0.0, [4, 4, 4]), rows(0.0, [5, 5, 5]), e.FINAL_SUITE, {})
    public = feedback.public_metrics(result[3], beat_spec_path=e.TASK_DIR/'beat_spec.json',
                                     configured_visible=e.CONFIG['evaluation']['solver_visible_metrics'])
    assert set(public) == {'candidate_avg_log2_prec_l2_abs_discrepancy_mean_bits',
                           'candidate_std_log2_prec_l2_abs_discrepancy_mean_bits',
                           'candidate_estimator_seconds_mean', 'beat_spec_progress'}
    assert set(public['beat_spec_progress']) == {*e.METRICS, e.TIME_METRIC}
    assert public['beat_spec_progress'][e.TIME_METRIC] == {'passed': 29, 'failed': 0, 'total': 29,
                                                         'pass_percent': 100.0, 'fail_percent': 0.0}
    text = json.dumps(public)
    assert 'reference' not in text and not any(w in text for w in e.FINAL_SUITE)


def test_pristine_reference_and_controls_keep_the_fixed_surface():
    assert fixed_surface_violations(PRISTINE, PRISTINE) == []
    for name in (e.TASK_DIR/'reference.patch', e.TASK_DIR/'baseline_naive.patch', *ACCEPTED, *POLICY, *BUILD):
        if name == 'fixed_clear_target':
            continue
        result = applied(name)
        try:
            assert fixed_surface_violations(PRISTINE, result.workdir) == [], name
        finally:
            shutil.rmtree(result.cleanup_root, ignore_errors=True)


@pytest.mark.parametrize('name', sorted(FIXED_SURFACE))
def test_edits_to_the_fixed_set_are_rejected(name, monkeypatch):
    result = applied(name)
    try:
        assert fixed_surface_violations(PRISTINE, result.workdir) == [FIXED_SURFACE[name]]
        # The evaluator refuses before compiling or running anything from the tree.
        monkeypatch.setattr(e, 'run', lambda *a, **k: pytest.fail('must not build a tree with a changed fixed surface'))
        with pytest.raises(e.EvaluationError, match='fixed surface changed') as failure:
            e.guards(result.workdir, Path(e.scratch_root()), time.monotonic()+10)
        assert failure.value.receipt['paths'] == [FIXED_SURFACE[name]]
    finally:
        shutil.rmtree(result.cleanup_root, ignore_errors=True)


def test_other_file_kinds_and_paths_outside_the_package_are_rejected(tmp_path):
    def patch(path, body='x\n'):
        return (f'diff --git a/{path} b/{path}\nnew file mode 100644\nindex 0000000..1111111\n--- /dev/null\n+++ b/{path}\n'
                f'@@ -0,0 +1 @@\n+{body}')
    for outside in ('frontier-build.sh', 'README.md', 'other/x.go', 'estimatorx/x.go'):
        ok, message, _ = validate_patch(patch(outside), e.CONFIG)
        assert not ok and 'outside the editable surface' in message
    for inside in ('estimator/notes.txt', 'estimator/x_test.go', 'estimator/experiments/new/new.go', 'estimator/data.bin'):
        p = tmp_path/'p.patch'; p.write_text(patch(inside, 'package estimator\n' if inside.endswith('.go') else 'x\n'))
        result = apply_submission_patch(p, e.TASK_DIR, e.CONFIG)
        assert result.ok
        try:
            assert fixed_surface_violations(PRISTINE, result.workdir) == [inside]
        finally:
            shutil.rmtree(result.cleanup_root, ignore_errors=True)
    assert e.CONFIG['submission']['allow_paths'] == ['estimator/']


def test_every_statement_that_reaches_the_estimator_is_on_the_clock():
    """Nothing the fixed programs call in package estimator runs before or after the clock."""
    for name in e.FINAL_SUITE:
        row = e.WORKLOADS[name]
        configured = c.configured_main((PRISTINE/row['main_path']).read_text(), row)
        derived = c.timed_main(configured, row).split('\n')
        start = derived.index('func main() {'); end = derived.index('}', start)
        on = False; collected = False
        for line in derived[start:end]:
            text = line.strip()
            if text == 'frontierStart = frontiertime.Now()':
                assert not on; on = True; collected = False; decrypts = False; continue
            if text == 'frontierEstimator += frontiertime.Since(frontierStart)':
                assert on and (collected or not decrypts); on = False; continue
            if text == 'frontierruntime.GC()':
                assert on and decrypts; collected = True; continue
            if on:
                decrypts = decrypts or 'Decrypt(' in line
            elif not text.startswith('//'):
                # Outside the clock: no estimator value, constructor or test-vector call.
                assert not c.ESTIMATOR_STATEMENT.search(line), (name, line)
        assert not on
        body = '\n'.join(derived[start:end])
        assert 'estimator.NewEstimator(' in body or 'bootEst.NewEvaluator(' in body
        for call in ('NewTestVector', 'Decrypt('):
            assert call in body
        # The reported lines are produced by the derived main after the last statistic.
        tail = body.split('\tstatsHave.Finalize()\n')[1]
        assert [l.split('"')[1] for l in tail.split('\n')[:4]] == ['parameters', 'final', 'final', 'estimator_ns']
    assert e.RELEASE['measured_gomaxprocs'] == 1 and e.clean_env()['GOMAXPROCS'] == '1'


def test_tampered_or_unrecognised_main_is_not_measured():
    row = e.WORKLOADS['poly_eval']
    source = (PRISTINE/row['main_path']).read_text()
    assert 'frontier' not in source
    with pytest.raises(ValueError):
        c.measured_main(source.replace('deg := 255', 'deg := 3'), row, e.observation_prefix())
    with pytest.raises(ValueError):
        c.timed_main(source.replace('est.Rescale(el, el)', 'est.Rescale(el, el) '), row)
    frozen = json.loads((e.TASK_DIR/'judge/timed_main_sha256.json').read_text())
    assert set(frozen) == set(e.FINAL_SUITE)
    assert all(set(v) <= {'authored', 'canonical_restored'} and 'authored' in v for v in frozen.values())


def test_no_file_of_package_estimator_is_instrumented():
    source = (e.TASK_DIR/'evaluator.py').read_text()
    assert 'frontierObserve' not in source and 'FrontierDiagnosticParameters' not in source
    for path in PRISTINE.rglob('*.go'):
        assert 'frontier' not in path.read_text().lower().replace('frontier-', ''), path


@GO
@pytest.mark.parametrize('name', POLICY)
def test_source_policy_rejects(name, tmp_path):
    result = applied(name)
    try:
        with pytest.raises(e.EvaluationError):
            e.guards(result.workdir, tmp_path, time.monotonic()+600)
    finally:
        shutil.rmtree(result.cleanup_root, ignore_errors=True)


@GO
@pytest.mark.parametrize('name', BUILD)
def test_interface_break_and_forged_hook_do_not_build(name, tmp_path):
    result = applied(name)
    try:
        identity = e.source_identity(result.workdir)
        failed = False
        try:
            e.guards(result.workdir, tmp_path, time.monotonic()+600)
        except e.EvaluationError:
            failed = True
        row, record = e.measure_one(result.workdir, 'mul_relin_rescale', tmp_path/'run', time.monotonic()+600, identity)
        assert row is None and record['status'] != 'ok' and (failed or name == 'forged_observation_hook')
    finally:
        shutil.rmtree(result.cleanup_root, ignore_errors=True)


@GO
@pytest.mark.parametrize('name', ('reference', *ACCEPTED))
def test_reference_and_controls_are_measured_and_forged_lines_ignored(name, tmp_path):
    result = applied(e.TASK_DIR/'reference.patch' if name == 'reference' else name)
    try:
        e.guards(result.workdir, tmp_path, time.monotonic()+900)
        row, record = e.measure_one(result.workdir, 'mul_relin_rescale', tmp_path/'run', time.monotonic()+900, e.source_identity(result.workdir))
        assert row is not None and record['status'] == 'ok'
        assert row['estimator_ns'] > 1000 and 20 < row['predicted_avg'] < 50 and row[e.METRICS[0]] < 1
    finally:
        shutil.rmtree(result.cleanup_root, ignore_errors=True)


def test_calibration_collector_runs_unchanged_on_the_new_measurement(monkeypatch, tmp_path):
    """The preregistered collector needs only measure_suite, guards and the reference payload.

    Measurement is stubbed: this checks the code path and the frozen arithmetic,
    not any band. A real collection needs a new preregistration that binds the
    current source files.
    """
    import random
    # The collector imports the evaluator by name; give it this module instance.
    for name, module in (('evaluator', e), ('formal_scoring', scoring), ('workload_configurations', c)):
        monkeypatch.setitem(sys.modules, name, module)
    calibration = load(e.TASK_DIR/'calibration.py', 'ckks_calibration')
    rng = random.Random(7)
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE', 'final')
    monkeypatch.setitem(e.RELEASE, 'mode', 'calibration')
    monkeypatch.setattr(calibration, 'verify_source_binding', lambda *a: True)
    monkeypatch.setattr(e, 'guards', lambda *a: None)
    monkeypatch.setattr(e, 'run', lambda *a: 'go version go1.23.12 linux/amd64')
    calls = []
    def suite(tree, names, dest, deadline, *, coverage=False):
        calls.append(coverage)
        return ({w: {**{k: abs(rng.gauss(0.05, 0.02)) for k in e.METRICS}, 'estimator_ns': 1} for w in names},
                {'observations': [], 'fixture_only': True})
    monkeypatch.setattr(e, 'measure_suite', suite)
    score, _, message, metrics = calibration.collect_native(str(e.TASK_DIR/'reference.patch'))
    assert score == 0 and metrics['status'] == 'calibration_collected' and metrics['beats_reference'] == 0
    assert metrics['calibration_pending'] is True and 'calibration-only' in message
    assert len(calls) == 2*(8+19+5+5)
    evidence = calibration.unpack(metrics['operator_evidence'])
    frozen = evidence['frozen']
    assert len(frozen['cells']) == 58 and frozen['K'] == 19 and frozen['batch_invocations'] == 1
    assert all(frozen['bands'][c] == frozen['q']*frozen['scales'][c] for c in frozen['cells'])
    assert {p['stage'] for p in evidence['pairs']} == {'pilot', 'calibration', 'validation', 'sensitivity_independent_incomplete'}
