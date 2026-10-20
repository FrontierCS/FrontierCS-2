import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

import evaluator
from test_fixtures import controlled_patch, copy_writable_tree, restored_tree, write_patch


@pytest.fixture(scope='module', autouse=True)
def downscaled_suites():
    development, final = evaluator.DEV_WORKLOADS, evaluator.FINAL_WORKLOADS
    evaluator.DEV_WORKLOADS = [development[index] for index in [0, 1, 6, 11]]
    evaluator.FINAL_WORKLOADS = [final[index] for index in [0, 1, 6, 11]]
    yield
    evaluator.DEV_WORKLOADS, evaluator.FINAL_WORKLOADS = development, final


@pytest.fixture(scope='module')
def reference_result():
    return evaluator.evaluate(str(evaluator.REFERENCE_PATCH))


def check_metrics(result, count):
    score, unbounded, message, metrics = result
    assert math.isfinite(score) and 0 <= score <= 100, message
    assert math.isfinite(unbounded)
    assert {'beats_reference', 'beat_spec_workload_passes', 'reference_score', 'margin'} <= metrics.keys(), message
    rows = metrics['beat_spec_workload_passes']
    assert len(rows) == count
    assert len({row['workload'] for row in rows}) == count
    assert all(isinstance(row['workload'], str) and row['workload'] for row in rows)
    assert all(set(row['metrics']) == set(evaluator.METRIC_NAMES) for row in rows)
    assert all(type(value) is bool for row in rows for value in row['metrics'].values())
    assert metrics['beats_reference'] == int(all(value for row in rows for value in row['metrics'].values()))


def test_reference_is_complete_and_finite(reference_result):
    check_metrics(reference_result, len(evaluator.DEV_WORKLOADS))
    assert reference_result[3]['development_im2col_discovered_count'] == len(evaluator.DEV_WORKLOADS)
    assert reference_result[0] > 70


def test_continuous_score_is_independent_of_private_reference():
    candidate = {'case': {'im2col_discovery': 1, 'rewrite_time_seconds': 1.0,
                          'egraph_node_growth': 1000}}
    weaker = {'case': {'im2col_discovery': 1, 'rewrite_time_seconds': 2.0,
                       'egraph_node_growth': 2000}}
    stronger = {'case': {'im2col_discovery': 1, 'rewrite_time_seconds': 0.5,
                         'egraph_node_growth': 500}}
    score_a, verdict_a = evaluator._aggregate(candidate, weaker)
    score_b, verdict_b = evaluator._aggregate(candidate, stronger)
    assert score_a == score_b == evaluator._score(candidate)
    assert all(verdict_a[0]['metrics'].values())
    assert not all(verdict_b[0]['metrics'].values())
    assert evaluator._score(stronger) > score_a > evaluator._score(weaker)
    candidate['case']['im2col_discovery'] = 0
    assert evaluator._score(candidate) == 0


def test_score_budget_anchor_and_workload_membership():
    anchor = {'case': {'im2col_discovery': 1, 'rewrite_time_seconds': 10.0,
                      'egraph_node_growth': 100000}}
    assert evaluator._score(anchor) == 80.0
    with pytest.raises(ValueError, match='membership'):
        evaluator._aggregate(anchor, {})


def test_reference_repeats_without_cached_measurements():
    binary = evaluator._trusted_binary()
    runs = [evaluator._measure(binary, evaluator.DEV_WORKLOADS[:2], binary) for _ in range(3)]
    for workload in runs[0]:
        assert {run[workload]['im2col_discovery'] for run in runs} == {1}
        assert len({run[workload]['egraph_node_growth'] for run in runs}) == 1
    scores = [evaluator._aggregate(run, runs[0])[0] for run in runs]
    assert max(scores) - min(scores) < 2.0


def test_valid_naive_patch_is_scored(reference_result):
    result = evaluator.evaluate(str(evaluator.BASELINE_PATCH))
    assert result[0] == 0
    assert result[3] == {"candidate_rejection": True, "beats_reference": 0}
    assert result[0] < reference_result[0]


def test_invalid_submission_scores_zero_without_crashing(tmp_path):
    bad = tmp_path / 'invalid.patch'
    bad.write_text('not a patch')
    score, unbounded, message, metrics = evaluator.evaluate(str(bad))
    assert score == unbounded == 0
    assert 'patch' in message
    assert metrics == {"candidate_rejection": True, "beats_reference": 0}


def test_valid_partial_and_distinct_exact_are_accepted(tmp_path, reference_result):
    partial = evaluator.evaluate(str(controlled_patch(tmp_path / 'partial', 'partial')))
    exact = evaluator.evaluate(str(controlled_patch(tmp_path / 'exact', 'exact')))
    check_metrics(partial, len(evaluator.DEV_WORKLOADS))
    check_metrics(exact, len(evaluator.DEV_WORKLOADS))
    assert 0 < partial[0] < reference_result[0]
    assert 0 < partial[3]['development_im2col_discovered_count'] < len(evaluator.DEV_WORKLOADS)
    assert exact[3]['development_im2col_discovered_count'] == len(evaluator.DEV_WORKLOADS)
    assert abs(exact[0] - reference_result[0]) < 2


def test_controlled_efficiency_variant_is_legal(tmp_path, reference_result):
    stronger = evaluator.evaluate(str(controlled_patch(tmp_path / 'selective', 'selective')))
    check_metrics(stronger, len(evaluator.DEV_WORKLOADS))
    assert stronger[3]['development_im2col_discovered_count'] == len(evaluator.DEV_WORKLOADS)
    assert stronger[0] > 0


def test_final_role_and_matched_split(monkeypatch):
    for role in ('train', 'agent', 'development', '', 'anything'):
        monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE', role)
        assert evaluator._suite()[1] is evaluator.DEV_WORKLOADS
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE', 'final')
    assert evaluator._suite()[1] is evaluator.FINAL_WORKLOADS
    assert len(evaluator.DEV_WORKLOADS) == len(evaluator.FINAL_WORKLOADS)
    for left, right in zip(evaluator.DEV_WORKLOADS, evaluator.FINAL_WORKLOADS):
        assert left['layout'] == right['layout']
        assert left['wrapper'] == right['wrapper']
        assert left['strides'] == right['strides']
        assert left['seed'] != right['seed']


def test_wrong_values_and_shapes_fail_independent_interpreter():
    binary = evaluator._trusted_binary()
    workload = evaluator.DEV_WORKLOADS[0]
    rows = evaluator._measure(binary, [workload], binary)
    valid = rows[workload['id']]['witness']
    wrong_values = valid.replace('(compute dot-product', '(compute reduce-sum')
    wrong_values = valid.replace('(access-tensor ' + workload['weight_name'] + ')', '(compute elementwise-mul (access-pair (access-tensor ' + workload['weight_name'] + ') (access-tensor ' + workload['weight_name'] + ')))')
    verdicts = evaluator._invoke(binary, {'mode': 'verify', 'workloads': [workload, workload, workload], 'witnesses': [valid, '(access-tensor ' + workload['data_name'] + ')', wrong_values]})
    assert verdicts == [True, False, False]


def test_immutable_mapping_and_io_policy(tmp_path):
    tree = tmp_path / 'tree'
    copy_writable_tree(evaluator.APP, tree)
    path = tree / evaluator.REWRITES
    starter = path.read_text()
    path.write_text(starter.replace('pub fn systolic_array()', 'pub fn altered_mapping()'))
    with pytest.raises(ValueError):
        evaluator._validate_candidate_tree(tree)
    path.write_text('use std::fs;\n' + starter)
    with pytest.raises(ValueError, match='pure generic'):
        evaluator._validate_candidate_tree(tree)


def test_score_is_not_zeroed_when_conjunction_fails():
    base = {'im2col_discovery': 1, 'rewrite_time_seconds': 1.0, 'egraph_node_growth': 100}
    score, passes = evaluator._aggregate({'work': {**base, 'rewrite_time_seconds': 2.0}}, {'work': base})
    assert score > 0
    assert passes[0]['metrics']['rewrite_time_seconds'] is False


def test_quota_render_and_allowlist():
    config = evaluator.CONFIG
    text = (evaluator.TASK / 'readme').read_text()
    assert f"At most {config['evaluation']['max_dev_submissions']} development submissions" in text
    assert '{max_trials}' not in text
    spec = json.loads((evaluator.TASK / 'beat_spec.json').read_text())
    assert config['evaluation']['solver_visible_metrics'] == [metric['name'] for metric in spec['solver_feedback']]


def test_sandbox_blocks_private_files_and_network(tmp_path):
    from sandbox import run_isolated
    private = tmp_path / 'private.txt'
    private.write_text('must-not-be-visible')
    script = f'''
import os, socket
try:
    open({str(private)!r}).read()
    raise AssertionError("private file was readable")
except PermissionError:
    pass
try:
    socket.socket()
    raise AssertionError("network was available")
except PermissionError:
    pass
assert not any("KEY" in key or "TOKEN" in key for key in os.environ)
print("isolated")
'''
    output = run_isolated(['/usr/bin/python3', '-c', script], mounts=[('/usr/bin/python3', '/python', False)], cwd='/', timeout=5, environment={'PATH': '/usr/bin:/bin'})
    assert output.strip() == 'isolated'


def test_manifest_and_exact_restoration(tmp_path):
    manifest = json.loads((evaluator.TASK / 'artifact_manifest.json').read_text())
    from pathlib import PurePosixPath
    upstream = PurePosixPath(manifest['upstream_root'])
    assert manifest['upstream_root'] not in ('', '.', './')
    assert not upstream.is_absolute() and '..' not in upstream.parts
    execution = manifest['authentic_execution']
    material_execution = manifest['material_execution']
    assert str(upstream / execution['entrypoint']) == material_execution['entrypoint']
    assert [str(upstream / path) for path in execution['retained_files']] == material_execution['retained_files']
    assert sorted(evaluator.CONFIG['submission']['allow_paths']) == sorted(manifest['solution_surface']['modify'] + manifest['solution_surface']['add_under'])
    tree = restored_tree(tmp_path / 'restored')
    try:
        import hashlib
        assert hashlib.sha256((tree / evaluator.REWRITES).read_bytes()).hexdigest() == manifest['reference_restored_sha256']
        assert hashlib.sha256((tree / evaluator.ANALYSIS).read_bytes()).hexdigest() == manifest['reference_core_sha256']
        for name, digest in manifest['retained_sha256'].items():
            assert hashlib.sha256((tree / name).read_bytes()).hexdigest() == digest
    finally:
        shutil.rmtree(tree.parent)


def test_calibration_and_runtime_dependencies():
    text = (evaluator.TASK / 'readme').read_text()
    calibration = text.split('## Calibration\n', 1)[1].split('\n## ', 1)[0]
    assert len(calibration.strip()) >= 500
    assert 'development calibration of the preceding authoring revision' in calibration
    assert 'non-reference candidate patches' in calibration
    assert 'Peak-to-peak rerun span' in calibration
    assert 'no tolerance' in calibration
    runtime = evaluator.CONFIG['runtime']
    required = {'bash', 'build-essential', 'cargo', 'git', 'patch', 'python3', 'rustc', 'libseccomp2', 'tar', 'util-linux', 'xz-utils'}
    assert required <= set(runtime['apt_packages'])
    assert required <= set(runtime['judge_apt_packages'])
    assert 'pyyaml==6.0.2' in runtime['pip_packages']
    assert 'pyyaml==6.0.2' in runtime['judge_pip_packages']
    assert 'pytest==9.1.1' in runtime['pip_packages']
    assert 'pytest==9.1.1' in runtime['judge_pip_packages']


def test_public_example_matches_statement():
    import re
    readme = (evaluator.TASK / 'readme').read_text()
    source = (evaluator.APP / 'frontier-example.rs').read_text()
    expression = re.search(r'```lisp\n(.*?)\n```', readme, re.S).group(1)
    program = re.search(r'const PROGRAM: &str = r#"(.*?)"#;', source, re.S).group(1)
    assert expression.split() == program.split()
    shapes = json.loads(re.search(r'```json\n(.*?)\n```', readme, re.S).group(1))
    assert shapes == {'data': [1, 1, 3, 3], 'weights': [1, 1, 2, 2]}
    assert 'name_to_dtype' in readme and 'access_axis' in readme
    assert '++' in readme and 'floor((I[j]-K[j])/S[j])+1' in readme


def test_public_example_runs_retained_ir(tmp_path):
    import sys
    target = evaluator.WORK_ROOT / 'public-example-target'
    source = copy_writable_tree(evaluator.APP, tmp_path / 'public-source')
    compiled = subprocess.run([
        sys.executable, str(evaluator.APP / 'frontier-build.py'),
        '--output', str(tmp_path / 'example'), '--target-dir', str(target),
        '--source-root', str(source),
        '--harness', str(evaluator.APP / 'frontier-example.rs'),
    ], capture_output=True, text=True, timeout=180)
    assert compiled.returncode == 0, compiled.stderr[-4000:]
    result = subprocess.run([str(target / 'debug/examples/frontier_probe')],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert 'implement compositional access-analysis make' in result.stderr


@pytest.mark.parametrize('mode', ['valid', 'wrong-library', 'wrong-manifest', 'failed-build'])
def test_cargo_collector_dispatches_the_retained_target(tmp_path, monkeypatch, mode):
    source = tmp_path / 'code'
    (source / 'src').mkdir(parents=True)
    shutil.copyfile(evaluator.APP / evaluator.ENTRYPOINT, source / 'src/lib.rs')
    shutil.copyfile(evaluator.APP / evaluator.PROJECT / 'Cargo.toml', source / 'Cargo.toml')
    if mode == 'wrong-library':
        (source / 'src/lib.rs').write_text('pub fn unrelated() {}')
    if mode == 'wrong-manifest':
        (source / 'Cargo.toml').write_text('[package]\nname = "unrelated"\n')
    calls = []

    def compile_target():
        calls.append(source)
        if mode == 'failed-build':
            raise RuntimeError('compiler rejected the submission')

    monkeypatch.setenv('PYTEST_ADDOPTS', '--collect-only')
    if mode == 'valid':
        evaluator._compile_library(source, compile_target)
        assert calls == [source]
    else:
        failure_message = 'compiler rejected' if mode == 'failed-build' else 'library build failed'
        with pytest.raises(RuntimeError, match=failure_message):
            evaluator._compile_library(source, compile_target)
        assert calls == ([source] if mode == 'failed-build' else [])
    import os
    assert os.environ['PYTEST_ADDOPTS'] == '--collect-only'


def test_import_by_path_handles_invalid_input(tmp_path):
    script = '''
import importlib.util, sys
spec = importlib.util.spec_from_file_location("isolated_evaluator", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
try:
    module.evaluate("does-not-exist.patch")
except FileNotFoundError:
    pass
else:
    raise AssertionError("missing accepted payload is not a scientific result")
'''
    import sys
    subprocess.run([sys.executable, '-c', script, str(evaluator.TASK / 'evaluator.py')], cwd=tmp_path, check=True)


def test_fresh_extension_module_participates(tmp_path):
    root = tmp_path / 'tree'
    restored_tree(root)
    source = root / evaluator.REWRITES
    source.write_text('#[path = "solver_extensions/helper.rs"]\nmod helper;\n' + source.read_text().replace('pub fn flatten_unflatten_any_access() -> RW {', 'pub fn flatten_unflatten_any_access() -> RW {\n    assert!(helper::enabled());'))
    helper = root / evaluator.PROJECT / 'src/language/solver_extensions/helper.rs'
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text('pub fn enabled() -> bool { true }\n')
    patch = tmp_path / 'extension.patch'
    write_patch(root, patch)
    result = evaluator.evaluate(str(patch))
    check_metrics(result, len(evaluator.DEV_WORKLOADS))
    assert result[0] > 0
