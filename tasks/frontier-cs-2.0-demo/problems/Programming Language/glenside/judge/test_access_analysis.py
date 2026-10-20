import hashlib
import json
from pathlib import Path

import pytest

import evaluator
from submission_patch import validate_patch
from test_fixtures import copy_writable_tree, restored_tree


def test_reference_restores_only_the_analysis_core(tmp_path):
    reference = evaluator.REFERENCE_PATCH.read_text()
    valid, _, _ = validate_patch(reference, evaluator.CONFIG)
    assert valid
    assert reference.count('diff --git ') == 1
    assert 'a/' + str(evaluator.ANALYSIS) in reference
    root = restored_tree(tmp_path / 'reference')
    evaluator._validate_candidate_tree(root)
    manifest = json.loads((evaluator.TASK / 'artifact_manifest.json').read_text())
    core = (root / evaluator.ANALYSIS).read_text()
    assert hashlib.sha256(core.encode()).hexdigest() == manifest['reference_core_sha256']
    source = (root / 'code/src/language/language.rs').read_text()
    expanded = source.replace('include!("solver_extensions/access_analysis.rs");\n', core)
    coverage = (evaluator.TASK / 'source-coverage/access_analysis_tests.rs').read_text()
    assert coverage.count('#[test]') == 132
    assert '#[cfg(test)]\nmod tests' not in source
    assert hashlib.sha256((expanded + '\n' + coverage).encode()).hexdigest() == manifest['restored_language_sha256']


def test_public_core_is_explicit_missing_behavior():
    source = (evaluator.APP / evaluator.ANALYSIS).read_text()
    assert source.count('unimplemented!') == 2
    assert 'fn make(' in source and 'fn merge(' in source
    assert 'match enode' not in source and 'match (to' not in source
    copies = []
    for path in (evaluator.APP / 'code/src').rglob('*.rs'):
        if 'impl egg::Analysis<Language> for MyAnalysis {' in path.read_text():
            copies.append(path.relative_to(evaluator.APP))
    assert copies == [evaluator.ANALYSIS]


def test_analysis_is_editable_but_types_are_immutable(tmp_path):
    tree = copy_writable_tree(evaluator.APP, tmp_path / 'tree')
    core = tree / evaluator.ANALYSIS
    core.write_text(core.read_text() + '\n')
    evaluator._validate_candidate_tree(tree)
    language = tree / 'code/src/language/language.rs'
    language.write_text(language.read_text() + '\n')
    with pytest.raises(ValueError, match='immutable'):
        evaluator._validate_candidate_tree(tree)


@pytest.mark.parametrize('name', ['systolic_array', 'systolic_array_with_blocking'])
def test_both_fixed_mappers_are_protected(tmp_path, name):
    tree = copy_writable_tree(evaluator.APP, tmp_path / 'tree')
    path = tree / evaluator.REWRITES
    source = path.read_text()
    body = evaluator._function(source, name)
    path.write_text(source.replace(body, body.replace('{', '{ panic!("changed");', 1)))
    with pytest.raises(ValueError, match='mapping function'):
        evaluator._validate_candidate_tree(tree)


@pytest.mark.parametrize('source', [
    'use std::fs;', 'unsafe fn forbidden() {}', 'include!("answer.rs");',
    'const MARKER: &str = "//"; unsafe fn forbidden() {}',
    'const MARKER: &str = r#"/*"#; unsafe fn forbidden() {}',
])
def test_analysis_helpers_cannot_add_io_or_source_redirection(tmp_path, source):
    tree = copy_writable_tree(evaluator.APP, tmp_path / 'tree')
    path = tree / evaluator.ANALYSIS
    path.write_text(path.read_text() + '\n' + source + '\n')
    with pytest.raises(ValueError, match='pure generic'):
        evaluator._validate_candidate_tree(tree)


def test_exact_comparisons_allow_equality_and_score_efficiency():
    row = {'im2col_discovery': 1, 'rewrite_time_seconds': 1.0, 'egraph_node_growth': 100}
    score, passes = evaluator._aggregate({'case': row}, {'case': row})
    assert all(passes[0]['metrics'].values())
    faster = {**row, 'rewrite_time_seconds': 0.5}
    smaller = {**row, 'egraph_node_growth': 50}
    assert evaluator._aggregate({'case': faster}, {'case': row})[0] > score
    assert evaluator._aggregate({'case': smaller}, {'case': row})[0] > score
    slower = {**row, 'rewrite_time_seconds': 1.00000001}
    larger = {**row, 'egraph_node_growth': 101}
    assert not evaluator._aggregate({'case': slower}, {'case': row})[1][0]['metrics']['rewrite_time_seconds']
    assert not evaluator._aggregate({'case': larger}, {'case': row})[1][0]['metrics']['egraph_node_growth']


def test_workload_and_resource_contract_is_preserved():
    assert len(evaluator.DEV_WORKLOADS) == len(evaluator.FINAL_WORKLOADS) == 14
    assert evaluator.CONFIG['evaluation']['max_dev_submissions'] == 100
    assert evaluator.CONFIG['runtime']['timeout_seconds'] == 300
    assert evaluator.CONFIG['environment']['cpus'] == 2
    assert evaluator.CONFIG['environment']['memory_mb'] == 4096
    assert evaluator.CONFIG['submission']['allow_paths'] == [
        'code/src/language/rewrites.rs', 'code/src/language/solver_extensions/']
