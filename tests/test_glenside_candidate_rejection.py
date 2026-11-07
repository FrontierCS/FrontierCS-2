"""Candidate rejection needs explicit provenance; trusted faults remain execution failures."""
import deployment_test_guard  # noqa: F401
import importlib.util
import json
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get('GLENSIDE_REJECTION_EXPORT', str(ROOT / 'tasks/problems/Programming Language/glenside')))
sys.path.insert(0, str(ROOT / 'scripts'))
import task_manifest  # noqa: E402
SHARED_VERIFIER = task_manifest.SHARED / 'verifier'


def shared_verifier(name):
    sys.path.insert(0, str(SHARED_VERIFIER))
    sys.modules.pop('beat_spec_feedback', None)
    spec = importlib.util.spec_from_file_location(name, SHARED_VERIFIER / 'fcs2_verifier.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class GlensideRejectionTests(unittest.TestCase):
    def setUp(self):
        old_tempdir = tempfile.tempdir
        self.addCleanup(lambda: setattr(tempfile, "tempdir", old_tempdir))
        old = dict(sys.modules)
        old_path = list(sys.path)
        def restore():
            sys.path[:] = old_path
            for name in set(sys.modules) - set(old):
                del sys.modules[name]
            sys.modules.update(old)
        self.addCleanup(restore)
        self.judge = task_manifest.judge_bundle(TASK)
        sys.path.insert(0, str(self.judge))
        for name in ('sandbox', 'cargo_target'):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location('glenside_rejection_evaluator', self.judge / 'evaluator.py')
        self.e = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.e)
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.e.WORK_ROOT = Path(work.name)

    def test_completed_analysis_mismatch_is_explicit_rejection(self):
        e = self.e
        with patch.object(e, '_invoke', side_effect=[[{'case': str(i), 'value': True} for i in range(34)], [{'case': str(i), 'value': False} for i in range(34)]]):
            with self.assertRaises(e.CandidateRejection):
                e._check_analysis_contract(Path('candidate'), Path('trusted'))

    def test_trusted_bad_contract_and_candidate_invoke_fault_are_not_rejection(self):
        e = self.e
        for responses in ([[], []], [[{'case': str(i), 'value': True} for i in range(34)], OSError('runtime unavailable')], [[{'case': str(i), 'value': True} for i in range(34)], subprocess.TimeoutExpired('probe', 60)]):
            with patch.object(e, '_invoke', side_effect=responses):
                with self.assertRaises((ValueError, RuntimeError)) as caught:
                    e._check_analysis_contract(Path('candidate'), Path('trusted'))
                self.assertNotIsInstance(caught.exception, e.CandidateRejection)

    def test_trusted_source_rejection_is_not_candidate_fail(self):
        e = self.e
        e._TRUSTED_BINARY = None
        with patch.object(e, '_build', side_effect=e.CandidateRejection('bad source')):
            with self.assertRaises(RuntimeError):
                e._trusted_binary()

    def test_compile_or_contract_rejection_is_a_bare_scored_zero(self):
        e = self.e
        tree = e.APP
        for stage in ('_build', '_check_analysis_contract'):
            with patch.object(e, '_is_reference', return_value=False), \
                 patch.object(e, '_trusted_binary', return_value=Path('trusted')), \
                 patch.object(e, '_build', return_value=Path('candidate')) as build, \
                 patch.object(e, '_check_analysis_contract') as contract:
                (build if stage == '_build' else contract).side_effect = e.CandidateRejection('candidate source does not compile')
                self.assertEqual(e.evaluate(str(tree)), (0.0, 0.0, 'candidate source does not compile', {}))
        self.assertIsNone(e._DEADLINE)

    def test_check_tree_maps_source_rules_to_shared_categories(self):
        import shutil
        e = self.e
        change = type('Change', (), {'status': 'M', 'path': 'code/src/language/rewrites.rs', 'added': ()})()
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / 'tree'
            shutil.copytree(e.APP, tree)
            rewrites = tree / 'code/src/language/rewrites.rs'
            rewrites.chmod(0o644)
            self.assertIsNone(e.check_tree(str(tree), str(e.APP), [change]))
            original = rewrites.read_text()
            for addition, category in (('\nfn leak() { println!("x"); }\n', 'content_policy'),
                                       ('\nfn f() -> u8 { let _ = SystolicArray; 0 }\n', 'content_policy'),
                                       ('\n// println is fine inside a comment\n', None)):
                rewrites.write_text(original + addition)
                verdict = e.check_tree(str(tree), str(e.APP), [change])
                self.assertEqual(verdict and verdict[0], category, addition)
            fixed = e._function(original, 'systolic_array')
            rewrites.write_text(original.replace(fixed, fixed.replace('{', '{ let _unused = 0;', 1)))
            self.assertEqual(e.check_tree(str(tree), str(e.APP), [change]), ('out_of_scope', [change.path]))

    def test_compile_diagnostic_requires_editable_primary_error(self):
        e = self.e
        def receipt(code=101, filename='src/language/rewrites.rs', level='error'):
            row = {'reason': 'compiler-message', 'message': {'level': level, 'spans': [{'is_primary': True, 'file_name': filename}]}}
            return e.IsolatedProcessError(code, json.dumps(row), 'build failed')
        self.assertTrue(e._candidate_compile_diagnostic(receipt()))
        self.assertTrue(e._candidate_compile_diagnostic(receipt(filename='/build/code/src/language/solver_extensions/foo.rs')))
        for value in (receipt(1), receipt(filename='src/language/interpreter.rs'), receipt(level='warning'), e.IsolatedProcessError(101, '', 'No space left on device')):
            self.assertFalse(e._candidate_compile_diagnostic(value))

    def test_pytest_build_preserves_original_typed_failure(self):
        e = self.e
        error = e.CandidateRejection('candidate syntax error')
        def fake_pytest(args, plugins):
            try:
                plugins[0].action()
            except e.CandidateRejection:
                return 1
            return 0
        with patch.object(e.pytest, 'main', side_effect=fake_pytest):
            with self.assertRaises(e.CandidateRejection) as caught:
                e._compile_library(Path('prepared'), lambda: (_ for _ in ()).throw(error))
        self.assertIs(caught.exception, error)

    def test_unknown_verifier_exception_is_execution_failure(self):
        v = shared_verifier('glenside_infra_adapter')
        for error in (OSError('tool missing'), subprocess.TimeoutExpired('probe', 60), RuntimeError('trusted measurement failed')):
            def fault(path):
                raise error
            with self.assertRaises(v.VerifierFailure) as caught:
                v.run_evaluator(fault, Path('accepted.patch'), 'eval', max_metrics_bytes=524288)
            self.assertEqual(caught.exception.code, 'evaluator_error')

    def test_semantic_projection_retains_wrong_shapes_and_merge_settling(self):
        e = self.e
        base = {'case': 'reshape', 'facts': [{'variant': 'AccessPattern', 'shape': [2], 'item_shape': [12], 'settled': False, 'accelerator': False}], 'output': {'shape': [2, 12]}}
        import copy
        alternative = copy.deepcopy(base)
        alternative['facts'][0]['settled'] = True
        self.assertEqual(e._semantic_contract_row(base), e._semantic_contract_row(alternative))
        for key, value in [('shape', [3]), ('item_shape', [8]), ('accelerator', True)]:
            bad = copy.deepcopy(base)
            bad['facts'][0][key] = value
            self.assertNotEqual(e._semantic_contract_row(base), e._semantic_contract_row(bad))
        left = {'case': 'settle-forward', 'merged': {'settled': True}, 'flags': [True, False]}
        right = {'case': 'settle-forward', 'merged': {'settled': False}, 'flags': [True, False]}
        self.assertNotEqual(e._semantic_contract_row(left), e._semantic_contract_row(right))
        for name in ('both-unsettled', 'zero-region-retained'):
            self.assertEqual(e._semantic_contract_row({'case': name, 'flags': [False, False]}), e._semantic_contract_row({'case': name, 'flags': [False, True]}))

    def test_node_zero_completeness_is_not_merge_preservation(self):
        e = self.e
        base = {'case': 'zero-pad', 'facts': [{'zero_regions': {}}]}
        alternative = {'case': 'zero-pad', 'facts': [{'zero_regions': {'2': [True, False]}}]}
        self.assertEqual(e._semantic_contract_row(base), e._semantic_contract_row(alternative))
        left = {'case': 'zero-region-retained', 'merged': {'zero_regions': {'2': [True]}}}
        right = {'case': 'zero-region-retained', 'merged': {'zero_regions': {}}}
        self.assertNotEqual(e._semantic_contract_row(left), e._semantic_contract_row(right))

    def test_dependency_error_on_editable_span_never_candidate_rejection(self):
        e = self.e
        row = {'reason': 'compiler-message', 'message': {'level': 'error', 'code': {'code': 'E0463'}, 'spans': [{'is_primary': True, 'file_name': 'src/language/rewrites.rs'}]}}
        self.assertFalse(e._candidate_compile_diagnostic(e.IsolatedProcessError(101, json.dumps(row), 'missing dependency')))

    def test_proven_semantic_panic_rejected_but_unknown_runtime_exit_is_not(self):
        e = self.e
        expected = [{'case': str(i), 'value': True} for i in range(34)]
        error = e.IsolatedProcessError(101, '', "thread 'main' panicked at examples/analysis_contract.rs:10:3:\nassertion public settled propagation failed")
        with patch.object(e, '_invoke', side_effect=[expected, error]):
            with self.assertRaises(e.CandidateRejection):
                e._check_analysis_contract(Path('candidate'), Path('trusted'))
        later_fault = e.IsolatedProcessError(101, '', error.stderr + "\nthread 'main' panicked at examples/probe.rs:20:3:\ntrusted transport failure")
        with patch.object(e, '_invoke', side_effect=[expected, later_fault]):
            with self.assertRaises(RuntimeError) as caught:
                e._check_analysis_contract(Path('candidate'), Path('trusted'))
            self.assertNotIsInstance(caught.exception, e.CandidateRejection)
        unknown = e.IsolatedProcessError(101, '', 'unable to launch trusted process')
        with patch.object(e, '_invoke', side_effect=[expected, unknown]):
            with self.assertRaises(RuntimeError):
                e._check_analysis_contract(Path('candidate'), Path('trusted'))

    @patch.dict(os.environ)
    def test_actual_verifier_final_fail_and_train_privacy(self):
        import tarfile
        import zstandard
        v = shared_verifier('glenside_rejection_adapter')
        marker = {}
        for mode in ('train', 'eval'):
            with tempfile.TemporaryDirectory() as tmp:
                r = Path(tmp)
                payload = r / 'payload'
                payload.write_bytes(b'candidate')
                dataset = r / 'dataset'
                dataset.write_bytes(b'dataset')
                scope = type('Scope', (), {'ok': True, 'tree': r, 'changes': (), 'metrics': {},
                                           'dispose': lambda self: None})()
                edit_scope = type('EditScope', (), {'check': staticmethod(lambda *a, _scope=scope, **k: _scope),
                                                    'CATEGORIES': {}})
                with (patch.object(v, 'extract_payload'),
                      patch.object(v, 'entrypoint_path', return_value=payload),
                      patch.object(v, 'load_edit_scope', return_value=(edit_scope, None)),
                      patch.object(v, 'stage_task', return_value=self.judge),
                      patch.object(v, 'load_evaluator', return_value=(None, None)),
                      patch.object(v, 'run_evaluator', return_value=(0.0, 0.0, 'candidate source does not compile', marker))):
                    result = v.verify(bundle=self.judge, entrypoint='solution.patch', payload=payload, dataset=dataset, mode=mode, result_path=r / 'result.json', evidence_path=r / 'evidence.tar.zst')
                self.assertEqual(result['status'], 'passed')
                # Train returns only score_unbounded and eval the private {beats_reference: 0},
                # as before the migration.
                self.assertEqual(result['metrics'], {'score_unbounded': 0})
                if mode == 'eval':
                    self.assertEqual(result['private_metrics'], {'beats_reference': 0})
                else:
                    self.assertNotIn('private_metrics', result)
                with zstandard.ZstdDecompressor().stream_reader((r / 'evidence.tar.zst').open('rb')) as reader, tarfile.open(fileobj=reader, mode='r|') as archive:
                    evidence = json.load(archive.extractfile(archive.next()))
                self.assertEqual(evidence['metrics'], marker)

if __name__ == '__main__':
    unittest.main()
