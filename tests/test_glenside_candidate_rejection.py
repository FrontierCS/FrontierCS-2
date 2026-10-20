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
TASK = Path(os.environ.get('GLENSIDE_REJECTION_EXPORT', str(ROOT / 'tasks/frontier-cs-2.0-demo/problems/Programming Language/glenside')))

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
        sys.path.insert(0, str(TASK / 'judge'))
        for name in ('submission_patch', 'sandbox', 'cargo_target'):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location('glenside_rejection_evaluator', TASK / 'judge/evaluator.py')
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

    def test_candidate_rejection_returns_zero_and_marker(self):
        e = self.e
        with tempfile.TemporaryDirectory() as tmp:
            answer = Path(tmp) / 'solution.patch'
            answer.write_text('not a patch')
            result = e.evaluate(str(answer))
        self.assertEqual(result[:2], (0, 0))
        self.assertEqual(result[3], {'candidate_rejection': True, 'beats_reference': 0})

    def test_application_disk_fault_and_missing_submission_surface(self):
        e = self.e
        with tempfile.TemporaryDirectory() as tmp:
            answer = Path(tmp) / 'solution.patch'
            with self.assertRaises(FileNotFoundError):
                e.evaluate(str(answer))
            answer.write_text('candidate')
            with patch.object(e, '_apply', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    e.evaluate(str(answer))
        self.assertIsNone(e._DEADLINE)

    def test_trusted_source_rejection_is_not_candidate_fail(self):
        e = self.e
        e._TRUSTED_BINARY = None
        with patch.object(e, '_apply', side_effect=e.CandidateRejection('bad patch')):
            with self.assertRaises(RuntimeError):
                e._trusted_binary()

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

    def test_verified_pristine_context_rejection_but_disk_fault_is_unknown(self):
        from submission_patch import apply_submission_patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pristine = root / 'repo'
            pristine.mkdir()
            (pristine / 'file.rs').write_text('original\n')
            submission = root / 'solution.patch'
            submission.write_text('diff --git a/file.rs b/file.rs\n--- a/file.rs\n+++ b/file.rs\n@@ -1 +1 @@\n-other\n+new\n')
            config = {'submission': {'allow_paths': ['file.rs']}}
            result = apply_submission_patch(submission, root, config, pristine=pristine)
            self.assertFalse(result.ok)
            self.assertTrue(result.candidate_rejection, result.message)
            import submission_patch as producer
            fake = type('Receipt', (), {'returncode': 1, 'stderr': 'error: unable to write: No space left on device', 'stdout': ''})()
            with patch.object(producer.subprocess, 'run', return_value=fake):
                result = apply_submission_patch(submission, root, config, pristine=pristine)
            self.assertFalse(result.candidate_rejection)

    def test_unknown_verifier_exception_is_execution_failure(self):
        sys.path.insert(0, str(TASK / 'verifier'))
        sys.modules.pop('beat_spec_feedback', None)
        spec = importlib.util.spec_from_file_location('glenside_infra_adapter', TASK / 'verifier/fcs2_verifier.py')
        v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v)
        for error in (OSError('tool missing'), subprocess.TimeoutExpired('probe', 60), RuntimeError('trusted measurement failed')):
            def fault(path):
                raise error
            with self.assertRaises(v.VerifierFailure) as caught:
                v.run_evaluator(fault, Path('accepted.patch'), 'eval')
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
        sys.path.insert(0, str(TASK / 'verifier'))
        sys.modules.pop('beat_spec_feedback', None)
        spec = importlib.util.spec_from_file_location('glenside_rejection_adapter', TASK / 'verifier/fcs2_verifier.py')
        v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v)
        marker = {'candidate_rejection': True, 'beats_reference': 0}
        for mode in ('train', 'eval'):
            with tempfile.TemporaryDirectory() as tmp:
                r = Path(tmp)
                payload = r / 'payload'
                payload.write_bytes(b'candidate')
                dataset = r / 'dataset'
                dataset.write_bytes(b'dataset')
                with (patch.object(v, 'load_candidate_diagnostic', return_value=None),
                      patch.object(v, 'extract_payload'),
                      patch.object(v, 'entrypoint_path', return_value=payload),
                      patch.object(v, 'stage_task', return_value=TASK / 'judge'),
                      patch.object(v, 'load_evaluator'),
                      patch.object(v, 'run_evaluator', return_value=(0, 0, 'candidate rejected', marker))):
                    result = v.verify(bundle=TASK / 'judge', entrypoint='solution.patch', payload=payload, dataset=dataset, mode=mode, result_path=r / 'result.json', evidence_path=r / 'evidence.tar.zst')
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(result['metrics'], {'score_unbounded': 0})
                if mode == 'eval':
                    self.assertEqual(result['private_metrics'], {'beats_reference': 0})
                else:
                    self.assertNotIn('private_metrics', result)
                with zstandard.ZstdDecompressor().stream_reader((r / 'evidence.tar.zst').open('rb')) as reader, tarfile.open(fileobj=reader, mode='r|') as archive:
                    evidence = json.load(archive.extractfile(archive.next()))
                self.assertEqual(evidence['metrics'], marker)

    def test_marker_validation_is_strict_and_not_zero_heuristic(self):
        spec = importlib.util.spec_from_file_location('glenside_beat_feedback', TASK / 'verifier/beat_spec_feedback.py')
        feedback = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feedback)
        self.assertEqual(feedback.rejection_verdict({'candidate_rejection': True, 'beats_reference': 0}, 0, 0), {'beats_reference': 0})
        self.assertIsNone(feedback.rejection_verdict({}, 0, 0))
        for metrics in ({'candidate_rejection': 1, 'beats_reference': 0}, {'candidate_rejection': True, 'beats_reference': 1}, {'candidate_rejection': True, 'beats_reference': 0, 'reference_score': 1}):
            with self.assertRaises(ValueError):
                feedback.rejection_verdict(metrics, 0, 0)

if __name__ == '__main__':
    unittest.main()
