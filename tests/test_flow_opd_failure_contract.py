"""flow_opd on the shared verifier: the scoring contract, the two-body rule (check_tree),
trusted faults stay unscored, and the shared GPU adapter keeps a failed remote result."""
import deployment_test_guard  # noqa: F401
import difflib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/frontier-cs-2.0-demo/problems/Machine Learning/flow_opd'
SHARED = ROOT / 'tasks/shared'
sys.path.insert(0, str(ROOT / 'scripts'))
import task_manifest  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scoring = load(TASK/'judge/judge/scoring.py', 'flow_failure_scoring')
feedback = load(SHARED/'verifier/beat_spec_feedback.py', 'flow_failure_feedback')
gpu = load(SHARED/'verifier-gpu/fcs2_verifier_gpu.py', 'flow_failure_gpu')
_PARAMETERS = Path(tempfile.mkdtemp(prefix='flow-failure-')) / 'verifier.json'
_PARAMETERS.write_text(json.dumps({'gpu': {'placement': 'runner', 'max_gpu_instances': 1,
                                           'lease_seconds': {'train': 7200, 'eval': 7200}}}))
gpu.PARAMETERS = _PARAMETERS  # gpu-tools v4 reads [gpu.verifier] from verifier.json
BUNDLE = task_manifest.judge_bundle(TASK)
sys.path.insert(0, str(BUNDLE))
evaluator = load(BUNDLE/'evaluator.py', 'flow_failure_evaluator')
PIPELINE = 'flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py'
SDE = 'flow_grpo/diffusers_patch/sd3_sde_with_logprob.py'


class CandidateRejectionTests(unittest.TestCase):
    def test_rejection_uses_frozen_reference_and_explicit_unmeasured_checks(self):
        metrics = scoring.failure_metrics(['a','b'], reference={'a':.25,'b':.75}, final=True)
        private = feedback.private_metrics(metrics)
        self.assertEqual(private['beats_reference'], 0)
        self.assertEqual(private['reference_score'], 50)
        self.assertEqual(private['margin'], -50)
        self.assertEqual(private['beat_spec_progress']['ocr_reward']['total'], 2)
        self.assertEqual(private['beat_spec_progress']['ocr_reward']['passed'], 0)
        self.assertEqual(metrics['candidate_rejection']['measurements'], 'not_established')
        self.assertNotIn('candidate_reward', metrics)

    def test_missing_invalid_or_incomplete_reference_is_a_trusted_failure(self):
        for ids, reference in [([],{}), (['a'],{}), (['a'],{'a':float('nan')}),
                               (['a','a'],{'a':.5})]:
            with self.subTest(ids=ids, reference=reference):
                with self.assertRaises(scoring.QualificationError):
                    scoring.failure_metrics(ids, reference=reference, final=True)

    def test_development_rejection_uses_public_reference_only(self):
        metrics = scoring.failure_metrics(['private'], reference={'private':.95},
                                          feedback_reference=[.2,.4], final=False)
        public = feedback.public_metrics(metrics, visible=task_manifest.read_task(TASK)['feedback'])
        self.assertEqual(metrics['reference_score'], 30.000000000000004)
        self.assertNotIn('reference_score', public)
        self.assertNotIn('private', json.dumps(public))

    def test_valid_science_is_unchanged(self):
        args = (['a','b'], {'a':1.,'b':.8}, {'a':1.,'b':.7})
        score, unbounded, message, metrics = scoring.aggregate(
            *args,candidate_valid=True,reference_valid=True,final=True)
        self.assertEqual(score, 90)
        self.assertEqual(unbounded, 90)
        self.assertEqual(message, 'Completed OCR evaluation.')
        self.assertEqual(metrics['reference_score'], 85)
        self.assertEqual(metrics['margin'], 5)
        self.assertEqual(metrics['beats_reference'], 1)
        self.assertEqual(metrics['beat_spec_workload_passes'], [
            {'workload':'a','metrics':{'ocr_reward':True}},
            {'workload':'b','metrics':{'ocr_reward':True}}])


class SurfaceTests(unittest.TestCase):
    """check_tree: only the two computational bodies may change; files stay UTF-8."""

    def tree_with(self, path, transform):
        scratch = Path(tempfile.mkdtemp(prefix='flow-surface-'))
        self.addCleanup(shutil.rmtree, scratch, True)
        tree = scratch/'tree'
        shutil.copytree(BUNDLE/'pristine', tree)
        target = tree/path
        target.chmod(0o644)
        target.write_bytes(transform(target.read_bytes()))
        return tree

    def change(self, path, status='M'):
        return [mock.Mock(status=status, path=path)]

    def test_pristine_passes(self):
        self.assertIsNone(evaluator.check_tree(str(BUNDLE/'pristine'), str(BUNDLE/'pristine'), []))

    def test_body_change_is_allowed(self):
        tree = self.tree_with(SDE, lambda b: b.replace(b'raise NotImplementedError', b'return None', 1))
        self.assertIsNone(evaluator.check_tree(str(tree), str(BUNDLE/'pristine'), self.change(SDE)))

    def test_import_or_signature_change_is_out_of_scope(self):
        for path, transform in (
                (SDE, lambda b: b'import os\n' + b),
                (PIPELINE, lambda b: b.replace(b'noise_level: float = 0.7', b'noise_level: float = 0.5', 1))):
            with self.subTest(path=path):
                tree = self.tree_with(path, transform)
                self.assertEqual(evaluator.check_tree(str(tree), str(BUNDLE/'pristine'), self.change(path)),
                                 ('out_of_scope', [path]))

    def test_non_utf8_file_is_malformed(self):
        tree = self.tree_with(SDE, lambda b: b + b'# \xff\n')
        self.assertEqual(evaluator.check_tree(str(tree), str(BUNDLE/'pristine'), self.change(SDE)),
                         ('malformed', [SDE]))


class RuntimeAdapterTests(unittest.TestCase):
    def test_instance_adapter_reaches_the_judge_transition_guard(self):
        # frontier_runtime.py loads parents[2]/judge/transition_checks.py for qualified profiles;
        # in the run's isolate instance that is {in}/judge, staged next to {in}/adapter.
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)
            (output/'corpus').mkdir()
            profile = output/'run-profile.json'
            profile.write_text(json.dumps({'dataset': str(output/'corpus')}))
            inputs = evaluator.instance_inputs(output, profile)
            staged = output/'in'
            staged.mkdir()
            for name in ('adapter', 'judge'):
                shutil.copytree(inputs[name], staged/name)
            runtime = staged/'adapter/app/frontier_runtime.py'
            self.assertTrue((runtime.parents[2]/'judge/transition_checks.py').is_file())
            self.assertTrue(os.access(staged/'adapter/app/frontier_shim/accelerate', os.X_OK))


class TrustedFailureTests(unittest.TestCase):
    def test_qualification_and_runtime_faults_remain_unscored(self):
        cases = (
            (mock.patch.object(evaluator, 'qualification', side_effect=evaluator.QualificationError('x')),
             'reference_or_qualification_invalid'),
            (mock.patch.object(evaluator, 'offline_model_root', side_effect=OSError('private')),
             'trusted_runtime_unclassified'),
        )
        for patcher, reason in cases:
            with self.subTest(reason=reason), patcher, mock.patch.dict(os.environ, {'FRONTIER_SUBMISSION_ROLE': 'final'}):
                with self.assertRaises(evaluator.TrustedEvaluationFailure) as caught:
                    evaluator.evaluate(str(BUNDLE/'pristine'))
                self.assertEqual(caught.exception.trusted_evaluation_failure, {'reason': reason})


class RemoteFailureRetentionTests(unittest.TestCase):
    def archive(self, path, payload, extra=None):
        with tarfile.open(path, 'w') as f:
            for name, body in [('result.json',json.dumps(payload).encode()), *(extra or [])]:
                member = tarfile.TarInfo(name);member.size=len(body)
                f.addfile(member, io.BytesIO(body))

    def test_failed_result_survives_missing_evidence_without_echoing_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/'outputs.tar'
            self.archive(archive, {'status':'failed','metrics':{},'failure':{
                'code':'invalid_evaluator_result','message':'private credentials and workload identifiers'}})
            result = gpu._read_remote_failure(archive, 32768)
        self.assertEqual(result, {'code':'invalid_evaluator_result','scientific_result':False,
                                  'evidence_available':False})
        self.assertNotIn('private', json.dumps(result))

    def test_success_score_unknown_code_and_extra_member_are_not_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/'outputs.tar'
            for payload, extra in [({'status':'passed','raw_score':0},None),
                ({'status':'failed','raw_score':0,'failure':{'code':'invalid_evaluator_result'}},None),
                ({'status':'failed','failure':{'code':'secret-code'}},None),
                ({'status':'failed','failure':{'code':[]}},None),
                ({'status':'failed','failure':{'code':{}}},None),
                ({'status':'failed','failure':{'code':'evaluator_error'}},[('other',b'x')])]:
                self.archive(archive,payload,extra)
                self.assertIsNone(gpu._read_remote_failure(archive,32768))

    def test_intact_failed_result_can_transport_its_original_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/'outputs.tar'
            self.archive(archive, {'status':'failed','failure':{'code':'trusted_evaluation_failure'}},
                         [('evidence.tar.zst',b'original-private-evidence')])
            result = gpu._read_remote_failure(archive,32768)
            self.assertTrue(result['evidence_available'])
            gpu._copy_outputs(archive,Path(temp)/'result.json',Path(temp)/'evidence.tar.zst',32768)
            self.assertEqual((Path(temp)/'evidence.tar.zst').read_bytes(),b'original-private-evidence')

    def test_remote_failure_control_always_cleans_and_retains_missing_companion(self):
        for companion, terminated in [(False, True), (True, True), (True, False)]:
            with self.subTest(companion=companion, terminated=terminated), tempfile.TemporaryDirectory() as temp:
                snapshots = []
                root = Path(temp)
                payload = {'status':'failed','failure':{'code':'trusted_evaluation_failure'}}

                def remote(ssh, command, outputs, *args, **kwargs):
                    self.archive(outputs, payload,
                                 [('evidence.tar.zst', b'original-private-evidence')] if companion else None)
                    return 1, False

                with (
                    mock.patch.dict(os.environ, {'FRONTIER_MAX_EVIDENCE_BYTES':'32768',
                                                  'FRONTIER_VERIFIER_DEADLINE_UNIX_MS':''}),
                    mock.patch.object(gpu, '_config', return_value=('url','verification','token','gpu',1)),
                    mock.patch.object(gpu, '_input_archive'),
                    mock.patch.object(gpu, '_create_runner', return_value={'runner_id':'owned','granted_seconds':7200}),
                    mock.patch.object(gpu, '_ssh_argv', return_value=['ssh']),
                    mock.patch.object(gpu, '_ssh_environment', return_value={}),
                    mock.patch.object(gpu, '_upload_inputs'),
                    mock.patch.object(gpu, '_run_remote_verifier', side_effect=remote),
                    mock.patch.object(gpu, '_terminate_runner', return_value=terminated) as delete,
                ):
                    arguments = dict(payload=root/'payload', dataset=root/'dataset',mode='eval',
                                     result_path=root/'result', evidence_path=root/'evidence',
                                     record_evidence=snapshots.append)
                    if companion and terminated:
                        gpu.run(**arguments)
                        self.assertEqual(json.loads((root/'result').read_text()), payload)
                        self.assertEqual((root/'evidence').read_bytes(), b'original-private-evidence')
                    else:
                        with self.assertRaises(gpu.VerifierGpuError) as caught:
                            gpu.run(**arguments)
                        self.assertEqual(caught.exception.private_evidence['remote_failure']['code'],
                                         'trusted_evaluation_failure')
                    delete.assert_called_once()
                    phases = {row['phase']:row for row in snapshots[-1]['events']}
                    self.assertEqual(phases['cleanup']['status'], 'completed')
                    self.assertEqual(phases['delete']['runner_termination'],
                                     'confirmed' if terminated else 'unconfirmed')


class SharedVerifierIntegrationTests(unittest.TestCase):
    """The run-failure and trusted-fault outcomes through the shared verifier."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SHARED/'verifier'))
        cls.adapter = load(SHARED/'verifier/fcs2_verifier.py', 'flow_shared_verifier')

    def invoke(self, evaluate, root, retained):
        adapter = self.adapter
        payload, dataset = root/'payload.tar', root/'dataset.tar'
        payload.write_bytes(b'payload');dataset.write_bytes(b'dataset')
        scope = mock.Mock(ok=True, tree=root, changes=(), metrics={})
        edit_scope = mock.Mock(check=mock.Mock(return_value=scope), CATEGORIES={})
        with (
            mock.patch.dict(os.environ, {'FRONTIER_VERIFIER_SCRATCH':str(root)}),
            mock.patch.object(adapter, 'extract_payload'),
            mock.patch.object(adapter, 'entrypoint_path', return_value=root/'payload.tar'),
            mock.patch.object(adapter, 'load_edit_scope', return_value=(edit_scope, None)),
            mock.patch.object(adapter, 'stage_task', return_value=BUNDLE),
            mock.patch.object(adapter, 'load_evaluator', return_value=(evaluate, None)),
            mock.patch.object(adapter, 'write_evidence', side_effect=lambda path, files: retained.append(files)),
        ):
            return adapter.verify(bundle=BUNDLE, entrypoint='payload/solution.patch', payload=payload,
                                  dataset=dataset, mode='eval',result_path=root/'result.json',
                                  evidence_path=root/'evidence.tar.zst')

    def test_failed_run_is_scored_fail_and_keeps_real_reference(self):
        metrics = scoring.failure_metrics(['a','b'], reference={'a':.25,'b':.75}, final=True)
        with tempfile.TemporaryDirectory() as temp:
            retained = []
            result = self.invoke(lambda _: (0,0,'Invalid or incomplete run.',metrics),Path(temp),retained)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['raw_score'], 0)
        self.assertEqual(result['private_metrics']['beats_reference'], 0)
        self.assertEqual(result['private_metrics']['reference_score'], 50)
        self.assertEqual(json.loads(retained[-1]['metrics.json'])['metrics'],metrics)

    def test_trusted_failure_is_unscored_and_evidence_is_written(self):
        def evaluate(_):
            raise scoring.TrustedEvaluationFailure('workload_runtime_unclassified')
        with tempfile.TemporaryDirectory() as temp:
            retained = []
            result = self.invoke(evaluate,Path(temp),retained)
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('raw_score', result)
        self.assertEqual(result['failure']['code'], 'evaluator_error')
        evidence = json.loads(retained[-1]['metrics.json'])['metrics']['evaluator_exception']
        self.assertEqual(evidence['exception_type'], 'TrustedEvaluationFailure')


if __name__ == '__main__':
    unittest.main()
