"""OCR candidate rejections and trusted faults retain separate native outcomes."""
import deployment_test_guard  # noqa: F401
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get('FRONTIER_FLOW_OPD_EXPORT', str(
    ROOT / 'tasks/frontier-cs-2.0-demo/problems/Machine Learning/flow_opd')))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scoring = load(TASK/'judge/judge/scoring.py', 'flow_failure_scoring')
feedback = load(TASK/'verifier/beat_spec_feedback.py', 'flow_failure_feedback')
with mock.patch.dict(sys.modules, {'beat_spec_feedback': feedback}):
    adapter = load(TASK/'verifier/fcs2_verifier.py', 'flow_failure_native')
gpu = load(TASK/'verifier/fcs2_verifier_gpu.py', 'flow_failure_gpu')


class CandidateRejectionTests(unittest.TestCase):
    def test_rejection_uses_frozen_reference_and_explicit_unmeasured_checks(self):
        metrics = scoring.failure_metrics(['a','b'], reference={'a':.25,'b':.75}, final=True)
        private = feedback.private_metrics(metrics, beat_spec_path=TASK/'judge/beat_spec.json')
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
        public = feedback.public_metrics(metrics, beat_spec_path=TASK/'judge/beat_spec.json')
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


class TrustedFailureTests(unittest.TestCase):
    def evaluator_case(self, case):
        code = '''import pathlib,sys,importlib.util,json,os,tempfile,types
from unittest import mock
p=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(p/'judge'))
s=importlib.util.spec_from_file_location('flow_evaluator',p/'judge/evaluator.py')
e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
os.environ['FRONTIER_SUBMISSION_ROLE']='final'
case=sys.argv[2]
with tempfile.NamedTemporaryFile(mode='w',suffix='.patch') as f:
 f.write('not a patch');f.flush()
 try:
  if case=='qualification':
   with mock.patch.object(e,'qualification',side_effect=e.QualificationError('private details')):r=e.evaluate(f.name)
  elif case=='runtime':
   with mock.patch.object(e,'validate_patch',return_value=(True,'',{})), mock.patch.object(e,'apply_submission_patch',side_effect=OSError('private details')):r=e.evaluate(f.name)
  elif case=='apply_false':
   with mock.patch.object(e,'validate_patch',return_value=(True,'',{})), mock.patch.object(e,'apply_submission_patch',return_value=types.SimpleNamespace(ok=False,cleanup_root=None)):r=e.evaluate(f.name)
  elif case=='pristine':
   with mock.patch.object(e,'check_surface',side_effect=SyntaxError('trusted scaffold invalid')):r=e.evaluate(f.name)
  elif case=='read':r=e.evaluate(f.name+'.missing')
  else:r=e.evaluate(f.name)
  print(json.dumps({'score':r[0],'rows':len(r[3]['beat_spec_workload_passes']),
                    'reference':r[3]['reference_score'],'status':r[3]['candidate_rejection']['status']}))
 except Exception as err:print(json.dumps({'trusted':getattr(err,'trusted_evaluation_failure',None)}))
'''
        return json.loads(subprocess.check_output([sys.executable,'-c',code,str(TASK),case]))

    def test_actual_invalid_patch_returns_eight_unmeasured_fail_checks(self):
        result = self.evaluator_case('patch')
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['rows'], 8)
        self.assertEqual(result['status'], 'invalid_submission')
        self.assertGreater(result['reference'], 0)

    def test_actual_qualification_and_unclassified_runtime_remain_unscored(self):
        self.assertEqual(self.evaluator_case('qualification'),
                         {'trusted':{'reason':'reference_or_qualification_invalid'}})
        self.assertEqual(self.evaluator_case('runtime'),
                         {'trusted':{'reason':'trusted_runtime_unclassified'}})

    def test_patch_io_apply_and_pristine_faults_are_not_candidate_rejections(self):
        for case in ['apply_false','pristine','read']:
            with self.subTest(case=case):
                self.assertEqual(self.evaluator_case(case),
                                 {'trusted':{'reason':'trusted_runtime_unclassified'}})

    def test_reason_enums_remain_unscored_with_private_evidence(self):
        for reason in ['reference_or_qualification_invalid','trusted_runtime_unclassified',
                       'workload_runtime_unclassified','workload_report_missing','evaluator_invocation_invalid']:
            def evaluate(_):
                raise scoring.TrustedEvaluationFailure(reason)
            with self.subTest(reason=reason), self.assertRaises(adapter.VerifierFailure) as caught:
                adapter.run_evaluator(evaluate, Path('submission'), 'eval')
            self.assertEqual(caught.exception.code, 'trusted_evaluation_failure')
            self.assertEqual(caught.exception.private_evidence, {'trusted_failure':{'reason':reason}})

    def test_arbitrary_failure_reason_is_rejected(self):
        for reason in ['secret-value', [], {}, None, 0]:
            def evaluate(_):
                raise scoring.TrustedEvaluationFailure(reason)
            with self.subTest(reason=reason), self.assertRaises(adapter.VerifierFailure) as caught:
                adapter.run_evaluator(evaluate, Path('submission'), 'eval')
            self.assertEqual(caught.exception.code, 'invalid_evaluator_result')
            self.assertNotIn('secret-value', caught.exception.message)

    def test_private_metric_error_retains_schema_only(self):
        with self.assertRaises(adapter.VerifierFailure) as caught:
            adapter._private_metrics({'beats_reference':0,'beat_spec_workload_passes':[],
                                      'reference_score':None,'margin':None}, TASK/'judge/beat_spec.json')
        details = caught.exception.private_evidence['evaluator_validation']
        self.assertEqual(details['workload_rows'], 0)
        self.assertFalse(details['reference_score_finite'])
        self.assertFalse(details['margin_finite'])


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
                    mock.patch.object(gpu, '_config', return_value=('url','verification','token','gpu',7200,1)),
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


class NativeEvidenceIntegrationTests(unittest.TestCase):
    def invoke(self, evaluate, root, retained):
        payload, dataset = root/'payload.tar', root/'dataset.tar'
        payload.write_bytes(b'payload');dataset.write_bytes(b'dataset')
        with (
            mock.patch.dict(os.environ, {'FRONTIER_VERIFIER_SCRATCH':str(root)}),
            mock.patch.object(adapter, 'load_candidate_diagnostic', return_value=None),
            mock.patch.object(adapter, 'extract_payload'),
            mock.patch.object(adapter, 'entrypoint_path', return_value=root/'submission.patch'),
            mock.patch.object(adapter, 'stage_task', return_value=TASK/'judge'),
            mock.patch.object(adapter, 'load_evaluator', return_value=evaluate),
            mock.patch.object(adapter, 'write_evidence', side_effect=lambda path, files: retained.append(files)),
        ):
            return adapter.verify(bundle=TASK, entrypoint='submission.patch', payload=payload,
                                  dataset=dataset, mode='eval',result_path=root/'result.json',
                                  evidence_path=root/'evidence.tar.zst')

    def test_candidate_rejection_is_scored_fail_and_keeps_real_reference(self):
        metrics = scoring.failure_metrics(['a','b'], reference={'a':.25,'b':.75}, final=True)
        with tempfile.TemporaryDirectory() as temp:
            retained = []
            result = self.invoke(lambda _: (0,0,'Rejected',metrics),Path(temp),retained)
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
        self.assertEqual(result['failure']['code'], 'trusted_evaluation_failure')
        self.assertEqual(json.loads(retained[-1]['metrics.json'])['metrics'],
                         {'trusted_failure':{'reason':'workload_runtime_unclassified'}})

    def test_post_evaluator_validation_failure_retains_schema_evidence(self):
        metrics = {'beats_reference':0,'beat_spec_workload_passes':[],
                   'reference_score':None,'margin':None}
        with tempfile.TemporaryDirectory() as temp:
            retained = []
            with self.assertRaises(adapter.VerifierFailure) as caught:
                self.invoke(lambda _: (0,0,'Rejected',metrics),Path(temp),retained)
        self.assertTrue(caught.exception._failure_evidence_written)
        self.assertEqual(caught.exception.code, 'invalid_evaluator_result')
        envelope = json.loads(retained[-1]['metrics.json'])
        self.assertNotIn('raw_score',envelope)
        self.assertEqual(envelope['metrics']['evaluator_validation']['workload_rows'],0)


if __name__ == '__main__':
    unittest.main()
