"""Offline calibration preflight/cleanup tests; no cloud or model execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'verifier'))
import public_calibration as calibration
from test_runtime_plan import funded_fixture


def fixture():
    runtime, config = funded_fixture()
    config['max_evidence_bytes'] = 1024*1024  # synthetic local capture bound, not publication choice
    config['verifier_quota'].update(max_train_wall_seconds=9000, max_gpu_seconds=7200)
    plan = {'schema': calibration.SCHEMA,
            'inputs': {'payload': {'sha256': 'sha256:' + 'a' * 64, 'bytes': 1},
                       'dataset': {'sha256': 'sha256:' + 'b' * 64, 'bytes': 1}},
            'phase': {'gpu_seconds': 7200, 'cleanup_seconds': 30, 'capture_seconds': 30, 'collector_seconds': 30, 'startup_seconds': 500,
                      'stop_seconds': 10, 'max_stdout_bytes': 1024, 'max_stderr_bytes': 1024},
            'relay': dict(runtime['relay'], batch_timeout_seconds={'train': 3600, 'eval': 3600},
                          correctness_timeout_seconds={'train': 1500, 'eval': 1500}, receipt_timeout_seconds=60),
            'correctness': dict(runtime['correctness'])}
    return plan, config


class PublicCalibrationTests(unittest.TestCase):
    def test_public_complete_scope_and_funding_reject_incomplete_or_hidden(self):
        plan, config = fixture()
        calibration.validate_calibration(plan, config)
        for kind in ('missing', 'legacy', 'sessions', 'timeout', 'extra', 'catalog'):
            p, c = copy.deepcopy(plan), copy.deepcopy(config)
            if kind == 'missing': p.pop('correctness')
            if kind == 'legacy': p['interactive_profiles'] = {}
            if kind == 'sessions': c['audit']['ssh_session_budgets']['verifier_train'] = 26
            if kind == 'timeout': p['relay']['correctness_timeout_seconds']['train'] = 1199
            if kind == 'extra': p['relay']['receipt_timeout_seconds'] = 9000
            if kind == 'catalog': c['sidecars'].pop('swe-004-check')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                calibration.validate_calibration(p, c)

    def test_insufficient_owner_grant_headroom_cannot_create_parent(self):
        plan, config = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(calibration, 'DriverGPU') as driver:
            receipt = Path(tmp) / 'clock'; receipt.write_text('synthetic clock receipt')
            with self.assertRaises(ValueError):
                calibration.execute(plan, config, output=Path(tmp)/'result', owner_deadline=time.monotonic()+3600,
                                    owner_clock_receipt=receipt, environment={}, clock_path=Path(tmp)/'clock-marker')
            driver.assert_not_called()

    def test_failure_keeps_operator_receipt_and_cleans_session(self):
        plan, config = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(calibration, 'DriverGPU') as driver, \
             patch.object(calibration, 'GPUSession') as session, patch.object(calibration, 'for_phase') as owned, \
             patch.object(calibration, 'run_phase', side_effect=RuntimeError('synthetic')):
            driver.return_value.verification_id = 'synthetic'
            session.return_value.capture_deadline = None
            receipt = Path(tmp)/'clock'; receipt.write_text('synthetic clock receipt')
            output = Path(tmp)/'result'
            with self.assertRaises(RuntimeError):
                calibration.execute(plan, config, output=output, owner_deadline=time.monotonic()+8000,
                                    owner_clock_receipt=receipt, environment={'FRONTIER_DRIVER_INTERNAL_URL':'http://fixture',
                                        'FRONTIER_VERIFICATION_ID':'synthetic', 'FRONTIER_VERIFIER_TOKEN':'synthetic'}, clock_path=Path(tmp)/'clock-marker')
            session.return_value.__exit__.assert_called_once()
            self.assertIs(session.call_args.kwargs['provisioning_owner'], owned.return_value)
            record = json.loads((output/'result.json').read_text())
            self.assertEqual(record['status'], 'runtime_failure')
            self.assertFalse(record['acceptance'])
            self.assertFalse(record['score_ready'])
            self.assertNotIn('operator_result', record)

    def test_exclusive_phase_lock_and_cleanup_only_resume(self):
        plan, config = fixture()
        env = {'FRONTIER_DRIVER_INTERNAL_URL':'http://fixture','FRONTIER_VERIFICATION_ID':'synthetic',
               'FRONTIER_VERIFIER_TOKEN':'synthetic'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(calibration, 'DriverGPU') as driver, \
             patch.object(calibration, 'run_phase') as run:
            marker = Path(tmp)/'marker'
            with calibration.execution_lock(marker), self.assertRaises(BlockingIOError):
                calibration.execute(plan,config,output=Path(tmp)/'blocked',environment=env,clock_path=marker)
            driver.assert_not_called()
            key = calibration.hashlib.sha256(b'synthetic\0public-calibration').hexdigest()
            clock = calibration.begin({'driver_origin':'http://fixture','verification_id':'synthetic',
                'request_key':key,'request':{'gpu_type':'H100','gpu_count':1,'max_duration_seconds':7200},
                'config_sha256':calibration.canonical_digest(config),'plan_sha256':calibration.canonical_digest(plan)},
                marker,cap_seconds=7200,cleanup_seconds=30,capture_seconds=60)
            clock.accept_grant('runner_known',7200)
            driver.return_value.request.return_value={'runner_id':'runner_known','state':'terminated'}
            record = calibration.execute(plan,config,output=Path(tmp)/'resume',environment=env,clock_path=marker,resume=True)
            self.assertEqual(record['status'],'cleanup_only')
            driver.return_value.request.assert_called_once_with('DELETE','/runner_known')
            run.assert_not_called()

    def test_capture_failure_signals_failure_and_retains_clock_provenance(self):
        plan, config = fixture()
        env = {'FRONTIER_DRIVER_INTERNAL_URL':'http://fixture','FRONTIER_VERIFICATION_ID':'synthetic',
               'FRONTIER_VERIFIER_TOKEN':'synthetic'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(calibration, 'DriverGPU') as driver, \
             patch.object(calibration, 'GPUSession') as session, patch.object(calibration, 'for_phase'), \
             patch.object(calibration, 'run_phase',return_value={'score':0}), \
             patch.object(calibration.subprocess,'run',side_effect=TimeoutError('synthetic capture')):
            session.return_value.capture_deadline = None
            output=Path(tmp)/'output'
            with self.assertRaises(RuntimeError):
                calibration.execute(plan,config,output=output,environment=env,clock_path=Path(tmp)/'marker')
            record=json.loads((output/'result.json').read_text())
            self.assertEqual(record['status'],'capture_failure')
            self.assertNotIn('operator_result',record)
            self.assertTrue((output/'evidence/clock-marker.json').is_file())
            self.assertNotIn('synthetic-token',(output/'evidence/clock-marker.json').read_text())



    def test_platform_wrapper_binds_inputs_and_retains_incomplete_diagnostic(self):
        plan, config = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, dataset = root/'payload.tar', root/'train.json'
            payload.write_bytes(b'p'); dataset.write_bytes(b'd')
            plan['inputs'] = {
                'payload': {'sha256': 'sha256:' + calibration.hashlib.sha256(b'p').hexdigest(), 'bytes': 1},
                'dataset': {'sha256': 'sha256:' + calibration.hashlib.sha256(b'd').hexdigest(), 'bytes': 1}}
            env = {'FRONTIER_VERIFICATION_MODE': 'train', 'FRONTIER_PAYLOAD_PATH': str(payload),
                   'FRONTIER_DATASET_PATH': str(dataset), 'FRONTIER_RESULT_PATH': str(root/'result.json'),
                   'FRONTIER_EVIDENCE_PATH': str(root/'evidence.tar.zst')}
            def captured_failure(*args, **kwargs):
                output = kwargs['output']; output.mkdir()
                (output/'result.json').write_text(json.dumps({'status': 'runtime_failure',
                    'local_capture_before_cutoff': True,
                    'collector_deadline_monotonic': time.monotonic()+10}))
                (output/'evidence.tar.zst').write_bytes(b'bounded')
                raise RuntimeError('synthetic')
            with patch.object(calibration, 'execute', side_effect=captured_failure):
                self.assertEqual(calibration.platform_wrapper(plan, config, env), 0)
            result = json.loads((root/'result.json').read_text())
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['raw_score'], 0.0)
            self.assertEqual(result['metrics']['diagnostic_complete'], 0)
            self.assertTrue((root/'evidence.tar.zst').is_file())

    def test_platform_wrapper_rejects_unbound_input_before_driver(self):
        plan, config = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(calibration, 'execute') as execute:
            root = Path(tmp); payload, dataset = root/'payload.tar', root/'train.json'
            payload.write_bytes(b'x'); dataset.write_bytes(b'y')
            env = {'FRONTIER_VERIFICATION_MODE': 'train', 'FRONTIER_PAYLOAD_PATH': str(payload),
                   'FRONTIER_DATASET_PATH': str(dataset), 'FRONTIER_RESULT_PATH': str(root/'result.json'),
                   'FRONTIER_EVIDENCE_PATH': str(root/'evidence.tar.zst')}
            calibration.platform_wrapper(plan, config, env)
            self.assertEqual(json.loads((root/'result.json').read_text())['status'], 'failed')
            execute.assert_not_called()
if __name__ == '__main__': unittest.main()
