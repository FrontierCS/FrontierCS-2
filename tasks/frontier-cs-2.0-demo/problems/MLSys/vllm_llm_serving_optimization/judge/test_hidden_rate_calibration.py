"""Complete hidden-rate calibration boundaries on the declared-sidecar topology."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'verifier'))
import hidden_rate_calibration as calibration
from instance_manifest import expected_groups
from test_runtime_plan import funded_fixture
from runtime_plan import phase_operations


def fixture(): return funded_fixture()


class HiddenRateCalibrationTests(unittest.TestCase):
    def test_exact_four_group_caps_validate(self):
        plan,config=fixture();calibration.validate_calibration(plan,config)
        self.assertEqual(set(plan['phases']['eval']['groups']),set(expected_groups('eval')))

    def test_rejects_rate_resource_or_profile_drift(self):
        plan,config=fixture();plan['phases']['eval']['groups'].pop('swe/hidden')
        with self.assertRaises(ValueError):calibration.validate_calibration(plan,config)
        plan,config=fixture();config['audit']['ssh_session_budgets']['verifier_eval']=phase_operations('eval')-1
        config['audit']['ssh_session_budgets']['agent']+=1
        config['audit']['max_ssh_sessions']=sum(config['audit']['ssh_session_budgets'].values())
        with self.assertRaises(ValueError):calibration.validate_calibration(plan,config)

    def test_eval_invokes_fixed_hidden_rates_and_never_returns_acceptance(self):
        plan,config=fixture()
        with tempfile.TemporaryDirectory() as tmp,patch.object(calibration,'run_parallel_phase',return_value={'score':99,'pass':True}) as run:
            result=calibration.execute(plan,config,candidate_patch=b'',output=Path(tmp)/'output',
                environment={},clock_path=Path(tmp)/'clock')
            self.assertFalse(result['acceptance']);self.assertFalse(result['score_publishable'])
            self.assertEqual(run.call_args.args[2],'eval')
            self.assertEqual(run.call_args.kwargs['hidden_jps'],{'swe':1.0,'bfcl':1.25})
            self.assertTrue(run.call_args.kwargs['calibration'])

    def test_platform_result_is_diagnostic_only(self):
        plan,config=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            env={'FRONTIER_VERIFICATION_MODE':'eval','FRONTIER_RESULT_PATH':str(root/'result.json'),
                'FRONTIER_EVIDENCE_PATH':str(root/'evidence.tar.zst'),
                'FRONTIER_DATASET_PATH':str(root/'dataset'),'FRONTIER_PAYLOAD_PATH':str(root/'payload')}
            def complete(*args,**kwargs):
                for group in expected_groups('eval'):
                    directory=kwargs['evidence']/group.replace('/','-');directory.mkdir(parents=True)
                    (directory/'parent-clock.json').write_text('{}')
                    (directory/'parent-cutoffs.json').write_text('{}')
                return {'score':99,'pass':True}
            with patch.object(calibration.diagnostic,'dataset_selector'),patch.object(calibration.diagnostic,'extract_patch',return_value=b''),patch.object(calibration.diagnostic,'run_parallel_phase',side_effect=complete),patch.object(calibration.diagnostic,'write_evidence',side_effect=lambda source,target,maximum:target.write_bytes(b'bounded')):
                self.assertEqual(calibration.platform_wrapper(plan,config,env),0)
            result=json.loads((root/'result.json').read_text())
            self.assertEqual(result['raw_score'],0)
            self.assertEqual(result['metrics'],{'diagnostic_complete':1,'calibration_only':1,'workload_groups':4})
            self.assertNotIn('private_metrics',result)


if __name__=='__main__':unittest.main()
