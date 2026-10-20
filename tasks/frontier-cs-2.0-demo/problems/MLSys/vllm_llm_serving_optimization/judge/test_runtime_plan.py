"""Synthetic allocation checks, not selected scientific or runtime budgets."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'verifier'))
from instance_manifest import expected_groups
from runtime_config import project_task_config
from runtime_plan import FORMAL_ANSWER_SECONDS, group_operations, phase_operations, validate_plan
from test_runtime_config import synthetic_task


def funded_fixture():
    task=synthetic_task()
    task['quota']['verifier']['max_gpu_instances']=4
    task['quota']['verifier']['max_gpu_seconds']=25*2*10+4*20
    task['network']['audit'].update({'max_model_sessions':256,'max_ssh_sessions':1754,
        'max_ssh_bytes_per_direction':10000,
        'ssh_session_budgets':{'agent':512,'verifier_train':900,'verifier_eval':342}})
    phases={mode:{'groups':{group:{'gpu_seconds':10 if mode=='train' else 20}
                          for group in expected_groups(mode)},'cleanup_seconds':1,'capture_seconds':1,'startup_seconds':5,
                  'stop_seconds':1,'max_stdout_bytes':1024,'max_stderr_bytes':1024} for mode in ('train','eval')}
    plan={'schema':'frontier.vllm-runtime-plan/v3','max_payload_bytes':100000,'phases':phases,
          'relay':{'job_bytes':8000,'response_bytes':8000,'receipt_bytes':8000,
                   'batch_timeout_seconds':{'train':80,'eval':280},'correctness_timeout_seconds':{'train':1200,'eval':1200},
                   'receipt_timeout_seconds':60},
          'correctness':{'request_bytes':4000,'output_bytes':8000}}
    config=project_task_config(task)
    config['verifier_quota']['max_train_wall_seconds']=1300
    config['verifier_quota']['eval_timeout_seconds']=1300
    return plan,config


class RuntimePlanTests(unittest.TestCase):
    def test_operations_follow_the_fixed_relay_protocol(self):
        self.assertEqual(group_operations('train','bfcl/public'),9)
        self.assertEqual(group_operations('train','swe/public'),9+3+3*5)
        self.assertEqual(group_operations('eval','swe/hidden'),9+3+3*48)
        self.assertEqual(phase_operations('train'),36)
        self.assertEqual(phase_operations('eval'),2*(9+3+3*48)+2*9)

    def test_complete_funded_plan_and_separate_pools(self):
        plan,config=funded_fixture()
        self.assertIs(validate_plan(plan,config),plan)
        self.assertEqual(FORMAL_ANSWER_SECONDS,18000)
        for target,key,value in (('pools','verifier_train',899),('pools','verifier_eval',phase_operations('eval')-1),('quota','max_gpu_seconds',579)):
            bad=copy.deepcopy(config)
            (bad['audit']['ssh_session_budgets'] if target=='pools' else bad['verifier_quota'])[key]=value
            with self.subTest(target=target,key=key),self.assertRaises(ValueError): validate_plan(plan,bad)

    def test_legacy_plans_and_one_slot_quota_fail_before_create(self):
        plan, config = funded_fixture()
        legacy = copy.deepcopy(plan)
        legacy['schema'] = 'frontier.vllm-runtime-plan/v1'
        with self.assertRaisesRegex(ValueError, 'invalid closed publisher runtime plan'):
            validate_plan(legacy, config)
        insufficient = copy.deepcopy(config)
        insufficient['verifier_quota']['max_gpu_instances'] = 1
        with self.assertRaisesRegex(ValueError, 'four concurrent'):
            validate_plan(plan, insufficient)
        missing_group = copy.deepcopy(plan)
        del missing_group['phases']['eval']['groups']['bfcl/hidden']
        with self.assertRaisesRegex(ValueError, 'explicit positive lifetime'):
            validate_plan(missing_group, config)

    def test_relay_bounds_stay_inside_stream_and_phase_allocation(self):
        plan,config=funded_fixture()
        for change in ('job','response','receipt','batch_timeout','correctness_timeout','short_test','receipt_fit','unknown','clock'):
            bad=copy.deepcopy(plan)
            if change=='job':bad['relay']['job_bytes']=10001
            elif change=='response':bad['relay']['response_bytes']=9000
            elif change=='receipt':bad['relay']['receipt_bytes']=9000
            elif change=='batch_timeout':bad['relay']['batch_timeout_seconds']['train']=1299
            elif change=='correctness_timeout':bad['relay']['correctness_timeout_seconds']['eval']=1299
            elif change=='short_test':bad['relay']['correctness_timeout_seconds']['train']=1199
            elif change=='receipt_fit':bad['correctness']['output_bytes']=8001
            elif change=='unknown':bad['token']='synthetic'
            else:bad['phases']['train']['startup_seconds']=1300
            with self.subTest(change=change),self.assertRaises(ValueError): validate_plan(bad,config)


if __name__=='__main__':unittest.main()
