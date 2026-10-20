"""Synthetic publisher-field binding tests; values are not task budget choices."""
import copy
import importlib.util
from pathlib import Path
import unittest

from sidecar_catalog import SCHEDULER, check_name, image_role, instance_index, render_profiles, testbed_name

spec = importlib.util.spec_from_file_location('runtime_config', Path(__file__).parents[1] / 'verifier/runtime_config.py')
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

SYNTHETIC_LIMITS = {'scheduler_request_bytes': 8_000_000, 'scheduler_response_bytes': 8_000_000,
                    'tool_request_bytes': 65536, 'tool_output_bytes': 1_000_000,
                    'worker_request_bytes': 1_000_000, 'worker_output_bytes': 4_000_000}


def synthetic_catalog():
    """TaskConfig-shaped catalog: the policy's image roles become tag/digest pairs."""
    catalog = {}
    for name, profile in render_profiles(SYNTHETIC_LIMITS).items():
        role = profile['image']
        digest = 'sha256:' + (f'{abs(hash(role)) % 256:02x}' * 32)
        catalog[name] = {**profile, 'image': {'tag': 'synthetic-' + role, 'digest': digest}}
    return catalog


def synthetic_task():
    return {'schema_version': 'frontier.task-config/v1',
            'images': {'verifier': {'reference': 'cpu-image', 'digest': 'sha256:' + '2' * 64,
                                    'gpu_tag': 'private-gpu', 'gpu_digest': 'sha256:' + '3' * 64},
                       'gpu_parent_resources': {'verifier': {'cpu_millis': 16000, 'memory_bytes': 64 * 1024 ** 3}},
                       'gpu_sidecars': synthetic_catalog()},
            'quota': {'wall_time_seconds': 18000, 'verifier': {'max_train_calls': 25,
                      'max_train_wall_seconds': 90, 'max_gpu_instances': 1,
                      'max_gpu_seconds': 600, 'eval_timeout_seconds': 300}},
            'network': {'audit': {'max_model_sessions': 256, 'max_ssh_sessions': 1754,
                                 'max_ssh_bytes_per_direction': 64,
                                 'max_bundle_bytes': 2048,
                                 'ssh_session_budgets': {'agent': 512, 'verifier_train': 900, 'verifier_eval': 342}}},
            'verifier': {'max_evidence_bytes': 512},
            'datasets': {'train': {'version': 'changes-with-publication'}}}


class RuntimeConfigTests(unittest.TestCase):
    def test_projection_binds_runtime_but_omits_every_image_digest(self):
        task = synthetic_task()
        projected = config.project_task_config(task)
        self.assertEqual(len(projected['sidecars']), 193)
        self.assertNotIn('image', projected['sidecars'][SCHEDULER])
        self.assertEqual(projected['sidecars'][SCHEDULER]['attestation_key'], True)
        self.assertNotIn('attestation_key', projected['sidecars'][check_name(0)])
        task['images']['verifier']['digest'] = 'sha256:' + '4' * 64
        task['images']['verifier']['gpu_digest'] = 'sha256:' + '5' * 64
        task['images']['gpu_sidecars'][check_name(3)]['image']['digest'] = 'sha256:' + '6' * 64
        task['datasets']['train']['version'] = 'new-object'
        task['quota']['wall_time_seconds'] = 1800
        config.verify_projection(projected, task)

    def test_policy_shape_projects_identically_to_task_config(self):
        task = synthetic_task()
        policy = {'schema_version': 'frontier.frontiersmith-task-policy/v1',
                  'sidecars': render_profiles(SYNTHETIC_LIMITS),
                  'gpu_parent_resources': task['images']['gpu_parent_resources'],
                  'quota': task['quota'], 'network': task['network'], 'verifier': task['verifier']}
        self.assertEqual(config.project_task_config(policy), config.project_task_config(task))
        self.assertEqual(policy['sidecars'][check_name(7)]['image'], image_role(7))

    def test_publication_rejects_profile_or_quota_drift(self):
        task = synthetic_task()
        projected = config.project_task_config(task)
        for changed in ('args', 'quota', 'phases', 'resources'):
            other = copy.deepcopy(task)
            if changed == 'args':
                other['images']['gpu_sidecars'][testbed_name(1, 'vanilla')]['args'].append('--changed')
            elif changed == 'quota':
                other['quota']['verifier']['eval_timeout_seconds'] += 1
            elif changed == 'phases':
                other['images']['gpu_sidecars'][SCHEDULER]['phases'] = ['eval']
            else:
                other['images']['gpu_parent_resources']['verifier']['cpu_millis'] += 1
            with self.subTest(changed=changed), self.assertRaises(config.RuntimeConfigurationError):
                config.verify_projection(projected, other)

    def test_catalog_must_match_the_fixed_declared_sidecars_exactly(self):
        for change in ('missing', 'extra', 'two_keys', 'no_key', 'wrong_port', 'wrong_instance',
                       'wrong_phases', 'agent_phase', 'bad_name', 'invoke'):
            task = synthetic_task()
            catalog = task['images']['gpu_sidecars']
            if change == 'missing':
                catalog.pop(check_name(49))
            elif change == 'extra':
                catalog['swe-050-check'] = copy.deepcopy(catalog[check_name(49)])
            elif change == 'two_keys':
                catalog[check_name(0)]['attestation_key'] = True
            elif change == 'no_key':
                del catalog[SCHEDULER]['attestation_key']
            elif change == 'wrong_port':
                args = catalog[testbed_name(2, 'candidate')]['args']
                args[args.index('--port') + 1] = '8003'
            elif change == 'wrong_instance':
                ids = sorted(instance_index(i) for i in [])
                args = catalog[check_name(4)]['args']
                args[args.index('--instance-id') + 1] = catalog[check_name(5)]['args'][
                    catalog[check_name(5)]['args'].index('--instance-id') + 1]
            elif change == 'wrong_phases':
                catalog[check_name(0)]['phases'] = ['eval']
            elif change == 'agent_phase':
                catalog[SCHEDULER]['phases'] = ['agent', 'train', 'eval']
            elif change == 'bad_name':
                catalog['Main'] = catalog.pop(check_name(0))
            else:
                catalog[check_name(0)]['invoke'] = {'args': ['x'], 'workdir': '/'}
            with self.subTest(change=change), self.assertRaises(config.RuntimeConfigurationError):
                config.project_task_config(task)

    def test_parent_reservation_must_admit_every_declared_sidecar(self):
        task = synthetic_task()
        task['images']['gpu_parent_resources']['verifier']['cpu_millis'] = 193 * 32 - 1
        with self.assertRaises(config.RuntimeConfigurationError):
            config.project_task_config(task)
        task = synthetic_task()
        task['images']['gpu_parent_resources']['verifier']['memory_bytes'] = 193 * 32 * 1024 * 1024
        config.project_task_config(task)
        task['images']['gpu_parent_resources']['verifier']['memory_bytes'] -= 1024 * 1024
        with self.assertRaises(config.RuntimeConfigurationError):
            config.project_task_config(task)
        task['images']['gpu_parent_resources']['verifier']['memory_bytes'] += 1
        with self.assertRaises(config.RuntimeConfigurationError):
            config.project_task_config(task)

    def test_separate_phase_requests_never_inherit_agent_wall_time(self):
        projected = config.project_task_config(synthetic_task())
        train = config.phase_limits(projected, 'train', requested_gpu_seconds=90)
        final = config.phase_limits(projected, 'eval', requested_gpu_seconds=300)
        self.assertEqual((train['timeout_seconds'], final['timeout_seconds']), (90, 300))
        self.assertEqual((train['ssh_sessions'], final['ssh_sessions']), (900, 342))
        for phase, duration in (('train', 91), ('eval', 301), ('eval', True), ('eval', 0)):
            with self.assertRaises(config.RuntimeConfigurationError):
                config.phase_limits(projected, phase, requested_gpu_seconds=duration)
        projected['audit']['ssh_session_budgets']['verifier_eval'] = 0
        with self.assertRaises(config.RuntimeConfigurationError):
            config.phase_limits(projected, 'eval', requested_gpu_seconds=1)

    def test_missing_or_conflicting_publisher_budgets_have_no_defaults(self):
        task = synthetic_task()
        del task['images']['gpu_parent_resources']
        with self.assertRaises(config.RuntimeConfigurationError):
            config.project_task_config(task)
        task = synthetic_task()
        task['network']['audit']['max_ssh_sessions'] = 1753
        with self.assertRaises(config.RuntimeConfigurationError):
            config.project_task_config(task)
        task = synthetic_task()
        task['network']['audit']['max_model_sessions'] = 2029
        with self.assertRaises(config.RuntimeConfigurationError):
            config.project_task_config(task)
        task = synthetic_task()
        task['images']['gpu_parent_resources']['verifier']['memory_bytes'] += 1
        with self.assertRaises(config.RuntimeConfigurationError):
            config.project_task_config(task)


if __name__ == '__main__':
    unittest.main()
