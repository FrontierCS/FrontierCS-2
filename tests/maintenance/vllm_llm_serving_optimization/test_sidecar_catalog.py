"""Fixed sidecar naming and the complete rendered catalog; no publication choice."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from instance_manifest import expected_groups
from sidecar_catalog import (CHECK_PORT, MAX_SIDECARS_PER_PARENT, SCHEDULER, TESTBED_PORT, check_name,
                             expected_sidecars, image_role, instance_index, phases_for_instance,
                             render_profiles, testbed_name)
from test_runtime_config import SYNTHETIC_LIMITS


class SidecarCatalogTests(unittest.TestCase):
    def test_worker_imports_exclude_target_cwd_without_changing_target_python(self):
        profiles = render_profiles(SYNTHETIC_LIMITS)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trusted, target = root / 'trusted', root / 'target'
            package = trusted / 'serving_eval'
            package.mkdir(parents=True)
            target.mkdir()
            (package / '__init__.py').write_text('')
            (trusted / 'requests.py').write_text("origin = 'trusted-worker'\n")
            (target / 'requests.py').write_text("origin = 'target-repository'\n")
            probe = (
                'import json, os, requests, subprocess, sys\n'
                'child = subprocess.check_output([sys.executable, "-c", '
                '"import requests; print(requests.origin)"], text=True)\n'
                'print(json.dumps(dict(worker=requests.origin, target=child.strip(), '
                'safe_path=sys.flags.safe_path, '
                'environment=os.environ.get("PYTHONSAFEPATH"))))\n'
            )
            environment = {**os.environ, 'PYTHONPATH': str(trusted)}
            environment.pop('PYTHONSAFEPATH', None)
            for name, module in ((testbed_name(0, 'candidate'), 'swe_tool_service'),
                                 (check_name(0), 'swe_correctness_service')):
                with self.subTest(module=module):
                    (package / (module + '.py')).write_text(probe)
                    args = profiles[name]['args']
                    self.assertEqual(args[1:4], ['-P', '-m', 'serving_eval.' + module])
                    execution = subprocess.run([sys.executable, *args[1:4]], cwd=target,
                                               env=environment, capture_output=True, text=True,
                                               check=True, timeout=30)
                    observed = json.loads(execution.stdout)
                    self.assertEqual(observed, {'worker': 'trusted-worker',
                                                'target': 'target-repository',
                                                'safe_path': True, 'environment': None})

    def test_counts_fit_the_provider_cap_and_follow_the_frozen_order(self):
        eval_names = expected_sidecars('eval')
        train_names = expected_sidecars('train')
        self.assertEqual(len(eval_names), 193)
        self.assertEqual(len(train_names), 21)
        self.assertLessEqual(len(eval_names), MAX_SIDECARS_PER_PARENT)
        self.assertEqual(eval_names[0], SCHEDULER)
        self.assertTrue(set(train_names) < set(eval_names))
        ids = expected_groups('eval')['swe/public']
        self.assertEqual(instance_index(ids[7]), 7)
        self.assertEqual(testbed_name(7, 'reference'), 'swe-007-reference')
        self.assertEqual(check_name(49), 'swe-049-check')
        self.assertEqual(image_role(3), 'swe-003')
        self.assertEqual(phases_for_instance(ids[0]), ['train', 'eval'])
        self.assertEqual(phases_for_instance(ids[-1]), ['eval'])
        with self.assertRaises(ValueError):
            testbed_name(0, 'other')
        with self.assertRaises(ValueError):
            instance_index('unknown__instance-1')

    def test_removed_cases_do_not_renumber_remaining_sidecars(self):
        ids = expected_groups('eval')['swe/public']
        indices = tuple(instance_index(ident) for ident in ids)
        self.assertEqual(indices, tuple(index for index in range(50) if index not in (18, 39)))
        self.assertEqual(instance_index('sympy__sympy-21847'), 19)
        self.assertEqual(instance_index('sphinx-doc__sphinx-7738'), 40)
        self.assertEqual(instance_index('astropy__astropy-6938'), 49)
        for removed in ('sympy__sympy-13177', 'pytest-dev__pytest-7168'):
            self.assertNotIn(removed, ids)
            with self.assertRaises(ValueError):
                instance_index(removed)
        profiles = render_profiles(SYNTHETIC_LIMITS)
        for index in (18, 39):
            self.assertNotIn(check_name(index), profiles)
            for variant in ('vanilla', 'reference', 'candidate'):
                self.assertNotIn(testbed_name(index, variant), profiles)

    def test_rendered_profiles_fix_every_command_and_share_one_image_per_instance(self):
        profiles = render_profiles(SYNTHETIC_LIMITS)
        self.assertEqual(set(profiles), set(expected_sidecars('eval')))
        self.assertEqual(profiles[SCHEDULER]['attestation_key'], True)
        self.assertEqual(profiles[SCHEDULER]['phases'], ['train', 'eval'])
        self.assertEqual(len({profile['image'] for profile in profiles.values()}), 49)
        for ident in expected_groups('eval')['swe/public']:
            index = instance_index(ident)
            for variant in ('vanilla', 'reference', 'candidate'):
                profile = profiles[testbed_name(index, variant)]
                self.assertEqual(profile['image'], image_role(index))
                self.assertIn('--instance-id', profile['args'])
                self.assertEqual(profile['args'][profile['args'].index('--instance-id') + 1], ident)
                self.assertEqual(profile['args'][profile['args'].index('--port') + 1], str(TESTBED_PORT))
                self.assertNotIn('attestation_key', profile)
            check = profiles[check_name(index)]
            self.assertEqual(check['args'][check['args'].index('--port') + 1], str(CHECK_PORT))
            self.assertEqual(check['phases'], phases_for_instance(ident))
        with self.assertRaises(ValueError):
            render_profiles({**SYNTHETIC_LIMITS, 'tool_output_bytes': 0})
        with self.assertRaises(ValueError):
            render_profiles({k: v for k, v in SYNTHETIC_LIMITS.items() if k != 'worker_output_bytes'})


if __name__ == '__main__':
    unittest.main()
