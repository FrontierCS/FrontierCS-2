"""All fixed SWE images must prove their original setup works without an index."""
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization'
sys.path.insert(0, str(TASK / 'judge'))
from serving_eval import swe_offline_setup as setup
from sidecar_catalog import instance_index


class OfflineSetupTests(unittest.TestCase):
    def test_requests_missing_plugins_use_only_original_pins_and_isolated_build_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'requirements.txt').write_text('\n'.join(setup.REQUESTS_ORIGINAL_TEST_PINS) + '\n')
            with patch.object(setup.subprocess, 'run') as run:
                setup.prepare_original_test_dependencies('unrelated', '/target/python', root, root / 'wheels')
                run.assert_not_called()
                setup.prepare_original_test_dependencies('psf__requests-3362', '/target/python', root, root / 'wheels')
            calls = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0][:3], ['/target/python', '-m', 'venv'])
            self.assertNotEqual(calls[1][0], '/target/python')
            self.assertEqual(calls[1][-2:], ['setuptools==44.0.0', 'wheel==0.38.4'])
            self.assertIn('--no-deps', calls[2])
            self.assertEqual(calls[3][:5], ['/target/python', '-m', 'pip', 'install', '--no-index'])
            self.assertEqual(calls[3][-len(setup.REQUESTS_ORIGINAL_TEST_PINS):], list(setup.REQUESTS_ORIGINAL_TEST_PINS))
            (root / 'requirements.txt').write_text('pytest-httpbin==999\n')
            with patch.object(setup.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'declarations differ'):
                    setup.prepare_original_test_dependencies('psf__requests-3362', '/target/python', root, root / 'wheels')
                run.assert_not_called()

    def test_every_fixed_image_has_instance_specific_offline_build_gate(self):
        rows = json.loads((TASK / 'judge/serving_eval/swe_data/instances.json').read_text())['instances']
        self.assertEqual(len(rows), 48)
        for row in rows:
            index = instance_index(row['instance_id'])
            with self.subTest(instance=row['instance_id']):
                docker = (TASK / f'sidecar-swe-{index:03d}/Dockerfile').read_text()
                self.assertIn('serving_eval.swe_offline_setup --instance-id ' + row['instance_id'], docker)
                self.assertIn('RUN --network=none PYTHONPATH=', docker)
                self.assertIn('--phase verify', docker)
                self.assertIn('FROM prepared AS runtime\n', docker)
                self.assertIn('FROM runtime AS offline-check\n', docker)
                self.assertIn('FROM runtime\nCOPY --from=offline-check --chmod=0444 ', docker)
                self.assertIn('PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/frontier/swe-wheelhouse', docker)
                self.assertIn('COPY --chmod=0444 judge/serving_eval/swe_syntax_probe.py ', docker)
                self.assertLess(docker.index('COPY --chmod=0444 judge/serving_eval/swe_test_worker.py '),
                                docker.index('FROM runtime AS offline-check\n'))
                self.assertIn('RUN chmod -R a-w /opt/frontier/swe-wheelhouse', docker)
                self.assertIn('/opt/frontier/worker-python/bin/python -P -m ', docker)

    def test_upstream_dependency_repairs_restore_only_affected_fixed_sources_after_prepare(self):
        rows = json.loads((TASK / 'judge/serving_eval/swe_data/instances.json').read_text())['instances']
        for row in rows:
            index = instance_index(row['instance_id'])
            docker = (TASK / f'sidecar-swe-{index:03d}/Dockerfile').read_text()
            restore = 'git -c core.fileMode=false checkout --force --detach ' + row['base_commit']
            if index in (7, 25, 26, 40):
                self.assertLess(docker.index('--phase prepare'), docker.index(restore))
                self.assertLess(docker.index(restore), docker.index('chown -R 10001:10001 /testbed'))
            else:
                self.assertNotIn('checkout --force --detach', docker)
            self.assertEqual('ENV PYTEST_ADDOPTS=-rA' in docker, index in (7, 40))

    def test_removed_cases_are_absent_from_active_inputs_and_image_roles(self):
        from serving_eval.accuracy import CorrectnessRuntimeError, _case, load_swe_cases
        from serving_eval.swe_inputs import load_swe_inputs, verify_projection
        rows = load_swe_cases('eval')
        self.assertEqual(len(rows), 48)
        self.assertEqual(len(load_swe_cases('train')), 5)
        self.assertEqual([row['instance_id'] for row in rows],
                         [row['instance_id'] for row in load_swe_inputs('eval')])
        self.assertTrue(verify_projection(TASK / 'judge/serving_eval/swe_data/instances.json'))
        catalog = json.loads((ROOT / 'config/task-catalog.json').read_text())
        entry = next(row for row in catalog['tasks'] if row['name'] == 'vllm_llm_serving_optimization')
        for index, ident in ((18, 'sympy__sympy-13177'), (39, 'pytest-dev__pytest-7168')):
            self.assertFalse((TASK / f'sidecar-swe-{index:03d}').exists())
            self.assertNotIn(f'swe-{index:03d}', entry['sidecar_images'])
            with self.assertRaises(CorrectnessRuntimeError):
                _case(ident)
        self.assertFalse(hasattr(setup, 'known_base_reporter_failure'))

    def test_pyproject_build_requirements_and_legacy_default_are_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(setup.build_requirements(root), ['setuptools>=40.8.0', 'wheel'])
            (root / 'pyproject.toml').write_text('[build-system]\nrequires = ["setuptools>=40.0", "setuptools_scm"]\n')
            self.assertEqual(setup.build_requirements(root), ['setuptools>=40.0', 'setuptools_scm'])

    def test_prepare_uses_target_python_and_same_command_then_proves_offline_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheels = root / 'wheels'
            with patch.object(setup, '_case', return_value={'base_commit': 'fixed'}), \
                    patch.object(setup, 'setup_commands', return_value=['python -m pip install -e .[test]']), \
                    patch.object(setup, 'build_test_request', return_value={'fixed': True}), \
                    patch.object(setup, 'parse_test_receipt') as parsed, \
                    patch('serving_eval.swe_test_worker.PreparedSWEWorker') as worker, \
                    patch.object(setup.subprocess, 'run') as run:
                worker.return_value.run_correctness.return_value = {'execution': 'complete'}
                setup.prepare('fixed-instance', root, wheels)
            calls = run.call_args_list
            self.assertEqual(calls[0].args[0][:3], ['/opt/miniconda3/envs/testbed/bin/python', '-m', 'pip'])
            self.assertEqual(calls[1].args[0], calls[2].args[0])
            self.assertIn('python -m pip install -e .[test]', calls[2].args[0][-1])
            self.assertEqual(calls[2].kwargs['env']['PIP_NO_INDEX'], '1')
            self.assertEqual(calls[2].kwargs['env']['PIP_FIND_LINKS'], str(wheels))
            self.assertTrue(all(call.kwargs['check'] for call in calls))
            self.assertTrue(json.loads((wheels / 'preflight.json').read_text())['canonical_setup_offline_passed'])
            parsed.assert_called_once_with({'fixed': True}, {'execution': 'complete'})

    def test_unparseable_active_case_cannot_leave_success_marker(self):
        from serving_eval.accuracy import CorrectnessRuntimeError
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheels = root / 'wheels'
            wheels.mkdir()
            with patch.object(setup, '_case', return_value={'base_commit': 'fixed'}), \
                    patch.object(setup, 'setup_commands', return_value=['python -m pip install -e .']), \
                    patch.object(setup, 'build_test_request', return_value={'fixed': True}), \
                    patch.object(setup, 'parse_test_receipt', side_effect=CorrectnessRuntimeError('unusable report')), \
                    patch('serving_eval.swe_test_worker.PreparedSWEWorker') as worker, \
                    patch.object(setup.subprocess, 'run'):
                worker.return_value.run_correctness.return_value = {'execution': 'complete'}
                with self.assertRaises(CorrectnessRuntimeError):
                    setup.prepare('fixed-instance', root, wheels, phase='verify')
            self.assertFalse((wheels / 'preflight.json').exists())

    def test_failed_offline_install_cannot_leave_success_marker(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(setup, '_case', return_value={'base_commit': 'fixed'}), \
                    patch.object(setup, 'setup_commands', return_value=['python -m pip install -e .']), \
                    patch.object(setup.subprocess, 'run', side_effect=[None, None, subprocess.CalledProcessError(1, 'pip')]):
                with self.assertRaises(subprocess.CalledProcessError):
                    setup.prepare('fixed-instance', root, root / 'wheels')
            self.assertFalse((root / 'wheels/preflight.json').exists())



if __name__ == '__main__':
    unittest.main()
