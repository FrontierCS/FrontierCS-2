import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('qwen_launcher', Path(__file__).with_name('launcher.py'))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LauncherTests(unittest.TestCase):
    def test_thinking_alias_uses_effective_high_and_no_numeric_output_budget(self):
        settings = launch.settings('http://127.0.0.1:8082/v1', 'qwen3.8-max-vl-thinking')
        generation = settings['modelProviders']['openai'][0]['generationConfig']
        self.assertEqual(generation['extra_body'], {
            'temperature': 1, 'max_tokens': None,
            'reasoning_effort': 'high', 'enable_thinking': True})
        self.assertEqual(generation['samplingParams'], {'temperature': 1})
        self.assertEqual(generation['contextWindowSize'], 1000000)
        self.assertNotIn('thinking_budget', generation['extra_body'])
        legacy = launch.settings('http://127.0.0.1:8082/v1', 'qwen3.8-max')
        self.assertEqual(legacy['modelProviders']['openai'][0]['generationConfig']['extra_body'],
                         launch.WIRE_PARAMETERS)

    def test_endpoint_rejects_provider_credentials_dns_and_foreign_ports(self):
        for value in ('https://provider.example/v1', 'http://provider.example:8082/v1',
                      'http://127.0.0.1:8083/v1', 'http://secret@127.0.0.1:8082/v1',
                      'http://127.0.0.1:8082/v1?key=x', 'http://127.0.0.1:8082/v1#x'):
            with self.subTest(value=value), patch.dict(os.environ, {'MODEL_ENDPOINT': value}):
                with self.assertRaises(launch.LaunchError):
                    launch.endpoint('MODEL_ENDPOINT', 8082, '/v1')
        for value in ('http://10.1.2.3:8082/v1', 'http://[::1]:8082/v1/'):
            with patch.dict(os.environ, {'MODEL_ENDPOINT': value}):
                self.assertEqual(launch.endpoint('MODEL_ENDPOINT', 8082, '/v1'), value.rstrip('/'))

    def test_clean_native_settings_and_environment_strip_provider_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'candidate'
            workspace = Path(directory) / 'workspace'
            workspace.mkdir()
            runtime = root / 'qwen-code'
            cli = runtime / 'app/cli-entry.js'
            node = runtime / 'node'
            for path in (cli, node, root / 'skills/frontier-submit/SKILL.md',
                         root / 'skills/frontier-gpu/SKILL.md', root / 'bin/frontier-submit',
                         root / 'bin/frontier-gpu', root / 'bin/frontier-gpu-connect-proxy'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            env = {'PATH': '/usr/bin:/bin', 'FRONTIER_DRIVER_API_URL': 'http://10.1.2.3:8080',
                   'FRONTIER_MODEL_API_BASE_URL': 'http://10.1.2.3:8082/v1',
                   'FRONTIER_MODEL_PROXY_TOKEN': 'scoped-run-token',
                   'FRONTIER_CODEX_MODEL': 'qwen3.8-max',
                   'FRONTIER_DRIVER_TOKEN_FILE': '/var/run/frontier/token',
                   'DASHSCOPE_API_KEY': 'private-provider-secret',
                   'OPENAI_API_KEY': 'private-provider-secret',
                   'QWEN_BASE_URL': 'https://foreign-provider.example',
                   'QWEN_STREAM_MAX_LIFETIME_MS': '1',
                   'OTHER_API_KEY': 'private-provider-secret', 'LD_LIBRARY_PATH': '/foreign'}
            original_is_file = Path.is_file
            def required_file(path):
                return str(path) == '/frontier/task/README.md' or original_is_file(path)
            with patch.multiple(launch, ROOT=root, WORKSPACE=workspace, NODE=node, CLI=cli), \
                    patch.dict(os.environ, env, clear=True), patch.object(Path, 'is_file', required_file):
                child = launch.prepare_environment()
                config_path = Path(child['QWEN_HOME']) / 'settings.json'
                config = json.loads(config_path.read_text())
                self.assertEqual(config_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(Path(child['HOME']).stat().st_mode & 0o777, 0o700)
                self.assertNotIn('private-provider-secret', json.dumps(config))
                self.assertNotIn('private-provider-secret', json.dumps(child))
                self.assertNotIn('LD_LIBRARY_PATH', child)
                self.assertEqual(child['FRONTIER_MODEL_PROXY_TOKEN'], 'scoped-run-token')
                self.assertEqual(child['FRONTIER_DRIVER_TOKEN_FILE'], '/var/run/frontier/token')
                self.assertEqual(child['QWEN_STREAM_MAX_LIFETIME_MS'], '0')
                model = config['modelProviders']['openai'][0]
                self.assertEqual(model['baseUrl'], env['FRONTIER_MODEL_API_BASE_URL'])
                self.assertEqual(model['envKey'], 'FRONTIER_MODEL_PROXY_TOKEN')
                self.assertEqual(model['generationConfig']['extra_body'], launch.WIRE_PARAMETERS)
                self.assertEqual(model['generationConfig']['contextWindowSize'], 1000000)
                self.assertNotIn('streamMaxLifetimeMs', model['generationConfig'])
                self.assertEqual(config['tools']['shell']['defaultTimeoutMs'], 0)
                self.assertEqual(config['mcpServers'], {})
                self.assertFalse(config['memory']['enableManagedAutoMemory'])
                with self.assertRaises(FileExistsError):
                    launch.prepare_environment()

    def test_saved_or_malformed_receipts_are_not_final_acceptance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(launch, 'WORKSPACE', Path(directory)):
            receipt_path = Path(directory) / '.frontier/final-submission.json'
            receipt_path.parent.mkdir()
            receipt = {'schema_version': 'frontier.candidate-submit-result/v1',
                       'operation': 'final', 'state': 'accepted',
                       'payload_id': 'payload_fixture', 'verification_id': 'verification_fixture'}
            for change in ({'operation': 'store-final', 'state': 'stored'}, {'state': 'running'},
                           {'verification_id': None}, {'payload_id': 'foreign'}, {'extra': 1}):
                receipt_path.write_text(json.dumps({**receipt, **change}))
                receipt_path.chmod(0o600)
                self.assertFalse(launch.accepted_final_receipt())
            receipt_path.write_text(json.dumps(receipt))
            self.assertTrue(launch.accepted_final_receipt())
            receipt_path.chmod(0o644)
            self.assertFalse(launch.accepted_final_receipt())
            receipt_path.unlink()
            target = Path(directory) / 'symlink-target'
            target.write_text(json.dumps(receipt)); target.chmod(0o600)
            receipt_path.symlink_to(target)
            self.assertFalse(launch.accepted_final_receipt())

    def test_native_failure_after_accepted_final_does_not_revoke_acceptance(self):
        with patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'command', return_value=['fixture']), \
                patch.object(launch.subprocess, 'run') as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=True):
            run.return_value.returncode = 1
            self.assertEqual(launch.main(), 0)
        with patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'command', return_value=['fixture']), \
                patch.object(launch.subprocess, 'run') as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=False):
            run.return_value.returncode = 0
            self.assertEqual(launch.main(), 2)


if __name__ == '__main__':
    unittest.main()
