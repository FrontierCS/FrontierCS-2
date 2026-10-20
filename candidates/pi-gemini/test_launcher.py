import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pi_launcher', Path(__file__).with_name('launcher.py'))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LauncherTests(unittest.TestCase):
    def test_model_uses_gateway_chat_completions_and_highest_gemini_thinking(self):
        config = launch.models('http://10.1.2.3:8082/v1', 'gemini-4-argon')
        provider = config['providers']['frontier']
        self.assertEqual(provider['baseUrl'], 'http://10.1.2.3:8082/v1')
        self.assertEqual(provider['api'], 'openai-completions')
        self.assertEqual(provider['apiKey'], '$FRONTIER_MODEL_PROXY_TOKEN')
        model = provider['models'][0]
        self.assertEqual(model['id'], 'gemini-4-argon')
        self.assertTrue(model['reasoning'])
        self.assertEqual(model['thinkingLevelMap']['high'], 'high')
        self.assertEqual([level for level, value in model['thinkingLevelMap'].items() if value],
                         ['low', 'medium', 'high'])
        self.assertEqual(model['contextWindow'], 1048576)
        self.assertEqual(model['maxTokens'], 1048576)
        self.assertFalse(model['compat']['supportsStore'])
        self.assertFalse(model['compat']['supportsDeveloperRole'])
        self.assertEqual(launch.settings('gemini-4-argon')['defaultThinkingLevel'], 'high')

    def test_settings_disable_extensions_network_and_project_trust(self):
        config = launch.settings('gemini-4-argon')
        self.assertEqual(config['defaultProjectTrust'], 'never')
        self.assertEqual(config['defaultTools'], ['read', 'bash', 'edit', 'write'])
        self.assertEqual(set(config['extensions']), {
            '-builtin:mcp', '-builtin:llama.cpp', '-builtin:codemode', '-builtin:tool-search'})
        self.assertEqual(config['packages'], [])
        self.assertFalse(config['enableInstallTelemetry'])
        self.assertEqual(config['httpIdleTimeoutMs'], 3600000)
        self.assertEqual(config['retry']['provider']['maxRetries'], 0)

    def test_command_is_headless_json_offline_without_mcp_or_discovery(self):
        with patch.dict(os.environ, {'FRONTIER_CODEX_MODEL': 'gemini-4-argon'}):
            argv = launch.command()
        for flag in ('--offline', '--no-approve', '--no-extensions', '--no-mcp',
                     '--no-skills', '--no-prompt-templates', '--no-themes'):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index('--mode') + 1], 'json')
        self.assertEqual(argv[argv.index('--model') + 1], 'gemini-4-argon')
        self.assertEqual(argv[argv.index('--provider') + 1], 'frontier')
        self.assertEqual(argv[argv.index('--thinking') + 1], 'high')
        self.assertEqual(argv[argv.index('--tools') + 1], 'read,bash,edit,write')
        self.assertEqual(argv[-2:], ['--', launch.PROMPT])
        self.assertNotIn('--print-timeout', argv)

    def test_endpoint_rejects_provider_credentials_dns_and_foreign_ports(self):
        for value in ('https://generativelanguage.googleapis.com/v1beta',
                      'http://provider.example:8082/v1', 'http://127.0.0.1:8083/v1',
                      'http://secret@127.0.0.1:8082/v1', 'http://127.0.0.1:8082/v1?key=x',
                      'http://127.0.0.1:8082/gemini/v1beta', 'http://127.0.0.1:8082/v1#x'):
            with self.subTest(value=value), patch.dict(os.environ, {'MODEL_ENDPOINT': value}):
                with self.assertRaises(launch.LaunchError):
                    launch.endpoint('MODEL_ENDPOINT', 8082, '/v1')
        for value in ('http://10.1.2.3:8082/v1', 'http://[::1]:8082/v1/'):
            with patch.dict(os.environ, {'MODEL_ENDPOINT': value}):
                self.assertEqual(launch.endpoint('MODEL_ENDPOINT', 8082, '/v1'), value.rstrip('/'))

    def test_clean_native_config_and_environment_strip_provider_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'candidate'
            workspace = Path(directory) / 'workspace'
            workspace.mkdir()
            runtime = root / 'pi'
            cli = runtime / 'app/cli.js'
            node = runtime / 'node'
            for path in (cli, node, root / 'skills/frontier-submit/SKILL.md',
                         root / 'skills/frontier-gpu/SKILL.md', root / 'bin/frontier-submit',
                         root / 'bin/frontier-gpu', root / 'bin/frontier-gpu-connect-proxy'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            env = {'PATH': '/usr/bin:/bin', 'FRONTIER_DRIVER_API_URL': 'http://10.1.2.3:8080',
                   'FRONTIER_MODEL_API_BASE_URL': 'http://10.1.2.3:8082/v1',
                   'FRONTIER_MODEL_PROXY_TOKEN': 'scoped-run-token',
                   'FRONTIER_CODEX_MODEL': 'gemini-4-argon',
                   'FRONTIER_DRIVER_TOKEN_FILE': '/var/run/frontier/token',
                   'GEMINI_API_KEY': 'private-provider-secret',
                   'GOOGLE_API_KEY': 'private-provider-secret',
                   'GOOGLE_GEMINI_BASE_URL': 'https://foreign-provider.example',
                   'GOOGLE_APPLICATION_CREDENTIALS': '/foreign/adc.json',
                   'OPENAI_API_KEY': 'private-provider-secret',
                   'PI_CODING_AGENT_DIR': '/foreign/agent', 'PI_OFFLINE': '0',
                   'OTHER_API_KEY': 'private-provider-secret', 'LD_LIBRARY_PATH': '/foreign'}
            original_is_file = Path.is_file
            def required_file(path):
                return str(path) == '/frontier/task/README.md' or original_is_file(path)
            with patch.multiple(launch, ROOT=root, WORKSPACE=workspace, NODE=node, CLI=cli), \
                    patch.dict(os.environ, env, clear=True), patch.object(Path, 'is_file', required_file):
                child = launch.prepare_environment()
                agent = Path(child['PI_CODING_AGENT_DIR'])
                self.assertEqual(agent, workspace / '.frontier-pi-home/agent')
                for name in ('models.json', 'settings.json'):
                    path = agent / name
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                    text = path.read_text()
                    self.assertNotIn('private-provider-secret', text)
                    self.assertNotIn('scoped-run-token', text)
                self.assertFalse((agent / 'auth.json').exists())
                self.assertEqual(Path(child['HOME']).stat().st_mode & 0o777, 0o700)
                self.assertNotIn('private-provider-secret', json.dumps(child))
                for name in ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_GEMINI_BASE_URL',
                             'GOOGLE_APPLICATION_CREDENTIALS', 'LD_LIBRARY_PATH'):
                    self.assertNotIn(name, child)
                self.assertEqual(child['PI_OFFLINE'], '1')
                self.assertEqual(child['PI_TELEMETRY'], '0')
                self.assertEqual(child['PI_SKIP_VERSION_CHECK'], '1')
                self.assertEqual(child['FRONTIER_MODEL_PROXY_TOKEN'], 'scoped-run-token')
                self.assertEqual(child['FRONTIER_DRIVER_TOKEN_FILE'], '/var/run/frontier/token')
                self.assertTrue(child['PATH'].startswith(str(root / 'bin') + ':'))
                models = json.loads((agent / 'models.json').read_text())
                self.assertEqual(models['providers']['frontier']['baseUrl'],
                                 env['FRONTIER_MODEL_API_BASE_URL'])
                with self.assertRaises(FileExistsError):
                    launch.prepare_environment()

    def test_rejects_missing_alias_token_or_split_hosts(self):
        base = {'FRONTIER_DRIVER_API_URL': 'http://10.1.2.3:8080',
                'FRONTIER_MODEL_API_BASE_URL': 'http://10.1.2.3:8082/v1',
                'FRONTIER_MODEL_PROXY_TOKEN': 'scoped', 'FRONTIER_CODEX_MODEL': 'gemini-4-argon',
                'FRONTIER_DRIVER_TOKEN_FILE': '/var/run/frontier/token'}
        cases = ({'FRONTIER_CODEX_MODEL': ''}, {'FRONTIER_CODEX_MODEL': 'google/gemini-4-argon'},
                 {'FRONTIER_MODEL_PROXY_TOKEN': ''},
                 {'FRONTIER_MODEL_API_BASE_URL': 'http://10.9.9.9:8082/v1'},
                 {'FRONTIER_DRIVER_TOKEN_FILE': '/tmp/token'})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, 'is_file', return_value=True), \
                    patch.object(launch, 'WORKSPACE', Path(directory)):
                for change in cases:
                    with self.subTest(change=change), \
                            patch.dict(os.environ, {**base, **change}, clear=True):
                        with self.assertRaises(launch.LaunchError):
                            launch.prepare_environment()
            self.assertFalse((Path(directory) / '.frontier-pi-home').exists())

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

    def test_final_receipt_decides_success_and_stdin_is_closed(self):
        with patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'command', return_value=['fixture']), \
                patch.object(launch.subprocess, 'run') as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=True):
            run.return_value.returncode = 1
            self.assertEqual(launch.main(), 0)
            self.assertIs(run.call_args.kwargs['stdin'], subprocess.DEVNULL)
        with patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'command', return_value=['fixture']), \
                patch.object(launch.subprocess, 'run') as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=False):
            run.return_value.returncode = 0
            self.assertEqual(launch.main(), 2)
            run.return_value.returncode = 7
            self.assertEqual(launch.main(), 7)


if __name__ == '__main__':
    unittest.main()
