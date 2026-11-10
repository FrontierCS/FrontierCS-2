import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('opencode_launcher', Path(__file__).with_name('launcher.py'))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LauncherTests(unittest.TestCase):
    def test_provider_is_bundled_openai_compatible_gateway_with_env_token(self):
        config = launch.config('http://10.1.2.3:8082/v1', 'inkling-2-small')
        provider = config['provider']['frontier']
        self.assertEqual(provider['npm'], '@ai-sdk/openai-compatible')
        self.assertEqual(provider['options']['baseURL'], 'http://10.1.2.3:8082/v1')
        self.assertEqual(provider['options']['apiKey'], '{env:FRONTIER_MODEL_PROXY_TOKEN}')
        self.assertIs(provider['options']['timeout'], False)
        self.assertEqual(provider['options']['headerTimeout'], 3600000)
        self.assertEqual(provider['options']['chunkTimeout'], 3600000)
        self.assertEqual(list(provider['models']), ['inkling-2-small'])
        model = provider['models']['inkling-2-small']
        self.assertTrue(model['reasoning'])
        self.assertTrue(model['tool_call'])
        self.assertEqual(model['limit'], {'context': 262144, 'input': 196608, 'output': 262144})
        self.assertEqual(config['model'], 'frontier/inkling-2-small')
        self.assertEqual(config['small_model'], 'frontier/inkling-2-small')
        self.assertEqual(config['enabled_providers'], ['frontier'])
        self.assertEqual(launch.OPENCODE_ENV['OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX'], '262144')

    def test_config_disables_network_features_and_permission_prompts(self):
        config = launch.config('http://10.1.2.3:8082/v1', 'inkling-2-small')
        self.assertIs(config['autoupdate'], False)
        self.assertEqual(config['share'], 'disabled')
        self.assertIs(config['lsp'], False)
        self.assertIs(config['formatter'], False)
        self.assertIs(config['snapshot'], False)
        self.assertEqual(config['mcp'], {})
        self.assertEqual(config['plugin'], [])
        self.assertEqual(config['compaction'], {'auto': True})
        self.assertEqual(config['skills'], {'paths': ['/opt/frontier/candidate/skills']})
        permission = config['permission']
        # OpenCode applies the last matching rule: the allow-all must come first.
        self.assertEqual(list(permission)[0], '*')
        self.assertEqual(permission['*'], 'allow')
        self.assertEqual(permission['external_directory'], 'allow')
        self.assertEqual(permission['webfetch'], 'deny')
        self.assertEqual(permission['websearch'], 'deny')
        for name in ('OPENCODE_DISABLE_AUTOUPDATE', 'OPENCODE_DISABLE_MODELS_FETCH',
                     'OPENCODE_DISABLE_LSP_DOWNLOAD', 'OPENCODE_PURE',
                     'OPENCODE_DISABLE_DEFAULT_PLUGINS', 'OPENCODE_DISABLE_PROJECT_CONFIG',
                     'OPENCODE_DISABLE_CLAUDE_CODE', 'OPENCODE_DISABLE_EXTERNAL_SKILLS'):
            self.assertEqual(launch.OPENCODE_ENV[name], '1')
        self.assertNotIn('OPENCODE_DISABLE_AUTOCOMPACT', launch.OPENCODE_ENV)

    def test_command_is_headless_json_run_with_titled_session(self):
        with patch.dict(os.environ, {'FRONTIER_CODEX_MODEL': 'inkling-2-small'}):
            argv = launch.command()
        self.assertEqual(argv[:2], ['/opt/frontier/candidate/opencode/opencode', 'run'])
        self.assertEqual(argv[argv.index('--format') + 1], 'json')
        self.assertEqual(argv[argv.index('--model') + 1], 'frontier/inkling-2-small')
        self.assertEqual(argv[argv.index('--dir') + 1], '/workspace')
        self.assertTrue(argv[argv.index('--title') + 1])
        self.assertIn('--auto', argv)
        self.assertNotIn(launch.PROMPT, argv)
        for flag in ('--continue', '--session', '--share', '--attach', '--variant'):
            self.assertNotIn(flag, argv)

    def test_endpoint_rejects_provider_credentials_dns_and_foreign_ports(self):
        for value in ('https://tinker.thinkingmachines.dev/v1',
                      'http://provider.example:8082/v1', 'http://127.0.0.1:8083/v1',
                      'http://secret@127.0.0.1:8082/v1', 'http://127.0.0.1:8082/v1?key=x',
                      'http://127.0.0.1:8082/openai/v1', 'http://127.0.0.1:8082/v1#x'):
            with self.subTest(value=value), patch.dict(os.environ, {'MODEL_ENDPOINT': value}):
                with self.assertRaises(launch.LaunchError):
                    launch.endpoint('MODEL_ENDPOINT', 8082, '/v1')
        for value in ('http://10.1.2.3:8082/v1', 'http://[::1]:8082/v1/'):
            with patch.dict(os.environ, {'MODEL_ENDPOINT': value}):
                self.assertEqual(launch.endpoint('MODEL_ENDPOINT', 8082, '/v1'), value.rstrip('/'))

    def test_clean_private_config_and_environment_strip_provider_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'candidate'
            workspace = Path(directory) / 'workspace'
            workspace.mkdir()
            runtime = root / 'opencode'
            cli = runtime / 'opencode'
            for path in (cli, runtime / 'bin/rg', root / 'skills/frontier-submit/SKILL.md',
                         root / 'skills/frontier-gpu/SKILL.md', root / 'bin/frontier-submit',
                         root / 'bin/frontier-gpu', root / 'bin/frontier-gpu-connect-proxy'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            env = {'PATH': '/usr/bin:/bin', 'PWD': '/elsewhere',
                   'FRONTIER_DRIVER_API_URL': 'http://10.1.2.3:8080',
                   'FRONTIER_MODEL_API_BASE_URL': 'http://10.1.2.3:8082/v1',
                   'FRONTIER_MODEL_PROXY_TOKEN': 'scoped-run-token',
                   'FRONTIER_CODEX_MODEL': 'inkling-2-small',
                   'FRONTIER_DRIVER_TOKEN_FILE': '/var/run/frontier/token',
                   'TINKER_API_KEY': 'private-provider-secret',
                   'OPENAI_API_KEY': 'private-provider-secret',
                   'OPENAI_BASE_URL': 'https://foreign-provider.example',
                   'ANTHROPIC_API_KEY': 'private-provider-secret',
                   'OPENCODE_CONFIG_CONTENT': '{"model": "foreign/model"}',
                   'OPENCODE_DISABLE_AUTOCOMPACT': '1', 'OPENCODE_PERMISSION': '{}',
                   'XDG_CONFIG_HOME': '/foreign/config',
                   'OTEL_EXPORTER_OTLP_ENDPOINT': 'https://collector.example',
                   'CRS_API_KEY': 'private-provider-secret', 'LD_LIBRARY_PATH': '/foreign'}
            original_is_file = Path.is_file
            def required_file(path):
                return str(path) == '/frontier/task/README.md' or original_is_file(path)
            with patch.multiple(launch, ROOT=root, WORKSPACE=workspace, RUNTIME=runtime, CLI=cli,
                                RUNTIME_BIN=runtime / 'bin'), \
                    patch.dict(os.environ, env, clear=True), patch.object(Path, 'is_file', required_file):
                child = launch.prepare_environment()
                home = workspace / '.frontier-opencode-home'
                self.assertEqual(Path(child['HOME']), home)
                self.assertEqual(home.stat().st_mode & 0o777, 0o700)
                self.assertEqual(Path(child['OPENCODE_CONFIG']), home / 'opencode.json')
                for name, sub in (('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                                  ('XDG_CACHE_HOME', 'cache'), ('XDG_STATE_HOME', 'state'),
                                  ('TMPDIR', 'tmp')):
                    self.assertEqual(Path(child[name]), home / sub)
                seeded = home / 'config/opencode'
                self.assertTrue((seeded / 'node_modules').is_dir())
                manifest = json.loads((seeded / 'package.json').read_text())
                lock = json.loads((seeded / 'package-lock.json').read_text())
                self.assertEqual(lock['packages']['']['dependencies'], manifest['dependencies'])
                self.assertIn('@opencode-ai/plugin', manifest['dependencies'])
                for path in (home / 'opencode.json', seeded / 'package.json'):
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                text = (home / 'opencode.json').read_text()
                self.assertNotIn('private-provider-secret', text)
                self.assertNotIn('scoped-run-token', text)
                config = json.loads(text)
                self.assertEqual(config['provider']['frontier']['options']['baseURL'],
                                 env['FRONTIER_MODEL_API_BASE_URL'])
                self.assertNotIn('private-provider-secret', json.dumps(child))
                for name in ('TINKER_API_KEY', 'OPENAI_API_KEY', 'OPENAI_BASE_URL',
                             'ANTHROPIC_API_KEY', 'CRS_API_KEY', 'OPENCODE_CONFIG_CONTENT',
                             'OPENCODE_DISABLE_AUTOCOMPACT', 'OPENCODE_PERMISSION',
                             'OTEL_EXPORTER_OTLP_ENDPOINT', 'LD_LIBRARY_PATH'):
                    self.assertNotIn(name, child)
                for name, value in launch.OPENCODE_ENV.items():
                    self.assertEqual(child[name], value)
                self.assertEqual(child['PWD'], str(workspace))
                self.assertEqual(child['FRONTIER_MODEL_PROXY_TOKEN'], 'scoped-run-token')
                self.assertEqual(child['FRONTIER_DRIVER_TOKEN_FILE'], '/var/run/frontier/token')
                self.assertTrue(child['PATH'].startswith(f"{root / 'bin'}:{runtime / 'bin'}:"))
                with self.assertRaises(FileExistsError):
                    launch.prepare_environment()

    def test_rejects_missing_alias_token_or_split_hosts(self):
        base = {'FRONTIER_DRIVER_API_URL': 'http://10.1.2.3:8080',
                'FRONTIER_MODEL_API_BASE_URL': 'http://10.1.2.3:8082/v1',
                'FRONTIER_MODEL_PROXY_TOKEN': 'scoped', 'FRONTIER_CODEX_MODEL': 'inkling-2-small',
                'FRONTIER_DRIVER_TOKEN_FILE': '/var/run/frontier/token'}
        cases = ({'FRONTIER_CODEX_MODEL': ''},
                 {'FRONTIER_CODEX_MODEL': 'thinkingmachines/Inkling-2-Small'},
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
            self.assertFalse((Path(directory) / '.frontier-opencode-home').exists())

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

    def test_final_receipt_decides_success_and_prompt_is_piped_verbatim(self):
        with patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'command', return_value=['fixture']), \
                patch.object(launch.subprocess, 'run') as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=True):
            run.return_value.returncode = 1
            self.assertEqual(launch.main(), 0)
            self.assertEqual(run.call_args.kwargs['input'], launch.PROMPT.encode())
            self.assertNotIn('stdin', run.call_args.kwargs)
        with patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'command', return_value=['fixture']), \
                patch.object(launch.subprocess, 'run') as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=False):
            run.return_value.returncode = 0
            self.assertEqual(launch.main(), 2)
            run.return_value.returncode = 7
            self.assertEqual(launch.main(), 7)

    def test_prompt_matches_pi_gemini_task_prompt(self):
        pi_spec = importlib.util.spec_from_file_location(
            'pi_launcher', Path(__file__).parent.parent / 'pi-gemini/launcher.py')
        pi = importlib.util.module_from_spec(pi_spec)
        pi_spec.loader.exec_module(pi)
        self.assertEqual(launch.PROMPT, pi.PROMPT)


if __name__ == '__main__':
    unittest.main()
