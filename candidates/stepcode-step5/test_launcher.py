import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('stepcode_launcher', Path(__file__).with_name('launcher.py'))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
ROUTES = json.loads(Path(__file__).with_name('gateway-models.json').read_text(encoding='utf-8'))


class LauncherTests(unittest.TestCase):
    def test_model_uses_gateway_chat_completions_and_openrouter_high_reasoning(self):
        config = launch.models('http://10.1.2.3:8082/v1', 'step-5-preview')
        provider = config['providers']['frontier']
        self.assertEqual(provider['baseUrl'], 'http://10.1.2.3:8082/v1')
        self.assertEqual(provider['api'], 'openai-completions')
        self.assertEqual(provider['apiKey'], '$FRONTIER_MODEL_PROXY_TOKEN')
        model = provider['models'][0]
        self.assertEqual(model['id'], 'step-5-preview')
        self.assertTrue(model['reasoning'])
        self.assertEqual(model['thinkingLevelMap']['high'], 'high')
        self.assertEqual([level for level, value in model['thinkingLevelMap'].items() if value],
                         ['low', 'medium', 'high'])
        self.assertEqual(model['contextWindow'], 1000000)
        self.assertEqual(model['maxTokens'], 64000)
        compat = model['compat']
        self.assertEqual(compat['thinkingFormat'], 'openrouter')
        self.assertFalse(compat['supportsStore'])
        self.assertFalse(compat['supportsDeveloperRole'])
        self.assertFalse(compat['sendSessionAffinityHeaders'])
        self.assertEqual(launch.settings('step-5-preview')['defaultThinkingLevel'], 'high')

    def test_settings_grant_unattended_full_access_and_disable_discovery(self):
        config = launch.settings('step-5-preview')
        self.assertEqual(config['permissionPreset'], 'full-access')
        self.assertIs(config['fullAccessAcknowledged'], True)
        self.assertEqual(config['nonInteractiveApproval'], 'allow')
        self.assertEqual(config['defaultProjectTrust'], 'never')
        self.assertEqual(config['defaultProvider'], 'frontier')
        self.assertEqual(config['defaultModel'], 'step-5-preview')
        for name in ('packages', 'extensions', 'skills', 'prompts', 'themes'):
            self.assertEqual(config[name], [])
        self.assertFalse(config['enableAnalytics'])
        self.assertEqual(config['httpIdleTimeoutMs'], 3600000)
        self.assertEqual(config['retry']['provider'], {'timeoutMs': 3600000, 'maxRetries': 0})
        self.assertEqual(config['retry']['maxRetries'], 8)
        self.assertTrue(config['compaction']['enabled'])

    def test_config_toml_round_trips_through_a_toml_parser(self):
        text = '\n'.join(launch.toml_document(launch.settings('step-5-preview'))) + '\n'
        self.assertEqual(tomllib.loads(text), launch.settings('step-5-preview'))
        self.assertIn('\n[retry.provider]\n', text)
        with self.assertRaises(TypeError):
            launch.toml_value(1.5)

    def test_command_is_headless_print_json_without_web_search_or_discovery(self):
        with patch.dict(os.environ, {'FRONTIER_CODEX_MODEL': 'step-5-preview'}):
            argv = launch.command()
        self.assertEqual(argv[:3], [str(launch.NODE), str(launch.CLI), '-p'])
        for flag in ('--no-approve', '--no-update-check', '--no-extensions',
                     '--no-skills', '--no-prompt-templates', '--no-themes'):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index('--mode') + 1], 'json')
        self.assertEqual(argv[argv.index('--model') + 1], 'step-5-preview')
        self.assertEqual(argv[argv.index('--provider') + 1], 'frontier')
        self.assertEqual(argv[argv.index('--thinking') + 1], 'high')
        self.assertEqual(argv[argv.index('--exclude-tools') + 1], 'search_web')
        self.assertNotIn('--tools', argv)
        self.assertNotIn('--api-key', argv)
        self.assertEqual([argv[index + 1] for index, value in enumerate(argv) if value == '--skill'],
                         [str(launch.ROOT / 'skills/frontier-submit'),
                          str(launch.ROOT / 'skills/frontier-gpu')])
        self.assertEqual(argv[-2:], ['--', launch.PROMPT])

    def test_endpoint_rejects_provider_credentials_dns_and_foreign_ports(self):
        for value in ('https://openrouter.ai/api/v1',
                      'http://provider.example:8082/v1', 'http://127.0.0.1:8083/v1',
                      'http://secret@127.0.0.1:8082/v1', 'http://127.0.0.1:8082/v1?key=x',
                      'http://127.0.0.1:8082/api/v1', 'http://127.0.0.1:8082/v1#x'):
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
            runtime = root / 'stepcode'
            cli = runtime / 'app/apps/cli/dist/main.js'
            node = runtime / 'node'
            for path in (cli, node, root / 'skills/frontier-submit/SKILL.md',
                         root / 'skills/frontier-gpu/SKILL.md', root / 'bin/frontier-submit',
                         root / 'bin/frontier-gpu', root / 'bin/frontier-gpu-connect-proxy'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            env = {'PATH': '/usr/bin:/bin', 'FRONTIER_DRIVER_API_URL': 'http://10.1.2.3:8080',
                   'FRONTIER_MODEL_API_BASE_URL': 'http://10.1.2.3:8082/v1',
                   'FRONTIER_MODEL_PROXY_TOKEN': 'scoped-run-token',
                   'FRONTIER_CODEX_MODEL': 'step-5-preview',
                   'FRONTIER_DRIVER_TOKEN_FILE': '/var/run/frontier/token',
                   'OPENROUTER_STEPFUN_API_KEY': 'private-provider-secret',
                   'OPENROUTER_API_KEY': 'private-provider-secret',
                   'STEP_API_KEY': 'private-provider-secret',
                   'STEPFUN_API_KEY': 'private-provider-secret',
                   'STEPCODE_SEARCH_API_KEY': 'private-provider-secret',
                   'STEP_BASE_URL': 'https://foreign-provider.example',
                   'STEP_PROVIDER': 'step', 'STEP_MODEL': 'step-3.7-flash',
                   'STEP_PERMISSION_PRESET': 'ask', 'STEP_AUTO_RESUME': '0',
                   'STEPCODE_CONFIG_PATH': '/foreign/stepcode.json',
                   'STEP_CODING_AGENT_DIR': '/foreign/agent', 'AI_AGENT': 'foreign',
                   'OPENAI_API_KEY': 'private-provider-secret',
                   'OTHER_API_KEY': 'private-provider-secret', 'LD_LIBRARY_PATH': '/foreign'}
            original_is_file = Path.is_file
            def required_file(path):
                return str(path) == '/frontier/task/README.md' or original_is_file(path)
            with patch.multiple(launch, ROOT=root, WORKSPACE=workspace, NODE=node, CLI=cli), \
                    patch.dict(os.environ, env, clear=True), patch.object(Path, 'is_file', required_file):
                child = launch.prepare_environment()
                home = workspace / '.frontier-stepcode-home'
                storage = home / '.stepcode'
                agent = Path(child['STEP_CODING_AGENT_DIR'])
                self.assertEqual(agent, storage / 'agent')
                self.assertEqual(child['HOME'], str(home))
                self.assertEqual(child['STEP_CODING_AGENT_SESSION_DIR'], str(agent / 'sessions'))
                self.assertEqual(child['STEP_CODING_AGENT_PLAN_DIR'], str(home / 'plans'))
                for name in ('models.json', 'config.toml'):
                    path = storage / name
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                    text = path.read_text()
                    self.assertNotIn('private-provider-secret', text)
                    self.assertNotIn('scoped-run-token', text)
                self.assertFalse((storage / 'auth.json').exists())
                for path in (home, storage, agent, home / 'tmp', home / 'plans'):
                    self.assertEqual(path.stat().st_mode & 0o777, 0o700)
                self.assertNotIn('private-provider-secret', json.dumps(child))
                self.assertNotIn('foreign', json.dumps(child))
                for name in ('OPENROUTER_STEPFUN_API_KEY', 'STEP_API_KEY', 'STEPFUN_API_KEY',
                             'STEP_BASE_URL', 'STEP_PROVIDER', 'STEP_MODEL', 'STEP_PERMISSION_PRESET',
                             'STEP_AUTO_RESUME', 'STEPCODE_CONFIG_PATH', 'AI_AGENT', 'LD_LIBRARY_PATH'):
                    self.assertNotIn(name, child)
                self.assertEqual(child['STEP_NON_INTERACTIVE_DENIAL'], 'continue')
                self.assertEqual(child['STEPCODE_DISABLE_PI_SERVICES'], '1')
                self.assertEqual(child['TMPDIR'], str(home / 'tmp'))
                self.assertEqual(child['FRONTIER_MODEL_PROXY_TOKEN'], 'scoped-run-token')
                self.assertEqual(child['FRONTIER_DRIVER_TOKEN_FILE'], '/var/run/frontier/token')
                self.assertTrue(child['PATH'].startswith(str(root / 'bin') + ':'))
                models = json.loads((storage / 'models.json').read_text())
                self.assertEqual(models['providers']['frontier']['baseUrl'],
                                 env['FRONTIER_MODEL_API_BASE_URL'])
                config = tomllib.loads((storage / 'config.toml').read_text())
                self.assertEqual(config['permissionPreset'], 'full-access')
                self.assertEqual(config['defaultModel'], 'step-5-preview')
                with self.assertRaises(FileExistsError):
                    launch.prepare_environment()

    def test_rejects_missing_alias_token_or_split_hosts(self):
        base = {'FRONTIER_DRIVER_API_URL': 'http://10.1.2.3:8080',
                'FRONTIER_MODEL_API_BASE_URL': 'http://10.1.2.3:8082/v1',
                'FRONTIER_MODEL_PROXY_TOKEN': 'scoped', 'FRONTIER_CODEX_MODEL': 'step-5-preview',
                'FRONTIER_DRIVER_TOKEN_FILE': '/var/run/frontier/token'}
        cases = ({'FRONTIER_CODEX_MODEL': ''}, {'FRONTIER_CODEX_MODEL': 'stepfun/step-5-preview'},
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
            self.assertFalse((Path(directory) / '.frontier-stepcode-home').exists())

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
        # Step ends a print run with exit 0 after an empty, reasoning-only or
        # length-truncated reply; without a receipt that is not success.
        answered = {'role': 'assistant', 'stopReason': 'toolUse',
                    'content': [{'type': 'toolCall'}]}
        with patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'command', return_value=['fixture']), \
                patch.object(launch, 'run_agent', return_value=(1, None)) as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=True):
            self.assertEqual(launch.main(), 0)
            self.assertEqual(run.call_count, 1)
        with patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'command', return_value=['fixture']), \
                patch.object(launch, 'run_agent', return_value=(0, answered)) as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=False):
            self.assertEqual(launch.main(), 2)
            self.assertEqual(run.call_count, 1)
            run.return_value = (7, None)
            self.assertEqual(launch.main(), 7)

    def test_run_agent_passes_events_through_and_closes_stdin(self):
        events = [b'{"type":"agent_start"}\n',
                  b'{"type":"message_end","message":{"role":"assistant","stopReason":"toolUse"}}\n',
                  b'{"type":"message_end","message":{"role":"toolResult"}}\n',
                  b'{"type":"message_end","message":{"role":"assistant","stopReason":"stop"}}\n']
        with patch.object(launch.subprocess, 'Popen') as popen, \
                patch.object(launch.sys, 'stdout') as stdout:
            process = popen.return_value.__enter__.return_value
            process.stdout = iter(events)
            process.returncode = 0
            self.assertEqual(launch.run_agent(['fixture'], {}),
                             (0, {'role': 'assistant', 'stopReason': 'stop'}))
        self.assertIs(popen.call_args.kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual([call.args[0] for call in stdout.buffer.write.call_args_list], events)

    def test_prompts_warn_about_control_markers_without_writing_them(self):
        for prompt in (launch.PROMPT, launch.RESUME_PROMPT):
            self.assertIn(launch.SPECIAL_TOKEN_NOTE, prompt)
            self.assertNotRegex(prompt, r'<\||\|>|</?(?:tool_call|function|parameter)\b')
        self.assertTrue(launch.PROMPT.startswith('Solve the task in /frontier/task/README.md.'))
        self.assertTrue(launch.RESUME_PROMPT.startswith('Your previous reply was cut off'))

    def test_truncated_reply_is_a_stop_or_length_without_tool_call(self):
        thinking = {'type': 'thinking', 'thinking': 'The issue in my example:'}
        cut_text = {'type': 'text', 'text': 'So turn-1 '}
        for reason in ('stop', 'length'):
            for content in ([thinking], [{'type': 'text', 'text': ' \n'}], [thinking, cut_text], []):
                self.assertTrue(launch.truncated_reply(
                    {'role': 'assistant', 'stopReason': reason, 'content': content}))
        for message in (None, {'stopReason': 'error', 'content': [thinking]},
                        {'stopReason': 'aborted', 'content': [thinking]},
                        {'stopReason': 'toolUse', 'content': [thinking]},
                        {'stopReason': 'stop', 'content': [thinking, {'type': 'toolCall'}]},
                        {'stopReason': 'length', 'content': [{'type': 'toolCall'}]},
                        {'stopReason': 'stop', 'content': ['text']},
                        {'stopReason': 'stop'}):
            self.assertFalse(launch.truncated_reply(message))

    def test_truncated_reply_resumes_the_session_a_bounded_number_of_times(self):
        cut = {'role': 'assistant', 'stopReason': 'stop',
               'content': [{'type': 'thinking', 'thinking': 'Hmm, wait'}]}
        with patch.dict(os.environ, {'FRONTIER_CODEX_MODEL': 'step-5-preview'}), \
                patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'run_agent', return_value=(0, cut)) as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=False):
            self.assertEqual(launch.main(), 2)
        self.assertEqual(run.call_count, 1 + launch.MAX_TRUNCATED_RESUMES)
        first, *resumes = [call.args[0] for call in run.call_args_list]
        self.assertNotIn('--continue', first)
        self.assertEqual(first[-1], launch.PROMPT)
        for argv in resumes:
            self.assertEqual(argv[-3:], ['--continue', '--', launch.RESUME_PROMPT])
        receipts = iter([False, True, True])
        with patch.dict(os.environ, {'FRONTIER_CODEX_MODEL': 'step-5-preview'}), \
                patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'run_agent', return_value=(0, cut)) as run, \
                patch.object(launch, 'accepted_final_receipt', side_effect=lambda: next(receipts)):
            self.assertEqual(launch.main(), 0)
        self.assertEqual(run.call_count, 2)
        with patch.dict(os.environ, {'FRONTIER_CODEX_MODEL': 'step-5-preview'}), \
                patch.object(launch, 'prepare_environment', return_value={}), \
                patch.object(launch, 'run_agent', return_value=(9, cut)) as run, \
                patch.object(launch, 'accepted_final_receipt', return_value=False):
            self.assertEqual(launch.main(), 9)
        self.assertEqual(run.call_count, 1)


class GatewayRouteTests(unittest.TestCase):
    def test_route_is_secret_free_openrouter_high_reasoning_without_output_cap(self):
        self.assertEqual([row['model_name'] for row in ROUTES['model_list']], ['step-5-preview'])
        row = ROUTES['model_list'][0]
        self.assertRegex(row['model_name'], launch.MODEL_ALIAS)
        params, info = row['litellm_params'], row['model_info']
        self.assertEqual(params['model'], 'openrouter/stepfun/step-5-preview')
        self.assertEqual(params['api_base'], 'https://openrouter.ai/api/v1')
        self.assertEqual(params['api_key'], 'os.environ/OPENROUTER_STEPFUN_API_KEY')
        self.assertNotRegex(json.dumps(ROUTES), re.compile(r'bearer |sk-or-|sk-', re.IGNORECASE))
        self.assertIs(params['drop_params'], True)
        self.assertEqual(set(params['additional_drop_params']),
                         {'max_tokens', 'max_completion_tokens', 'reasoning_effort'})
        self.assertEqual(params['extra_body'], {
            'reasoning': {'effort': launch.THINKING_LEVEL},
            # Prefer StepFun's first-party endpoint (reports reasoning tokens);
            # allow OpenRouter fallback so one upstream outage does not stall runs.
            'provider': {'order': ['StepFun'], 'allow_fallbacks': True},
        })
        self.assertEqual(info['mode'], 'chat')
        self.assertIs(info['supports_function_calling'], True)
        self.assertIs(info['supports_reasoning'], True)


if __name__ == '__main__':
    unittest.main()
