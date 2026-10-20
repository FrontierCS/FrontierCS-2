import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('deepseek_launcher', Path(__file__).with_name('launcher.py'))
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_official_gateway_config_preserves_replay_and_explicit_capacities():
    rows = json.loads(launcher.render_config('http://127.0.0.1:8082/v1',
                                           'deepseek-v4-1-flash', 1048576, 1048576, 'high'))
    config = {row['id']: row for row in rows}
    route = config['llm-pi-ai']['config']['providers']['frontier']
    assert route['apiKeyEnv'] == 'FRONTIER_MODEL_PROXY_TOKEN'
    assert route['compat']['requiresReasoningContentOnAssistantMessages']
    assert route['compat']['maxTokensField'] == 'max_tokens'
    assert route['models'][0]['contextWindow'] == 1048576
    assert route['models'][0]['maxTokens'] == 1048576
    assert 'medium' not in route['models'][0]['reasoningEfforts']
    assert route['retryPolicy']['maxRetries'] == 16
    assert config['tool-web']['disabled']
    assert config['session-telemetry-otel']['disabled']


@pytest.mark.parametrize('value', ['https://example.com/v1', 'http://localhost:8082/v1',
                                 'http://127.0.0.1:8082/v1?token=x',
                                 'http://user:secret@127.0.0.1:8082/v1',
                                 'http://127.0.0.1:8080/v1'])
def test_model_endpoint_rejects_external_or_credential_bearing_urls(monkeypatch, value):
    monkeypatch.setenv('URL', value)
    with pytest.raises(launcher.LaunchError):
        launcher.endpoint('URL', 8082, '/v1')


def test_command_uses_official_headless_json_and_async_instruction():
    command = launcher.command()
    assert '@deepseek-ai/dsh/lib/bin.js' in command[1]
    assert command[2:4] == ['--profile', 'headless']
    assert '--json' in command
    assert 'Train is asynchronous' in launcher.PROMPT
    assert '--no-wait' not in launcher.PROMPT


@pytest.mark.parametrize('value', ['', '0', '1048577', 'infinite'])
def test_missing_or_invalid_token_capacity_fails(monkeypatch, value):
    monkeypatch.setenv('CAPACITY', value)
    with pytest.raises(launcher.LaunchError):
        launcher.positive_budget('CAPACITY')


def test_saved_only_receipt_is_not_accepted_final(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, 'WORKSPACE', tmp_path)
    path = tmp_path / '.frontier/final-submission.json'
    path.parent.mkdir()
    receipt = {'schema_version': 'frontier.candidate-submit-result/v1',
               'operation': 'store-final', 'state': 'stored',
               'payload_id': 'payload_one', 'verification_id': 'verification_one'}
    path.write_text(json.dumps(receipt))
    path.chmod(0o600)
    assert not launcher.accepted_final_receipt()
    receipt.update(operation='final', state='accepted')
    path.write_text(json.dumps(receipt))
    assert launcher.accepted_final_receipt()
    path.chmod(0o644)
    assert not launcher.accepted_final_receipt()


@pytest.mark.parametrize(('returncode', 'accepted', 'expected'),
                         [(0, False, 2), (0, True, 0), (137, True, 0), (137, False, 137)])
def test_process_success_requires_final_receipt(monkeypatch, returncode, accepted, expected):
    from types import SimpleNamespace
    monkeypatch.setattr(launcher, 'prepare_environment', lambda: {})
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=returncode))
    monkeypatch.setattr(launcher, 'accepted_final_receipt', lambda: accepted)
    assert launcher.main() == expected


def test_environment_scrubs_provider_keys_and_keeps_only_scoped_auth(monkeypatch, tmp_path):
    root = tmp_path / 'candidate'
    runtime = root / 'deepseek-harness'
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    readme = tmp_path / 'task/README.md'
    for path in [runtime / 'node', runtime / 'lib/ld-linux-x86-64.so.2',
                 runtime / 'app/node_modules/@deepseek-ai/dsh/lib/bin.js', readme,
                 root / 'skills/frontier-submit/SKILL.md', root / 'skills/frontier-gpu/SKILL.md',
                 root / 'bin/frontier-submit', root / 'bin/frontier-gpu',
                 root / 'bin/frontier-gpu-connect-proxy']:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    for name, value in [('ROOT', root), ('RUNTIME', runtime), ('NODE', runtime / 'node'),
                        ('LOADER', runtime / 'lib/ld-linux-x86-64.so.2'),
                        ('CLI', runtime / 'app/node_modules/@deepseek-ai/dsh/lib/bin.js'),
                        ('SKILLS', root / 'skills'), ('WORKSPACE', workspace), ('README', readme)]:
        monkeypatch.setattr(launcher, name, value)
    for name, value in {'FRONTIER_DRIVER_API_URL': 'http://10.0.0.1:8080',
                        'FRONTIER_MODEL_API_BASE_URL': 'http://10.0.0.1:8082/v1',
                        'FRONTIER_CODEX_MODEL': 'deepseek-v4-1-flash',
                        'FRONTIER_MODEL_PROXY_TOKEN': 'scoped-fixture',
                        'FRONTIER_DRIVER_TOKEN_FILE': '/var/run/frontier/token',
                        'FRONTIER_MODEL_CONTEXT_TOKENS': '1048576',
                        'FRONTIER_MODEL_OUTPUT_TOKENS': '262144',
                        'FRONTIER_MODEL_REASONING_EFFORT': 'max',
                        'OTHER_API_KEY': 'must-not-inherit', 'META_API_KEY': 'must-not-inherit',
                        'MODAL_PROXY_TOKEN_ID': 'must-not-inherit',
                        'MODAL_PROXY_TOKEN_SECRET': 'must-not-inherit',
                        'DEEPSEEK_API_KEY': 'must-not-inherit', 'DSH_HOME': '/stale-home',
                        'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'}.items():
        monkeypatch.setenv(name, value)
    child = launcher.prepare_environment()
    assert child['FRONTIER_MODEL_PROXY_TOKEN'] == 'scoped-fixture'
    for key in ['OTHER_API_KEY', 'META_API_KEY', 'MODAL_PROXY_TOKEN_ID',
                'MODAL_PROXY_TOKEN_SECRET', 'DEEPSEEK_API_KEY']:
        assert key not in child
    config = (workspace / '.frontier-deepseek-home/frontier.patch.yml').read_text()
    assert 'must-not-inherit' not in config and 'scoped-fixture' not in config
    assert child['DSH_HOME'] != '/stale-home'
    temp = workspace / '.frontier-deepseek-home/tmp'
    assert temp.is_dir() and temp.stat().st_mode & 0o777 == 0o700
    assert all(child[name] == str(temp) for name in ['TMPDIR', 'TMP', 'TEMP'])


@pytest.mark.parametrize(('tool', 'source'), [('frontier-submit', 'frontier_submit.py'),
                                           ('frontier-gpu', 'frontier_gpu.py'),
                                           ('frontier-gpu-connect-proxy', 'frontier_gpu_connect_proxy.py')])
def test_helper_restores_only_fixed_driver_path_after_harness_scrub(monkeypatch, tool, source):
    spec = importlib.util.spec_from_file_location('platform_tool', Path(__file__).with_name('platform_tool.py'))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    seen = []
    monkeypatch.delenv('FRONTIER_DRIVER_TOKEN_FILE', raising=False)
    monkeypatch.delenv('FRONTIER_MODEL_PROXY_TOKEN', raising=False)
    monkeypatch.setattr(helper.sys, 'argv', [tool, '--help'])
    monkeypatch.setattr(helper.runpy, 'run_path', lambda path, **kwargs: seen.append(path))
    helper.main()
    assert helper.os.environ['FRONTIER_DRIVER_TOKEN_FILE'] == '/var/run/frontier/token'
    assert 'FRONTIER_MODEL_PROXY_TOKEN' not in helper.os.environ
    assert seen == [f'/opt/frontier/candidate/deepseek-harness/tools/{source}']
