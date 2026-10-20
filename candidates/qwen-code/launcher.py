#!/usr/bin/env python3
"""Launch unmodified Qwen Code with the run-local Frontier model gateway."""
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path('/opt/frontier/candidate')
WORKSPACE = Path('/workspace')
RUNTIME = ROOT / 'qwen-code'
NODE = RUNTIME / 'node'
CLI = RUNTIME / 'app/node_modules/@qwen-code/qwen-code/cli-entry.js'
MODEL_ALIAS = re.compile(r'^[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?$')
WIRE_PARAMETERS = {
    'temperature': 1,
    'max_tokens': 131072,
    'thinking_budget': 131072,
    'preserve_thinking': True,
    'enable_thinking': True,
}
PROMPT = """Solve the task in /frontier/task/README.md. Read it completely first.
Read /opt/frontier/candidate/skills/frontier-submit/SKILL.md and follow it.
Work in /workspace. The task README defines the answer archive format.
After a representative local check, submit train promptly. Train is asynchronous:
frontier-submit train returns acceptance, and you may continue coding while
verification runs. Read feedback later with frontier-submit status.
Choose and save your current final answer with frontier-submit store-final,
updating it when useful. Storage does not judge or end your session.
For tasks supporting Agent GPU execution, read the frontier-gpu skill and use
that client within the task budget unless the README requires a task-local
legacy client. Do not mix clients. Never call the Driver API directly, read
its credential, or write a replacement submission client.
When ready, run frontier-submit final exactly once as your last platform action.
After acceptance, make no further tool call. Return a brief completion message.
Web search and URL fetching are disabled; use local task files and tools.
"""


class LaunchError(RuntimeError):
    pass


def endpoint(name, port, path):
    value = os.environ.get(name, '').strip()
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == 'http' and parsed.port == port
                 and parsed.path.rstrip('/') == path and not parsed.username
                 and not parsed.password and not parsed.query and not parsed.fragment)
        ipaddress.ip_address(parsed.hostname or '')
    except ValueError as error:
        raise LaunchError(f'{name} must be a run-local HTTP IP endpoint') from error
    if not valid:
        raise LaunchError(f'{name} must be a run-local HTTP IP endpoint')
    return value.rstrip('/')


def settings(model_url, model):
    wire = dict(WIRE_PARAMETERS)
    sampling = {'temperature': 1, 'max_tokens': 131072}
    if model == 'qwen3.8-max-vl-thinking':
        # The native CLI otherwise synthesizes an output cap. Null delegates
        # the output budget to the uncapped gateway/provider configuration.
        wire = {'temperature': 1, 'max_tokens': None,
                'reasoning_effort': 'high', 'enable_thinking': True}
        sampling = {'temperature': 1}
    return {
        '$version': 4,
        'security': {'auth': {'selectedType': 'openai'}},
        'model': {'name': model},
        'modelProviders': {'openai': [{
            'id': model, 'envKey': 'FRONTIER_MODEL_PROXY_TOKEN',
            'baseUrl': model_url, 'wireApi': 'chat-completions',
            'generationConfig': {
                'contextWindowSize': 1000000,
                'timeout': 3600000, 'streamIdleTimeoutMs': 3600000,
                'maxRetries': 16,
                # Qwen's final extra_body merge preserves the requested cap
                # when its per-send context calculation would reduce it.
                'samplingParams': sampling,
                'extra_body': wire,
            },
        }]},
        'skills': {'disabledLevels': ['project', 'extension', 'bundled'],
                   'directories': [str(ROOT / 'skills')]},
        'mcpServers': {},
        'memory': {'enableManagedAutoMemory': False,
                   'enableManagedAutoDream': False, 'enableAutoSkill': False},
        'telemetry': {'enabled': False},
        'tools': {'disabled': ['web_fetch', 'web_search'],
                  'shell': {'defaultTimeoutMs': 0}},
        'experimental': {'webSearch': {'enabled': False}},
    }


def prepare_environment():
    for path in (NODE, CLI, Path('/frontier/task/README.md'),
                 ROOT / 'skills/frontier-submit/SKILL.md',
                 ROOT / 'skills/frontier-gpu/SKILL.md',
                 ROOT / 'bin/frontier-submit', ROOT / 'bin/frontier-gpu',
                 ROOT / 'bin/frontier-gpu-connect-proxy'):
        if not path.is_file():
            raise LaunchError(f'required runtime file is missing: {path}')
    if not WORKSPACE.is_dir():
        raise LaunchError('workspace is unavailable')
    driver = endpoint('FRONTIER_DRIVER_API_URL', 8080, '')
    model_url = endpoint('FRONTIER_MODEL_API_BASE_URL', 8082, '/v1')
    if urlsplit(driver).hostname != urlsplit(model_url).hostname:
        raise LaunchError('Driver and model gateway must share the run-local IP')
    model = os.environ.get('FRONTIER_CODEX_MODEL', '')
    if not MODEL_ALIAS.fullmatch(model):
        raise LaunchError('FRONTIER_CODEX_MODEL must be an explicit model alias')
    token = os.environ.get('FRONTIER_MODEL_PROXY_TOKEN', '')
    if not token or len(token) > 16384:
        raise LaunchError('FRONTIER_MODEL_PROXY_TOKEN is missing or invalid')
    if os.environ.get('FRONTIER_DRIVER_TOKEN_FILE') != '/var/run/frontier/token':
        raise LaunchError('Driver token file must use the closed Agent mount')
    home = WORKSPACE / '.frontier-qwen-home'
    home.mkdir(mode=0o700, exist_ok=False)
    temp = home / 'tmp'
    temp.mkdir(mode=0o700)
    qwen = home / 'qwen'
    qwen.mkdir(mode=0o700)
    descriptor = os.open(qwen / 'settings.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as destination:
        json.dump(settings(model_url, model), destination, indent=2)
        destination.write('\n')
    child = os.environ.copy()
    for name in tuple(child):
        if (name.startswith(('QWEN_', 'DASHSCOPE_', 'OPENAI_', 'ANTHROPIC_',
                             'DEEPSEEK_', 'KIMI_', 'MODAL_PROXY_'))
                or name.endswith('_API_KEY')):
            child.pop(name)
    child.update({'HOME': str(home), 'QWEN_HOME': str(qwen),
                  'TMPDIR': str(temp), 'TMP': str(temp), 'TEMP': str(temp),
                  # Qwen 0.25.0 drops this knob from model-scoped settings.
                  # Use its native environment control; Driver owns the clock.
                  'QWEN_STREAM_MAX_LIFETIME_MS': '0',
                  'ENABLE_WEB_SEARCH': 'false',
                  'PATH': f"{ROOT / 'bin'}:{child.get('PATH', '/usr/bin:/bin')}"})
    child.pop('LD_LIBRARY_PATH', None)
    return child


def command():
    return [str(NODE), str(CLI), '--auth-type', 'openai',
            '--model', os.environ['FRONTIER_CODEX_MODEL'],
            '--approval-mode', 'yolo', '--output-format', 'stream-json',
            '--prompt', PROMPT]


def accepted_final_receipt():
    path = WORKSPACE / '.frontier/final-submission.json'
    try:
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077
                or not 0 < metadata.st_size <= 16384):
            return False
        receipt = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (isinstance(receipt, dict) and set(receipt) == {
        'schema_version', 'operation', 'state', 'payload_id', 'verification_id'}
        and receipt.get('schema_version') == 'frontier.candidate-submit-result/v1'
        and receipt.get('operation') == 'final' and receipt.get('state') == 'accepted'
        and isinstance(receipt.get('payload_id'), str)
        and receipt['payload_id'].startswith('payload_')
        and isinstance(receipt.get('verification_id'), str)
        and receipt['verification_id'].startswith('verification_'))


def main():
    try:
        child = prepare_environment()
        completed = subprocess.run(command(), cwd=WORKSPACE, env=child, check=False)
    except (LaunchError, OSError) as error:
        print(f'frontier-qwen-launch: {error}', file=sys.stderr)
        return 2
    if accepted_final_receipt():
        return 0
    if completed.returncode == 0:
        print('frontier-qwen-launch: no accepted final submission receipt', file=sys.stderr)
        return 2
    return completed.returncode if 0 < completed.returncode < 256 else 2


if __name__ == '__main__':
    raise SystemExit(main())
