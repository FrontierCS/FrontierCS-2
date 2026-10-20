#!/usr/bin/env python3
"""Launch unmodified pi with the run-local Frontier model gateway."""
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
RUNTIME = ROOT / 'pi'
NODE = RUNTIME / 'node'
CLI = RUNTIME / 'app/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js'
PROVIDER = 'frontier'
MODEL_ALIAS = re.compile(r'^[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?$')
# Gemini thinking levels as forwarded by LiteLLM's Chat Completions
# reasoning_effort mapping. "high" is the highest level Google documents for
# Gemini 3-family thinking_level; Pi levels above it are disabled so a
# request can never silently fall back to an unsupported label.
THINKING_LEVEL = 'high'
THINKING_LEVEL_MAP = {'off': None, 'minimal': None, 'low': 'low',
                      'medium': 'medium', 'high': 'high', 'xhigh': None, 'max': None}
CONTEXT_WINDOW = 1048576
MAX_OUTPUT_TOKENS = 1048576
# Pi's default built-ins. grep/find need fd and rg, which Pi would download;
# the agent uses shell search instead.
TOOLS = 'read,bash,edit,write'
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


def models(model_url, model):
    return {'providers': {PROVIDER: {
        'baseUrl': model_url,
        'api': 'openai-completions',
        # Pi resolves $NAME from the process environment at request time;
        # the value itself is never written to disk.
        'apiKey': '$FRONTIER_MODEL_PROXY_TOKEN',
        'models': [{
            'id': model, 'name': model, 'reasoning': True,
            'thinkingLevelMap': THINKING_LEVEL_MAP,
            'input': ['text', 'image'],
            'contextWindow': CONTEXT_WINDOW, 'maxTokens': MAX_OUTPUT_TOKENS,
            'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
            'compat': {
                'supportsStore': False, 'supportsDeveloperRole': False,
                'supportsReasoningEffort': True, 'supportsUsageInStreaming': True,
                'maxTokensField': 'max_completion_tokens', 'thinkingFormat': 'openai',
                'sendSessionAffinityHeaders': False, 'supportsLongCacheRetention': False,
            },
        }],
    }}}


def settings(model):
    return {
        'defaultProvider': PROVIDER, 'defaultModel': model,
        'defaultThinkingLevel': THINKING_LEVEL,
        'defaultProjectTrust': 'never',
        'defaultTools': TOOLS.split(','),
        'extensions': ['-builtin:mcp', '-builtin:llama.cpp', '-builtin:codemode',
                       '-builtin:tool-search'],
        'packages': [], 'skills': [], 'prompts': [], 'themes': [],
        'enableInstallTelemetry': False, 'enableAnalytics': False,
        'cacheWarming': 'off', 'quietStartup': True,
        # Driver owns the Agent deadline. One hour matches the other
        # overlays' request/idle bounds for long silent reasoning.
        'httpIdleTimeoutMs': 3600000,
        'retry': {'enabled': True, 'maxRetries': 16, 'baseDelayMs': 2000,
                  'maxAgentDelayMs': 60000,
                  'provider': {'timeoutMs': 3600000, 'maxRetries': 0}},
        'compaction': {'enabled': True},
    }


def write_private_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as destination:
        json.dump(value, destination, indent=2)
        destination.write('\n')


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
    home = WORKSPACE / '.frontier-pi-home'
    home.mkdir(mode=0o700, exist_ok=False)
    temp = home / 'tmp'
    temp.mkdir(mode=0o700)
    agent = home / 'agent'
    agent.mkdir(mode=0o700)
    (agent / 'sessions').mkdir(mode=0o700)
    write_private_json(agent / 'models.json', models(model_url, model))
    write_private_json(agent / 'settings.json', settings(model))
    child = os.environ.copy()
    for name in tuple(child):
        if (name.startswith(('PI_', 'GEMINI_', 'GOOGLE_', 'OPENAI_', 'ANTHROPIC_',
                             'AZURE_', 'AWS_', 'VERTEX', 'OPENROUTER_',
                             'MODAL_PROXY_'))
                or name.endswith('_API_KEY')):
            child.pop(name)
    child.update({'HOME': str(home), 'PI_CODING_AGENT_DIR': str(agent),
                  'PI_CODING_AGENT_SESSION_DIR': str(agent / 'sessions'),
                  'PI_OFFLINE': '1', 'PI_SKIP_VERSION_CHECK': '1', 'PI_TELEMETRY': '0',
                  'TMPDIR': str(temp), 'TMP': str(temp), 'TEMP': str(temp),
                  'PATH': f"{ROOT / 'bin'}:{child.get('PATH', '/usr/bin:/bin')}"})
    child.pop('LD_LIBRARY_PATH', None)
    return child


def command():
    return [str(NODE), str(CLI), '--mode', 'json', '--offline', '--no-approve',
            '--provider', PROVIDER, '--model', os.environ['FRONTIER_CODEX_MODEL'],
            '--thinking', THINKING_LEVEL, '--tools', TOOLS,
            '--no-extensions', '--no-mcp', '--no-prompt-templates', '--no-themes',
            '--no-skills', '--skill', str(ROOT / 'skills/frontier-submit'),
            '--skill', str(ROOT / 'skills/frontier-gpu'),
            '--', PROMPT]


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
        # Pi prepends piped stdin to the prompt; the Agent has no input stream.
        completed = subprocess.run(command(), cwd=WORKSPACE, env=child,
                                   stdin=subprocess.DEVNULL, check=False)
    except (LaunchError, OSError) as error:
        print(f'frontier-pi-launch: {error}', file=sys.stderr)
        return 2
    if accepted_final_receipt():
        return 0
    if completed.returncode == 0:
        print('frontier-pi-launch: no accepted final submission receipt', file=sys.stderr)
        return 2
    return completed.returncode if 0 < completed.returncode < 256 else 2


if __name__ == '__main__':
    raise SystemExit(main())
