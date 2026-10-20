#!/usr/bin/env python3
"""Run the official DeepSeek Harness against the run-scoped Frontier proxy."""

from __future__ import annotations

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
RUNTIME = ROOT / 'deepseek-harness'
WORKSPACE = Path('/workspace')
README = Path('/frontier/task/README.md')
SKILLS = ROOT / 'skills'
NODE = RUNTIME / 'node'
LIB = RUNTIME / 'lib'
LOADER = LIB / 'ld-linux-x86-64.so.2'
CLI = RUNTIME / 'app/node_modules/@deepseek-ai/dsh/lib/bin.js'
MODEL_ALIAS = re.compile(r'^[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?$')

PROMPT = """Solve the task in /frontier/task/README.md. Read it completely first.
Read /opt/frontier/candidate/skills/frontier-submit/SKILL.md and follow it.
Work in /workspace. The task README defines the answer archive format.
After a representative local check, submit train promptly. Train is asynchronous:
frontier-submit train returns acceptance, and you may continue coding while
verification runs. Read feedback later with frontier-submit status.
Choose and save your current final answer with frontier-submit store-final,
updating it when useful. Storage does not judge or end your session.
For tasks explicitly supporting Agent GPU execution, read the frontier-gpu skill
and use that client within the task budget, unless the README explicitly requires
a task-local legacy client. Do not mix clients. Never call the Driver API directly,
read its credential, or write a replacement submission client.
When ready, run frontier-submit final exactly once as your last platform action.
After acceptance, make no further tool call. Return a brief completion message.
Web search and URL fetching are disabled; use local task files and tools.
"""


class LaunchError(RuntimeError):
    pass


def endpoint(name: str, port: int, path: str) -> str:
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


def render_config(model_url: str, model: str, context: int, output: int,
                  effort: str) -> str:
    # JSON is valid YAML; env references keep the scoped credential out of config.
    rows = [
        {'id': 'agent-default-model', 'config': {
            'provider': 'frontier', 'model': model, 'reasoningEffort': effort}},
        {'id': 'llm-pi-ai', 'config': {'providers': {'frontier': {
            'apiKeyEnv': 'FRONTIER_MODEL_PROXY_TOKEN', 'api': 'openai-completions',
            'baseURL': model_url, 'reasoning': effort,
            'timeoutMs': 3600000, 'streamIdleTimeoutMs': 3600000,
            'retryPolicy': {'mode': 'normal', 'maxRetries': 16},
            'compat': {'supportsDeveloperRole': False, 'supportsStore': False,
                       'supportsReasoningEffort': True, 'supportsUsageInStreaming': True,
                       'maxTokensField': 'max_tokens', 'thinkingFormat': 'openai',
                       'requiresReasoningContentOnAssistantMessages': True},
            'models': [{'id': model, 'contextWindow': context, 'maxTokens': output,
                        'input': ['text'], 'reasoningEfforts': {
                            'off': 'none', 'low': 'low', 'high': 'high',
                            'xhigh': 'xhigh', 'max': 'max'}}]}}}},
        {'id': 'llm-deepseek', 'disabled': True},
        {'id': 'llm-deepseek-account', 'disabled': True},
        {'id': 'session-title-llm', 'disabled': True},
        {'id': 'session-telemetry-otel', 'disabled': True},
        {'id': 'tool-web', 'disabled': True},
        {'id': 'web-search-deepseek', 'disabled': True},
        {'id': 'skill-filesystem', 'config': {
            'includeDefaultRoots': False, 'customSkillDirs': [str(SKILLS)], 'watch': False}},
        {'id': 'bash-sandbox', 'config': {
            'timeoutMs': 3600000, 'maxTimeoutMs': 3600000,
            'maxOutputBytes': 8388608, 'maxSpillBytes': 1073741824}},
    ]
    return json.dumps(rows, ensure_ascii=True, indent=2) + '\n'


def positive_budget(name: str) -> int:
    value = os.environ.get(name, '')
    if not value.isdecimal() or not 1 <= int(value) <= 1048576:
        raise LaunchError(f'{name} must be an explicit token capacity between 1 and 1048576')
    return int(value)


def prepare_environment() -> dict[str, str]:
    for path in (NODE, LOADER, CLI, README, SKILLS / 'frontier-submit/SKILL.md',
                 SKILLS / 'frontier-gpu/SKILL.md', ROOT / 'bin/frontier-submit',
                 ROOT / 'bin/frontier-gpu', ROOT / 'bin/frontier-gpu-connect-proxy'):
        if not path.is_file():
            raise LaunchError(f'required runtime file is missing: {path}')
    if not WORKSPACE.is_dir():
        raise LaunchError('workspace is unavailable')
    driver = endpoint('FRONTIER_DRIVER_API_URL', 8080, '')
    model_url = endpoint('FRONTIER_MODEL_API_BASE_URL', 8082, '/v1')
    if urlsplit(driver).hostname != urlsplit(model_url).hostname:
        raise LaunchError('Driver and model proxy must share the run-local IP')
    model = os.environ.get('FRONTIER_CODEX_MODEL', '')
    if not MODEL_ALIAS.fullmatch(model):
        raise LaunchError('FRONTIER_CODEX_MODEL must be an explicit model alias')
    token = os.environ.get('FRONTIER_MODEL_PROXY_TOKEN', '')
    if not token or len(token) > 16384:
        raise LaunchError('FRONTIER_MODEL_PROXY_TOKEN is missing or invalid')
    if os.environ.get('FRONTIER_DRIVER_TOKEN_FILE') != '/var/run/frontier/token':
        raise LaunchError('Driver token file must use the closed Agent mount')
    context = positive_budget('FRONTIER_MODEL_CONTEXT_TOKENS')
    output = positive_budget('FRONTIER_MODEL_OUTPUT_TOKENS')
    if output > context:
        raise LaunchError('output token capacity exceeds the context window')
    effort = os.environ.get('FRONTIER_MODEL_REASONING_EFFORT', '')
    if effort not in {'low', 'high', 'xhigh', 'max'}:
        raise LaunchError('FRONTIER_MODEL_REASONING_EFFORT must select a supported thinking effort')
    child = os.environ.copy()
    for name in tuple(child):
        if (name.startswith(('DSH_', 'DEEPSEEK_', 'OPENAI_', 'ANTHROPIC_', 'KIMI_',
                             'MODAL_PROXY_')) or name.endswith('_API_KEY')):
            child.pop(name)
    home = WORKSPACE / '.frontier-deepseek-home'
    home.mkdir(mode=0o700, exist_ok=False)
    temp = home / 'tmp'
    temp.mkdir(mode=0o700)
    patch = home / 'frontier.patch.yml'
    descriptor = os.open(patch, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as destination:
        destination.write(render_config(model_url, model, context, output, effort))
    child.update({'DSH_HOME': str(home), 'DSH_PERMISSION_MODE': 'danger-full-access',
                  'HOME': str(home), 'TMPDIR': str(temp), 'TMP': str(temp),
                  'TEMP': str(temp),
                  'PATH': f"{ROOT / 'bin'}:{child.get('PATH', '/usr/bin:/bin')}"})
    child.pop('LD_LIBRARY_PATH', None)
    return child


def command() -> list[str]:
    # A private ELF interpreter keeps execPath pointing at Node for native reexec.
    return [str(NODE), str(CLI),
            '--profile', 'headless', '--patch',
            str(WORKSPACE / '.frontier-deepseek-home/frontier.patch.yml'),
            '--json', PROMPT]


def accepted_final_receipt() -> bool:
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


def main() -> int:
    try:
        child = prepare_environment()
        completed = subprocess.run(command(), cwd=WORKSPACE, env=child, check=False)
    except (LaunchError, OSError) as error:
        print(f'frontier-deepseek-launch: {error}', file=sys.stderr)
        return 2
    if accepted_final_receipt():
        return 0
    if completed.returncode == 0:
        print('frontier-deepseek-launch: no accepted final submission receipt', file=sys.stderr)
        return 2
    return completed.returncode if 0 < completed.returncode < 256 else 2


if __name__ == '__main__':
    raise SystemExit(main())
