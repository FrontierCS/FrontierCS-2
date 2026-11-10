#!/usr/bin/env python3
"""Launch unmodified OpenCode with the run-local Frontier model gateway."""
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
RUNTIME = ROOT / 'opencode'
CLI = RUNTIME / 'opencode'
RUNTIME_BIN = RUNTIME / 'bin'
VERSION = '1.19.5'
PROVIDER = 'frontier'
MODEL_ALIAS = re.compile(r'^[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?$')
CONTEXT_WINDOW = 262144
MAX_OUTPUT_TOKENS = 262144
# Tinker reports fewer prompt tokens than it counts when it renders the history's
# reasoning (214,639 reported, 261,528 counted, 11-07). Compaction keys on the
# reported count, so the input limit leaves a quarter of the window as margin.
INPUT_LIMIT = 196608
# One hour matches the other overlays' request/idle bounds for long silent
# reasoning; the Driver owns the Agent deadline, so no total request cap.
PROVIDER_TIMEOUT_MS = 3600000
SESSION_TITLE = 'Frontier task'
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
# OpenCode switches that keep a run offline and isolated (opencode 1.19.5
# source: core/src/flag/flag.ts, opencode/src/effect/runtime-flags.ts).
OPENCODE_ENV = {
    'OPENCODE_DISABLE_AUTOUPDATE': '1',        # no version check or self-upgrade
    'OPENCODE_DISABLE_MODELS_FETCH': '1',      # bundled models.dev snapshot only
    'OPENCODE_DISABLE_LSP_DOWNLOAD': '1',      # never download language servers
    'OPENCODE_DISABLE_DEFAULT_PLUGINS': '1',   # no third-party provider auth plugins
    'OPENCODE_PURE': '1',                      # no external npm plugins
    'OPENCODE_DISABLE_PROJECT_CONFIG': '1',    # ignore workspace opencode.json/.opencode/AGENTS.md
    'OPENCODE_DISABLE_CLAUDE_CODE': '1',       # no ~/.claude prompt or skills
    'OPENCODE_DISABLE_EXTERNAL_SKILLS': '1',   # no ~/.agents or ~/.claude skill scan
    'OPENCODE_DISABLE_EMBEDDED_WEB_UI': '1',
    # Default per-response cap is 32000; send the model's full output limit.
    # Tinker clamps it to the remaining window.
    'OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX': str(MAX_OUTPUT_TOKENS),
}


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


def config(model_url, model):
    return {
        'autoupdate': False, 'share': 'disabled', 'snapshot': False,
        'lsp': False, 'formatter': False, 'mcp': {}, 'plugin': [],
        'enabled_providers': [PROVIDER],
        'model': f'{PROVIDER}/{model}', 'small_model': f'{PROVIDER}/{model}',
        'provider': {PROVIDER: {
            'npm': '@ai-sdk/openai-compatible', 'name': 'Frontier model gateway',
            'options': {
                'baseURL': model_url,
                # OpenCode substitutes {env:NAME} from the process environment
                # when it loads the config; the token is never written to disk.
                'apiKey': '{env:FRONTIER_MODEL_PROXY_TOKEN}',
                'timeout': False, 'headerTimeout': PROVIDER_TIMEOUT_MS,
                'chunkTimeout': PROVIDER_TIMEOUT_MS,
            },
            'models': {model: {
                'name': model, 'attachment': False, 'reasoning': True,
                'temperature': True, 'tool_call': True,
                # input is explicit: without it OpenCode treats context minus
                # the full output limit (0 here) as usable and compacts every turn.
                'limit': {'context': CONTEXT_WINDOW, 'input': INPUT_LIMIT,
                          'output': MAX_OUTPUT_TOKENS},
                'cost': {'input': 0, 'output': 0},
            }},
        }},
        # Last matching rule wins: allow everything, then deny network tools.
        'permission': {'*': 'allow', 'external_directory': 'allow',
                       'doom_loop': 'allow', 'webfetch': 'deny', 'websearch': 'deny'},
        'skills': {'paths': [str(ROOT / 'skills')]},
        'instructions': [],
        'compaction': {'auto': True},
    }


def write_private_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as destination:
        json.dump(value, destination, indent=2)
        destination.write('\n')


def seed_global_config_dir(directory):
    """Mark OpenCode's global config dir as already holding its plugin SDK.

    At startup OpenCode runs an npm install of @opencode-ai/plugin in every
    writable config directory unless node_modules exists and the lockfile root
    already lists every declared dependency (core/src/npm.ts). Plugins are
    disabled, so the package itself is never loaded; this only stops the
    registry request.
    """
    directory.mkdir(mode=0o700)
    (directory / 'node_modules').mkdir(mode=0o700)
    dependencies = {'@opencode-ai/plugin': VERSION}
    write_private_json(directory / 'package.json', {'private': True,
                                                     'dependencies': dependencies})
    write_private_json(directory / 'package-lock.json', {
        'lockfileVersion': 3, 'requires': True,
        'packages': {'': {'dependencies': dependencies}}})
    descriptor = os.open(directory / '.gitignore', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as destination:
        destination.write('node_modules\npackage.json\npackage-lock.json\nbun.lock\n.gitignore')


def prepare_environment():
    for path in (CLI, RUNTIME_BIN / 'rg', Path('/frontier/task/README.md'),
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
    home = WORKSPACE / '.frontier-opencode-home'
    home.mkdir(mode=0o700, exist_ok=False)
    for name in ('tmp', 'data', 'cache', 'state', 'config'):
        (home / name).mkdir(mode=0o700)
    seed_global_config_dir(home / 'config/opencode')
    config_path = home / 'opencode.json'
    write_private_json(config_path, config(model_url, model))
    child = os.environ.copy()
    for name in tuple(child):
        if (name.startswith(('OPENCODE_', 'XDG_', 'GEMINI_', 'GOOGLE_', 'OPENAI_',
                             'ANTHROPIC_', 'AZURE_', 'AWS_', 'VERTEX', 'OPENROUTER_',
                             'MODAL_PROXY_', 'TINKER_', 'OTEL_'))
                or name.endswith('_API_KEY')):
            child.pop(name)
    temp = home / 'tmp'
    child.update(OPENCODE_ENV)
    child.update({'HOME': str(home), 'OPENCODE_CONFIG': str(config_path),
                  'XDG_CONFIG_HOME': str(home / 'config'), 'XDG_DATA_HOME': str(home / 'data'),
                  'XDG_CACHE_HOME': str(home / 'cache'), 'XDG_STATE_HOME': str(home / 'state'),
                  'TMPDIR': str(temp), 'TMP': str(temp), 'TEMP': str(temp),
                  'PWD': str(WORKSPACE),
                  'PATH': f"{ROOT / 'bin'}:{RUNTIME_BIN}:{child.get('PATH', '/usr/bin:/bin')}"})
    child.pop('LD_LIBRARY_PATH', None)
    return child


def command():
    return [str(CLI), 'run', '--format', 'json', '--dir', str(WORKSPACE),
            '--model', f"{PROVIDER}/{os.environ['FRONTIER_CODEX_MODEL']}",
            # A non-default title skips OpenCode's extra title-generation call.
            '--title', SESSION_TITLE,
            # Approve any remaining permission prompt instead of auto-rejecting
            # it; denied tools (webfetch, websearch, question) stay denied.
            '--auto']


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
        # `opencode run` reads the message from piped stdin verbatim; argv
        # message words would be re-quoted. The Agent has no other input.
        completed = subprocess.run(command(), cwd=WORKSPACE, env=child,
                                   input=PROMPT.encode(), check=False)
    except (LaunchError, OSError) as error:
        print(f'frontier-opencode-launch: {error}', file=sys.stderr)
        return 2
    if accepted_final_receipt():
        return 0
    if completed.returncode == 0:
        print('frontier-opencode-launch: no accepted final submission receipt', file=sys.stderr)
        return 2
    return completed.returncode if 0 < completed.returncode < 256 else 2


if __name__ == '__main__':
    raise SystemExit(main())
