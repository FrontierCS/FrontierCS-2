#!/usr/bin/env python3
"""Run Google's official Antigravity CLI (agy) against the run-scoped Frontier proxy."""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path('/opt/frontier/candidate')
RUNTIME = ROOT / 'agy'
AGY = RUNTIME / 'antigravity'
LOADER = RUNTIME / 'lib/ld-linux-x86-64.so.2'
WORKSPACE = Path('/workspace')
TASK = Path('/frontier/task')
README = TASK / 'README.md'
SKILLS = ROOT / 'skills'
HOME_NAME = '.frontier-agy-home'
MODEL_ALIAS = re.compile(r'^[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?$')
AGY_MODEL = re.compile(r'^gemini-[a-z0-9.-]{1,64}$')
DEFAULT_AGY_MODEL = 'gemini-4-argon-high'
EFFORTS = {'low', 'medium', 'high', 'xhigh', 'max'}

sys.path.insert(0, str(RUNTIME))
import gemini_chat_bridge  # noqa: E402

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


def settings() -> dict:
    return {
        # Native Gemini API-key provider; the "key" is the bridge's loopback nonce.
        'modelProvider': 'gemini',
        'enableTelemetry': False,
        'notifications': False,
        'toolPermission': 'always-proceed',
        'artifactReviewPolicy': 'always-proceed',
        'allowNonWorkspaceAccess': True,
        'enableTerminalSandbox': False,
    }


def private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, exist_ok=False)
    return path


def write_private(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as destination:
        destination.write(text)


def prepare() -> tuple[dict[str, str], dict[str, str], Path]:
    for path in (AGY, LOADER, RUNTIME / 'gemini_chat_bridge.py', README,
                 SKILLS / 'frontier-submit/SKILL.md', SKILLS / 'frontier-gpu/SKILL.md',
                 ROOT / 'bin/frontier-submit', ROOT / 'bin/frontier-gpu',
                 ROOT / 'bin/frontier-gpu-connect-proxy'):
        if not path.is_file():
            raise LaunchError(f'required runtime file is missing: {path}')
    if not WORKSPACE.is_dir():
        raise LaunchError('workspace is unavailable')
    driver = endpoint('FRONTIER_DRIVER_API_URL', 8080, '')
    model_url = endpoint('FRONTIER_MODEL_API_BASE_URL', 8082, '/v1')
    if urlsplit(driver).hostname != urlsplit(model_url).hostname:
        raise LaunchError('Driver and model proxy must share the run-local IP')
    alias = os.environ.get('FRONTIER_CODEX_MODEL', '')
    if not MODEL_ALIAS.fullmatch(alias):
        raise LaunchError('FRONTIER_CODEX_MODEL must be an explicit model alias')
    token = os.environ.get('FRONTIER_MODEL_PROXY_TOKEN', '')
    if not token or len(token) > 16384:
        raise LaunchError('FRONTIER_MODEL_PROXY_TOKEN is missing or invalid')
    if os.environ.get('FRONTIER_DRIVER_TOKEN_FILE') != '/var/run/frontier/token':
        raise LaunchError('Driver token file must use the closed Agent mount')
    agy_model = os.environ.get('FRONTIER_AGY_MODEL', DEFAULT_AGY_MODEL)
    if not AGY_MODEL.fullmatch(agy_model):
        raise LaunchError('FRONTIER_AGY_MODEL must name an agy Gemini model slug')
    effort = os.environ.get('FRONTIER_AGY_EFFORT', '')
    if effort and effort not in EFFORTS:
        raise LaunchError('FRONTIER_AGY_EFFORT must be one of low|medium|high|xhigh|max')

    home = private_dir(WORKSPACE / HOME_NAME)
    temp = private_dir(home / 'tmp')
    for sub in ('.gemini', '.gemini/antigravity-cli', '.config', '.cache', '.local',
                '.local/share', '.local/state'):
        private_dir(home / sub)
    write_private(home / '.gemini/antigravity-cli/settings.json',
                  json.dumps(settings(), indent=2) + '\n')

    child = os.environ.copy()
    child.pop('LD_LIBRARY_PATH', None)
    child.update({
        'HOME': str(home), 'TMPDIR': str(temp), 'TMP': str(temp), 'TEMP': str(temp),
        'XDG_CONFIG_HOME': str(home / '.config'), 'XDG_CACHE_HOME': str(home / '.cache'),
        'XDG_DATA_HOME': str(home / '.local/share'), 'XDG_STATE_HOME': str(home / '.local/state'),
        'PATH': f"{ROOT / 'bin'}:{child.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        'NO_BROWSER': '1',
    })
    plan = {'alias': alias, 'token': token, 'model_url': model_url,
            'agy_model': agy_model, 'effort': effort}
    return child, plan, home


def command(plan: dict[str, str], home: Path) -> list[str]:
    args = [str(AGY), '--print', PROMPT, '--model', plan['agy_model'],
            '--output-format', 'stream-json', '--dangerously-skip-permissions',
            '--disable-slash-commands', '--print-timeout', '6h',
            '--add-dir', str(TASK), '--add-dir', str(SKILLS),
            '--log-file', str(home / 'agy.log')]
    if plan['effort']:
        args += ['--effort', plan['effort']]
    return args


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
    bridge = None
    try:
        child, plan, home = prepare()
        nonce = secrets.token_urlsafe(32)
        bridge = gemini_chat_bridge.Bridge(plan['model_url'], plan['token'], plan['alias'],
                                           nonce, str(home / 'model-bridge.jsonl'))
        port = bridge.start()
        child['GEMINI_API_KEY'] = nonce
        child['GOOGLE_GEMINI_BASE_URL'] = f'http://127.0.0.1:{port}'
        completed = subprocess.run(command(plan, home), cwd=WORKSPACE, env=child,
                                   stdin=subprocess.DEVNULL, check=False)
    except (LaunchError, OSError, ValueError) as error:
        print(f'frontier-agy-launch: {error}', file=sys.stderr)
        return 2
    finally:
        if bridge is not None:
            bridge.stop()
    if accepted_final_receipt():
        return 0
    if completed.returncode == 0:
        print('frontier-agy-launch: no accepted final submission receipt', file=sys.stderr)
        return 2
    return completed.returncode if 0 < completed.returncode < 256 else 2


if __name__ == '__main__':
    raise SystemExit(main())
