#!/usr/bin/env python3
"""Official Claude Code, API-key auth, bounded incremental Agent-reported trace."""
from __future__ import annotations
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

RUNTIME = Path('/opt/frontier/candidate/claude-code')
WORKSPACE = Path('/workspace')
BIN = Path('/opt/frontier/candidate/bin')
VERSION = '2.1.296'
EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
CHUNK = 64 * 1024
MAX_BYTES = 64 * 1024 * 1024
PROMPT = '''Solve the task in /frontier/task/README.md. Read the whole README and
/opt/frontier/candidate/skills/frontier-submit/SKILL.md before working in /workspace.
Use frontier-submit train for task-approved asynchronous feedback, and status to
inspect it. Use store-final to save the best usable archive as work progresses.
For Agent GPU work, read /opt/frontier/candidate/skills/frontier-gpu/SKILL.md;
use the Driver-owned helper unless the task requires its own GPU client.
Follow the task's scientific budgets. Never read credentials or call the Driver
API directly; use the installed helpers. Submit the selected final archive using
frontier-submit final once. After acceptance, return a short completion message
without further tool calls. WebSearch and WebFetch are disabled.
'''

class LaunchError(Exception): pass

class Redactor:
    """Keep enough trailing bytes to redact credentials split across reads."""
    def __init__(self, secrets):
        secrets = [s for s in secrets if s]
        self.pattern = re.compile(b'|'.join(re.escape(s) for s in secrets)) if secrets else None
        self.keep = max(map(len, secrets), default=1) - 1
        self.pending = b''
    def feed(self, data, final=False):
        self.pending += data
        end = len(self.pending) if final else max(0, len(self.pending) - self.keep)
        if self.pattern:
            for match in self.pattern.finditer(self.pending):
                if match.start() < end < match.end(): end = match.end()
            value = self.pattern.sub(b'[REDACTED]', self.pending[:end])
        else: value = self.pending[:end]
        self.pending = self.pending[end:]
        return value

class Uploader:
    def __init__(self, endpoint, bearer):
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'http' or parsed.port != 8080 or parsed.path not in ('', '/') or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise LaunchError('invalid run-local Driver endpoint')
        try: ipaddress.ip_address(parsed.hostname or '')
        except ValueError: raise LaunchError('Driver endpoint must use a frozen IP') from None
        self.host, self.port, self.bearer = parsed.hostname, parsed.port, bearer
        self.used = 0
    def request(self, method, path, data, headers=None):
        for _ in range(2):
            conn = http.client.HTTPConnection(self.host, self.port, timeout=3)
            try:
                conn.request(method, path, body=data, headers={'Authorization': f'Bearer {self.bearer}', 'Content-Length': str(len(data)), **(headers or {})})
                response = conn.getresponse()
                # Do not print server responses, request headers or credentials.
                if response.status == 204: return True
                if response.status < 500 and response.status != 429: return False
            except (OSError, http.client.HTTPException): pass
            finally: conn.close()
        return False

class Stream:
    def __init__(self, name, upload, secrets):
        self.name, self.upload = name, upload
        self.redactor = Redactor(secrets)
        self.pending = bytearray()
        self.count, self.digest, self.truncated = 0, hashlib.sha256(), False
    def feed(self, data, final=False):
        redacted = self.redactor.feed(data, final)
        if not self.truncated:
            self.pending.extend(redacted)
            while len(self.pending) >= CHUNK: self.flush(CHUNK)
        if final: self.flush()
    def flush(self, limit=CHUNK):
        if self.truncated: self.pending.clear(); return
        value = bytes(self.pending[:limit])
        if not value: return
        digest = 'sha256:' + hashlib.sha256(value).hexdigest()
        if self.upload.used + len(value) > MAX_BYTES or not self.upload.request('PUT', f'/v1/agent-traces/{self.name}/{self.count}', value, {'x-content-sha256': digest, 'Content-Type': 'application/octet-stream'}):
            self.truncated = True; self.pending.clear(); return
        del self.pending[:len(value)]
        self.upload.used += len(value); self.count += 1; self.digest.update(value)
    def complete(self, code):
        body = json.dumps({'chunks': self.count, 'sha256': 'sha256:' + self.digest.hexdigest(), 'exit_code': code, 'truncated': self.truncated, 'cli_version': VERSION}).encode()
        return self.upload.request('POST', f'/v1/agent-traces/{self.name}/complete', body, {'Content-Type':'application/json'})

def prepare_environment():
    api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    model = os.environ.get('FRONTIER_CLAUDE_MODEL', '')
    if not api_key.startswith('sk-ant-') or len(api_key) > 16384 or not re.fullmatch(r'claude-[a-z0-9._-]{1,120}', model):
        raise LaunchError('missing or invalid API key/model')
    for path in [RUNTIME/'claude', RUNTIME/'lib/ld-linux-x86-64.so.2', Path('/frontier/task/README.md'), BIN/'frontier-submit']:
        if not stat.S_ISREG(path.stat().st_mode): raise LaunchError('required runtime file unavailable')
    token_path = Path(os.environ.get('FRONTIER_DRIVER_TOKEN_FILE', '/var/run/frontier/token'))
    if not 0 < token_path.stat().st_size <= 16384: raise LaunchError('invalid Driver credential file')
    bearer = token_path.read_text().strip()
    upload = Uploader(os.environ.get('FRONTIER_DRIVER_API_URL', ''), bearer)
    child = os.environ.copy()
    for key in list(child):
        if key.startswith(('ANTHROPIC_', 'CLAUDE_', 'CODEX_')) or key.endswith('_API_KEY') or key.lower() in ('http_proxy','https_proxy','all_proxy') or key in ('FRONTIER_MODEL_API_BASE_URL', 'LD_PRELOAD', 'NODE_OPTIONS'):
            child.pop(key, None)
    home = Path(tempfile.mkdtemp(prefix='.frontier-claude-', dir=WORKSPACE))
    config = home/'config'; config.mkdir(mode=0o700)
    temp = home/'tmp'; temp.mkdir(mode=0o700)
    child.update({'ANTHROPIC_API_KEY': api_key, 'CLAUDE_CONFIG_DIR':str(config),
                  'HOME': str(home), 'TMPDIR':str(temp), 'PATH':f'{BIN}:{child.get("PATH", "/usr/bin:/bin")}',
                  'DISABLE_AUTOUPDATER':'1', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1',
                  'DISABLE_TELEMETRY':'1', 'DISABLE_ERROR_REPORTING':'1'})
    effort = os.environ.get('FRONTIER_REASONING_EFFORT', '')
    if effort and effort not in EFFORTS: raise LaunchError('invalid effort level')
    return child, upload, [api_key.encode(), bearer.encode()], model, effort

def command(model, effort=''):
    return [str(RUNTIME/'lib/ld-linux-x86-64.so.2'), '--library-path', str(RUNTIME/'lib'), str(RUNTIME/'claude'), '-p', PROMPT, '--model', model,
            *(['--effort', effort] if effort else []),
            '--output-format', 'stream-json', '--verbose', '--include-partial-messages',
            '--dangerously-skip-permissions', '--disallowedTools', 'WebSearch,WebFetch',
            '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}', '--setting-sources', '']

def capture(process, upload, secrets):
    streams = [Stream('events', upload, secrets), Stream('stderr', upload, secrets)]
    selector = selectors.DefaultSelector()
    for pipe, stream in zip([process.stdout, process.stderr], streams):
        os.set_blocking(pipe.fileno(), False); selector.register(pipe, selectors.EVENT_READ, stream)
    last_flush = time.monotonic()
    while selector.get_map():
        for key, _ in selector.select(timeout=1):
            data = os.read(key.fileobj.fileno(), 16384)
            key.data.feed(data, final=not data)
            if not data: selector.unregister(key.fileobj); key.fileobj.close()
        if time.monotonic() - last_flush >= 2:
            for stream in streams: stream.flush()
            last_flush = time.monotonic()
    selector.close()
    code = process.wait()
    for stream in streams:
        if not stream.complete(code): print(f'Claude {stream.name} trace incomplete', file=sys.stderr)
    return code

def accepted_final():
    path = WORKSPACE/'.frontier/final-submission.json'
    try:
        meta = path.lstat()
        if not stat.S_ISREG(meta.st_mode) or meta.st_mode & 0o077 or not 0 < meta.st_size <= 16384: return False
        value = json.loads(path.read_bytes())
        return value.get('schema_version') == 'frontier.candidate-submit-result/v1' and value.get('operation') == 'final' and value.get('state') == 'accepted' and value.get('payload_id','').startswith('payload_') and value.get('verification_id','').startswith('verification_')
    except (OSError, ValueError, AttributeError): return False

def main():
    try:
        child, upload, secrets, model, effort = prepare_environment()
        process = subprocess.Popen(command(model, effort), cwd=WORKSPACE, env=child, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        def terminate(_signum, _frame):
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
        code = capture(process, upload, secrets)
        if accepted_final(): return 0
        if code == 0: print('Claude exited without an accepted final receipt', file=sys.stderr)
        return code if 0 < code < 256 else 2
    except (LaunchError, OSError, ValueError):
        print('Claude launcher could not initialize; check runtime and credential configuration', file=sys.stderr)
        return 2

if __name__ == '__main__': raise SystemExit(main())
