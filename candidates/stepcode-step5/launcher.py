#!/usr/bin/env python3
"""Launch unmodified Step Code with the run-local Frontier model gateway."""
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
RUNTIME = ROOT / 'stepcode'
NODE = RUNTIME / 'node'
CLI = RUNTIME / 'app/apps/cli/dist/main.js'
PROVIDER = 'frontier'
MODEL_ALIAS = re.compile(r'^[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?$')
# OpenRouter reasoning effort, sent natively by Step's `openrouter` thinking
# format as `reasoning: {effort}`. The trusted gateway alias also pins
# `high`, so workflow nodes that ask for a lower default are not weakened.
THINKING_LEVEL = 'high'
THINKING_LEVEL_MAP = {'off': None, 'minimal': None, 'low': 'low',
                      'medium': 'medium', 'high': 'high', 'xhigh': None, 'max': None}
# OpenRouter's top provider for stepfun/step-5-preview: 1,000,000 input
# tokens and 64,000 completion tokens. The gateway drops the output cap.
CONTEXT_WINDOW = 1000000
MAX_OUTPUT_TOKENS = 64000
# Step's default tool profile, minus search_web: Frontier runs have no web
# access and Step's search uses a StepFun account credential.
EXCLUDED_TOOLS = 'search_web'
# Literal chat-template markers in model output become real control tokens
# upstream (see MAX_TRUNCATED_RESUMES).
SPECIAL_TOKEN_NOTE = """Never write a chat-template control marker literally, in reasoning, replies,
commands or files: the im_start / im_end markers wrapped in angle brackets and
pipes, or the tool_call, function and parameter tags (opening or closing).
The serving stack treats them as real control tokens and cuts your reply.
Refer to them in words, read them from tokenizer files, or build them by
string concatenation in code.
"""
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
""" + SPECIAL_TOKEN_NOTE
# Upstream serving turns a literal control marker in the model's output into a
# real control token, and long reasoning can hit the output cap: the reply ends
# with `stop` or `length` and no tool call, and Step's print mode exits as if
# the model had finished. While no final is accepted, resume the same session a
# bounded number of times.
MAX_TRUNCATED_RESUMES = 20
RESUME_PROMPT = """Your previous reply was cut off before any answer or tool call.
Continue the task from where you left off. The run ends only after
frontier-submit final returns acceptance.
""" + SPECIAL_TOKEN_NOTE
STRIPPED_PREFIXES = ('STEP_', 'STEPCODE_', 'STEPFUN_', 'PI_', 'GEMINI_', 'GOOGLE_',
                     'OPENAI_', 'ANTHROPIC_', 'AZURE_', 'AWS_', 'VERTEX', 'OPENROUTER_',
                     'MODAL_PROXY_')


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
        # Step resolves $NAME from the process environment at request time;
        # the value itself is never written to disk.
        'apiKey': '$FRONTIER_MODEL_PROXY_TOKEN',
        'models': [{
            'id': model, 'name': model, 'reasoning': True,
            'thinkingLevelMap': THINKING_LEVEL_MAP,
            'input': ['text', 'image'],
            'contextWindow': CONTEXT_WINDOW, 'maxTokens': MAX_OUTPUT_TOKENS,
            'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0},
            'compat': {
                # Step's native endpoint dialect: Chat Completions, system role.
                'supportsStore': False, 'supportsDeveloperRole': False,
                'supportsReasoningEffort': True, 'supportsUsageInStreaming': True,
                'maxTokensField': 'max_tokens', 'thinkingFormat': 'openrouter',
                'sendSessionAffinityHeaders': False, 'supportsLongCacheRetention': False,
            },
        }],
    }}}


def settings(model):
    """Step's unified config.toml: Pi settings plus Step permission policy."""
    return {
        'defaultProvider': PROVIDER, 'defaultModel': model,
        'defaultThinkingLevel': THINKING_LEVEL,
        'defaultProjectTrust': 'never',
        'packages': [], 'extensions': [], 'skills': [], 'prompts': [], 'themes': [],
        'enableAnalytics': False, 'quietStartup': True,
        # Full Access is the only Step tier that runs every command without an
        # approval channel; it requires the saved risk acknowledgment. The
        # Agent container is the sandbox, as for the other overlays.
        'permissionPreset': 'full-access', 'fullAccessAcknowledged': True,
        'nonInteractiveApproval': 'allow',
        # Driver owns the Agent deadline. One hour matches the other
        # overlays' request/idle bounds for long silent reasoning.
        'httpIdleTimeoutMs': 3600000,
        # Step's agent retry backoff is uncapped (base * 2^n): 8 retries wait
        # at most 510 s in total. 429s use Step's separate capped budget.
        'retry': {'enabled': True, 'maxRetries': 8, 'rateLimitMaxRetries': 16,
                  'baseDelayMs': 2000,
                  'provider': {'timeoutMs': 3600000, 'maxRetries': 0}},
        'compaction': {'enabled': True},
    }


def toml_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return '[' + ', '.join(toml_value(item) for item in value) + ']'
    raise TypeError(f'unsupported TOML value: {value!r}')


def toml_document(value, prefix=()):
    scalars = [f'{key} = {toml_value(item)}' for key, item in value.items()
               if not isinstance(item, dict)]
    lines = ([f"[{'.'.join(prefix)}]"] if prefix and scalars else []) + scalars
    for key, item in value.items():
        if isinstance(item, dict):
            lines.extend(([''] if lines else []) + toml_document(item, prefix + (key,)))
    return lines


def write_private(path, text):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as destination:
        destination.write(text)


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
    home = WORKSPACE / '.frontier-stepcode-home'
    home.mkdir(mode=0o700, exist_ok=False)
    temp = home / 'tmp'
    temp.mkdir(mode=0o700)
    plans = home / 'plans'
    plans.mkdir(mode=0o700)
    # Step keeps config.toml, models.json and auth.json beside (not inside)
    # its agent directory.
    storage = home / '.stepcode'
    storage.mkdir(mode=0o700)
    agent = storage / 'agent'
    agent.mkdir(mode=0o700)
    (agent / 'sessions').mkdir(mode=0o700)
    write_private(storage / 'models.json', json.dumps(models(model_url, model), indent=2) + '\n')
    write_private(storage / 'config.toml', '\n'.join(toml_document(settings(model))) + '\n')
    child = os.environ.copy()
    for name in tuple(child):
        if name.startswith(STRIPPED_PREFIXES) or name.endswith('_API_KEY') or name == 'AI_AGENT':
            child.pop(name)
    child.update({'HOME': str(home), 'STEP_CODING_AGENT_DIR': str(agent),
                  'STEP_CODING_AGENT_SESSION_DIR': str(agent / 'sessions'),
                  'STEP_CODING_AGENT_PLAN_DIR': str(plans),
                  # Report a blocked recoverable confirmation as a failed tool
                  # result instead of ending the unattended run.
                  'STEP_NON_INTERACTIVE_DENIAL': 'continue',
                  # Step's documented switch for optional catalog, update and
                  # install-telemetry work; inherited by subagent processes.
                  'STEPCODE_DISABLE_PI_SERVICES': '1',
                  'TMPDIR': str(temp), 'TMP': str(temp), 'TEMP': str(temp),
                  'PATH': f"{ROOT / 'bin'}:{child.get('PATH', '/usr/bin:/bin')}"})
    child.pop('LD_LIBRARY_PATH', None)
    return child


def command(resume=False):
    return [str(NODE), str(CLI), '-p', '--mode', 'json', '--no-approve', '--no-update-check',
            '--provider', PROVIDER, '--model', os.environ['FRONTIER_CODEX_MODEL'],
            '--thinking', THINKING_LEVEL, '--exclude-tools', EXCLUDED_TOOLS,
            '--no-extensions', '--no-prompt-templates', '--no-themes',
            '--no-skills', '--skill', str(ROOT / 'skills/frontier-submit'),
            '--skill', str(ROOT / 'skills/frontier-gpu'),
            *(['--continue'] if resume else []),
            '--', RESUME_PROMPT if resume else PROMPT]


def run_agent(argv, child):
    """Run Step, pass its JSON events through, and return the exit code and last assistant message."""
    last = None
    # Print mode reads piped stdin into the prompt; the Agent has no input stream.
    with subprocess.Popen(argv, cwd=WORKSPACE, env=child, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE) as process:
        for line in process.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            if line.startswith(b'{"type":"message_end"'):
                try:
                    message = json.loads(line)['message']
                except (ValueError, KeyError, TypeError):
                    continue
                if isinstance(message, dict) and message.get('role') == 'assistant':
                    last = message
    return process.returncode, last


def truncated_reply(message):
    """A `stop` or `length` reply without a tool call.

    Upstream cuts replies at literal control markers, sometimes after a few
    words of answer text, and long reasoning can exhaust the 64,000-token
    output cap; Step's print mode ends the run on either.
    """
    if not isinstance(message, dict) or message.get('stopReason') not in ('stop', 'length'):
        return False
    blocks = message.get('content')
    if not isinstance(blocks, list):
        return False
    return all(isinstance(block, dict) and block.get('type') != 'toolCall' for block in blocks)


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
        returncode, last = run_agent(command(), child)
        for attempt in range(MAX_TRUNCATED_RESUMES):
            if returncode != 0 or accepted_final_receipt() or not truncated_reply(last):
                break
            print(f'frontier-stepcode-launch: resuming after truncated reply '
                  f'({attempt + 1}/{MAX_TRUNCATED_RESUMES})', file=sys.stderr)
            returncode, last = run_agent(command(resume=True), child)
    except (LaunchError, OSError) as error:
        print(f'frontier-stepcode-launch: {error}', file=sys.stderr)
        return 2
    if accepted_final_receipt():
        return 0
    if returncode == 0:
        print('frontier-stepcode-launch: no accepted final submission receipt', file=sys.stderr)
        return 2
    return returncode if 0 < returncode < 256 else 2


if __name__ == '__main__':
    raise SystemExit(main())
