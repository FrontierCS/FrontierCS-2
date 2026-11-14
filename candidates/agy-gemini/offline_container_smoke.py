#!/usr/bin/env python3
"""Offline smoke: real launcher + real agy + bridge against loopback fakes.

Run in an unprivileged, network-disabled container with a read-only root and a
writable /workspace.  The fake model speaks OpenAI Chat Completions exactly
like the Driver proxy -> LiteLLM path (thought signatures embedded in tool-call
ids); the fake Driver implements the payload/train/checkpoint/final routes the
shared frontier-submit client uses.
"""

import base64
import hashlib
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading

CAPABILITY = 'offline-driver-capability'
PROXY_TOKEN = 'offline-proxy-token'
ALIAS = 'fixture-gemini'
SIG = base64.b64encode(b'fixture-thought-signature').decode()
CALLS = [
    ('run_command', {'CommandLine': 'printf fixture-ok > /workspace/solution.txt && tar -cf /workspace/final.tar -C /workspace solution.txt',
                     'toolSummary': 'Create answer archive', 'WaitMsBeforeAsync': 10000}),
    ('read_file', {'Path': '/frontier/task/README.md'}),
    ('run_command', {'CommandLine': 'frontier-submit train /workspace/final.tar && frontier-submit store-final /workspace/final.tar && frontier-submit final /workspace/final.tar',
                     'toolSummary': 'Submit selected answer', 'WaitMsBeforeAsync': 10000}),
]
state = {'main': 0, 'aux': 0, 'errors': [], 'replayed_ids': [], 'tool_results': [],
         'efforts': set(), 'tools_offered': None}
submissions = []
payloads = {}
lock = threading.Lock()


def call_id(index):
    return f'call_fixture_{index}__thought__{SIG}'


class Model(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def do_POST(self):
        try:
            assert self.path == '/v1/chat/completions', self.path
            assert self.headers['authorization'] == f'Bearer {PROXY_TOKEN}'
            assert self.headers['content-type'] == 'application/json'
            body = json.loads(self.rfile.read(int(self.headers['content-length'])))
            assert body['model'] == ALIAS, body['model']
            assert body['stream'] is True and body['stream_options'] == {'include_usage': True}
            with lock:
                if body.get('reasoning_effort'):
                    state['efforts'].add(body['reasoning_effort'])
                is_main = bool(body.get('tools'))
                if is_main:
                    names = sorted(t['function']['name'] for t in body['tools'])
                    state['tools_offered'] = names
                    assert 'search_web' not in names and 'ask_question' not in names, names
                    for message in body['messages']:
                        for call in message.get('tool_calls') or []:
                            state['replayed_ids'].append(call['id'])
                        if message.get('role') == 'tool':
                            state['tool_results'].append(message)
                    index = state['main']
                    state['main'] += 1
                else:
                    state['aux'] += 1
                    index = None
        except Exception as error:  # noqa: BLE001
            with lock:
                state['errors'].append(repr(error))
            self.send_response(400)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.end_headers()

        def frame(delta, finish=None):
            chunk = {'id': 'chatcmpl_fixture', 'object': 'chat.completion.chunk', 'model': ALIAS,
                     'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
            self.wfile.write(b'data: ' + json.dumps(chunk).encode() + b'\n\n')

        frame({'role': 'assistant', 'reasoning_content': 'Offline fixture reasoning.'})
        if index is not None and index < len(CALLS):
            name, args = CALLS[index]
            text = json.dumps(args)
            frame({'tool_calls': [{'index': 0, 'id': call_id(index), 'type': 'function',
                                   'function': {'name': name, 'arguments': text[:20]}}]})
            frame({'tool_calls': [{'index': 0, 'function': {'arguments': text[20:]}}]})
            frame({}, 'tool_calls')
        else:
            frame({'content': 'Fixture complete.' if index is not None else '{"severity": 0, "confidence": 1, "reason": "safe"}'})
            frame({}, 'stop')
        usage = {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8,
                 'completion_tokens_details': {'reasoning_tokens': 1}}
        self.wfile.write(b'data: ' + json.dumps({'id': 'chatcmpl_fixture', 'choices': [], 'usage': usage}).encode() + b'\n\n')
        self.wfile.write(b'data: [DONE]\n\n')


class Driver(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        assert self.headers['authorization'] == f'Bearer {CAPABILITY}'
        raw = self.rfile.read(int(self.headers['content-length']))
        if self.path == '/v1/payloads':
            digest = 'sha256:' + hashlib.sha256(raw).hexdigest()
            assert self.headers['x-content-sha256'] == digest
            payload_id = 'payload_' + digest[7:27]
            payloads[payload_id] = digest
            response = {'payload_id': payload_id}
        else:
            body = json.loads(raw)
            assert body['payload_id'] in payloads
            submissions.append(self.path)
            response = {
                '/v1/verifier-train': {'verification_id': 'verification_train_fixture',
                                       'state': 'running', 'remaining_train_calls': 1023},
                '/v1/submissions/checkpoint': {'remaining_runtime_s': 17900,
                                               'remaining_checkpoint_submissions': None},
                '/v1/submissions': {'verification_id': 'verification_final_fixture'},
            }[self.path]
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    assert os.getuid() != 0
    try:
        tempfile.mkdtemp(dir='/tmp')
        raise AssertionError('root filesystem /tmp must be read-only')
    except OSError:
        pass
    servers = [http.server.ThreadingHTTPServer(('127.0.0.1', 8082), Model),
               http.server.ThreadingHTTPServer(('127.0.0.1', 8080), Driver)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8',
           'FRONTIER_DRIVER_API_URL': 'http://127.0.0.1:8080',
           'FRONTIER_DRIVER_TOKEN_FILE': '/var/run/frontier/token',
           'FRONTIER_MODEL_API_BASE_URL': 'http://127.0.0.1:8082/v1',
           'FRONTIER_MODEL_PROXY_TOKEN': PROXY_TOKEN, 'FRONTIER_CODEX_MODEL': ALIAS,
           'GEMINI_API_KEY': 'ambient-must-not-leak',
           'TMPDIR': '/tmp'}
    completed = subprocess.run(['/opt/frontier/candidate/launch'], cwd='/workspace', env=env,
                               stdin=subprocess.DEVNULL, capture_output=True, timeout=300)
    output = completed.stdout.decode(errors='replace') + completed.stderr.decode(errors='replace')
    bridge_log = open('/workspace/.frontier-agy-home/model-bridge.jsonl').read()
    assert completed.returncode == 0, output[-4000:] + bridge_log[-3000:]
    assert not state['errors'], state['errors']
    assert 'ambient-must-not-leak' not in output and PROXY_TOKEN not in output
    assert state['main'] == len(CALLS) + 1, (state['main'], output[-3000:])
    expected = [call_id(i) for i in range(len(CALLS))]
    assert all(i in state['replayed_ids'] for i in expected), state['replayed_ids']
    assert len(state['tool_results']) >= len(CALLS)
    tool_ids = {m['tool_call_id'] for m in state['tool_results']}
    assert set(expected) <= tool_ids, tool_ids
    readme = [m for m in state['tool_results'] if m['tool_call_id'] == expected[1]][-1]
    assert 'Offline fixture' in readme['content'], readme
    assert submissions == ['/v1/verifier-train', '/v1/submissions/checkpoint', '/v1/submissions'], submissions
    receipt = json.load(open('/workspace/.frontier/final-submission.json'))
    assert receipt['state'] == 'accepted' and receipt['operation'] == 'final'
    assert open('/workspace/solution.txt').read() == 'fixture-ok'
    assert PROXY_TOKEN not in bridge_log
    print(json.dumps({'status': 'READONLY_FIXTURE_PASSED', 'uid': os.getuid(),
                      'main_requests': state['main'], 'aux_requests': state['aux'],
                      'efforts': sorted(state['efforts']), 'tools_offered': state['tools_offered'],
                      'replayed_signature_ids': len(expected), 'submissions': submissions,
                      'real_provider_requests': 0}))
    for server in servers:
        server.shutdown()


if __name__ == '__main__':
    main()
