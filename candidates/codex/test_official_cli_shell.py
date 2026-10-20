"""Opt-in actual official CLI shell policy test against a local Responses fixture."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock


def events(output):
    rows = [{'type': 'response.created', 'response': {'id': 'resp_shell', 'status': 'in_progress', 'output': []}}]
    for index, item in enumerate(output):
        rows.extend([{'type': 'response.output_item.added', 'output_index': index, 'item': item},
                     {'type': 'response.output_item.done', 'output_index': index, 'item': item}])
    rows.append({'type': 'response.completed', 'response': {'id': 'resp_shell', 'status': 'completed', 'output': output}})
    return ''.join('event: ' + row['type'] + '\ndata: ' + json.dumps(row) + '\n\n' for row in rows).encode()


@unittest.skipUnless(os.environ.get('FRONTIER_TEST_CODEX_BINARY'), 'Requires pinned official Codex executable')
class OfficialShellTests(unittest.TestCase):
    def test_login_denied_then_omitted_login_resolves_helper(self):
        binary = Path(os.environ['FRONTIER_TEST_CODEX_BINARY'])
        self.assertEqual(hashlib.sha256(binary.read_bytes()).hexdigest(),
                         '628e35487d888a28f0427fc37501da529dd3d81fcbad7a2c3421c1e8670d92e1')
        spec = importlib.util.spec_from_file_location('shell_launcher', Path(__file__).with_name('launcher.py'))
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['content-length'])))
                requests.append(body)
                if len(requests) <= 2:
                    arguments = {'cmd': 'frontier-submit', 'shell': '/bin/sh'}
                    if len(requests) == 1:
                        arguments['login'] = True
                    output = [{'id': 'fc_shell_' + str(len(requests)), 'type': 'function_call',
                               'call_id': 'call_shell_' + str(len(requests)), 'name': 'exec_command',
                               'arguments': json.dumps(arguments), 'status': 'completed'}]
                else:
                    output = [{'id': 'msg_done', 'type': 'message', 'role': 'assistant',
                               'content': [{'type': 'output_text', 'text': 'SHELL_POLICY_OK', 'annotations': []}]}]
                raw = events(output)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                tools = directory / 'tools'
                tools.mkdir()
                helper = tools / 'frontier-submit'
                helper.write_text('#!/bin/sh\nprintf "HELPER_RESOLVED\\n"\n')
                helper.chmod(0o755)
                launcher.CODEX = binary
                launcher.WORKSPACE = directory
                with mock.patch.dict(os.environ, {'FRONTIER_CODEX_MODEL': 'gpt-6-astra',
                                                 'FRONTIER_CODEX_REASONING_EFFORT': 'xhigh'}):
                    argv = launcher.command('http://127.0.0.1:' + str(server.server_port) + '/v1')
                environment = {key: value for key, value in os.environ.items()
                               if not any(word in key.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL'))}
                environment.update(HOME=str(directory), CODEX_HOME=str(directory),
                                   PATH=str(tools) + ':' + environment.get('PATH', '/usr/bin:/bin'),
                                   FRONTIER_MODEL_PROXY_TOKEN='offline-dummy-not-a-credential')
                process = subprocess.run(argv, input='Perform the supplied shell-policy fixture.\n',
                                         capture_output=True, text=True, env=environment, timeout=25)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(len(requests), 3)
                returned = [item for request in requests[1:] for item in request.get('input', [])
                            if isinstance(item, dict) and item.get('type') == 'function_call_output']
                by_call = {item['call_id']: item['output'] for item in returned}
                self.assertIn('login', str(by_call['call_shell_1']).lower())
                self.assertNotIn('HELPER_RESOLVED', str(by_call['call_shell_1']))
                self.assertIn('HELPER_RESOLVED', str(by_call['call_shell_2']))
                self.assertIn('SHELL_POLICY_OK', process.stdout)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
