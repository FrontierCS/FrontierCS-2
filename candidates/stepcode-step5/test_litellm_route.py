"""Opt-in LiteLLM check of the reviewed step-5-preview route.

Runs `gateway-models.json` through `litellm.Router` against a loopback
OpenRouter-shaped Chat Completions fixture. Set `FRONTIER_TEST_LITELLM=1` where
the gateway's pinned LiteLLM version is importable, for example inside the
pinned LiteLLM image with this directory mounted read-only. The upstream URL
and key reference are replaced by loopback dummies; no provider is called.
"""
import asyncio
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import unittest

ROUTES = json.loads(Path(__file__).with_name('gateway-models.json').read_text(encoding='utf-8'))
ALIAS = 'step-5-preview'
UPSTREAM_MODEL = 'stepfun/step-5-preview'


def chat_stream():
    base = {'id': 'gen-fixture', 'object': 'chat.completion.chunk', 'created': 1, 'model': UPSTREAM_MODEL}
    choice = lambda delta, finish=None: dict(base, choices=[{'index': 0, 'delta': delta, 'finish_reason': finish}])
    call = {'index': 0, 'id': 'call_next', 'type': 'function',
            'function': {'name': 'run_command', 'arguments': '{"command":"ls"}'}}
    chunks = [choice({'role': 'assistant', 'reasoning': 'thinking'}), choice({'tool_calls': [call]}),
              choice({}, 'tool_calls'),
              dict(base, choices=[], usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15,
                                            'completion_tokens_details': {'reasoning_tokens': 3}})]
    return ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks).encode() + b'data: [DONE]\n\n'


def step_request(effort):
    """The Chat Completions body Step Code sends with thinkingFormat openrouter."""
    tool = {'type': 'function', 'function': {'name': 'run_command', 'description': 'run',
            'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}}}}
    return {'messages': [{'role': 'system', 'content': 'fixture system'},
                         {'role': 'user', 'content': [{'type': 'text', 'text': 'fixture task'}]},
                         {'role': 'assistant', 'content': None, 'reasoning_content': 'earlier',
                          'tool_calls': [{'id': 'call_first', 'type': 'function',
                                          'function': {'name': 'run_command', 'arguments': '{"command":"pwd"}'}}]},
                         {'role': 'tool', 'tool_call_id': 'call_first', 'content': '/workspace\n'}],
            'stream': True, 'stream_options': {'include_usage': True}, 'max_tokens': 64000,
            'tools': [tool], 'reasoning': {'effort': effort}}


@unittest.skipUnless(os.environ.get('FRONTIER_TEST_LITELLM'), 'Requires the pinned LiteLLM package')
class LiteLLMRouteTests(unittest.TestCase):
    def test_route_pins_high_reasoning_and_drops_client_output_cap(self):
        for effort in ('high', 'medium'):
            with self.subTest(effort=effort):
                body, chunks = self.forward(step_request(effort))
                self.assertEqual(body['model'], UPSTREAM_MODEL)
                self.assertEqual(body['reasoning'], {'effort': 'high'})
                for name in ('max_tokens', 'max_completion_tokens', 'reasoning_effort'):
                    self.assertNotIn(name, body)
                self.assertEqual(body['tools'][0]['function']['name'], 'run_command')
                replay = [m for m in body['messages'] if m['role'] == 'assistant'][0]
                self.assertEqual(replay['tool_calls'][0]['id'], 'call_first')
                deltas = [c.choices[0].delta for c in chunks if c.choices]
                self.assertTrue(any(getattr(d, 'reasoning_content', None) == 'thinking' for d in deltas))
                usage = [c.usage for c in chunks if getattr(c, 'usage', None)]
                self.assertEqual(usage[-1].total_tokens, 15)

    def forward(self, request):
        import litellm
        captured = []

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers['content-length'])
                captured.append((self.path, self.headers.get('authorization'),
                                 json.loads(self.rfile.read(length))))
                payload = chat_stream()
                self.send_response(200)
                self.send_header('content-type', 'text/event-stream')
                self.send_header('content-length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            routes = copy.deepcopy(ROUTES['model_list'])
            routes[0]['litellm_params']['api_base'] = f'http://127.0.0.1:{server.server_port}/api/v1'
            routes[0]['litellm_params']['api_key'] = 'loopback-dummy'
            router = litellm.Router(model_list=routes)

            async def run():
                stream = await router.acompletion(model=ALIAS, **request)
                return [chunk async for chunk in stream]

            chunks = asyncio.run(run())
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(len(captured), 1)
        path, authorization, body = captured[0]
        self.assertEqual(path, '/api/v1/chat/completions')
        self.assertEqual(authorization, 'Bearer loopback-dummy')
        return body, chunks


if __name__ == '__main__':
    unittest.main()
