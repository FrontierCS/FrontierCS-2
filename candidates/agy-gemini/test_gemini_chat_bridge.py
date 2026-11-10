import base64
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemini_chat_bridge as bridge  # noqa: E402

SIG = base64.b64encode(b'sig').decode()


class RequestTranslation(unittest.TestCase):
    def test_history_tools_and_effort(self):
        body = {
            'systemInstruction': {'parts': [{'text': 'sys'}]},
            'contents': [
                {'role': 'user', 'parts': [{'text': 'hi'}]},
                {'role': 'model', 'parts': [
                    {'text': 'hidden', 'thought': True},
                    {'functionCall': {'id': f'call_1__thought__{SIG}', 'name': 'run_command',
                                      'args': {'CommandLine': 'ls'}}, 'thoughtSignature': SIG}]},
                # agy places tool results in a model-role content.
                {'role': 'model', 'parts': [{'functionResponse': {
                    'id': f'call_1__thought__{SIG}', 'name': 'run_command',
                    'response': {'output': 'a.txt'}}}]},
            ],
            'tools': [{'functionDeclarations': [
                {'name': 'run_command', 'description': 'd', 'parametersJsonSchema': {
                    '$schema': 'x', 'type': 'object', 'properties': {}}},
                {'name': 'search_web', 'parametersJsonSchema': {'type': 'object'}},
                {'name': 'ask_question', 'parametersJsonSchema': {'type': 'object'}},
                {'name': 'legacy', 'parameters': {'type': 'OBJECT', 'properties': {
                    'n': {'type': 'INTEGER'}}}}]}],
            'generationConfig': {'maxOutputTokens': 65536,
                                 'thinkingConfig': {'thinkingLevel': 'HIGH', 'includeThoughts': True}},
        }
        request = bridge.gemini_to_chat(body, 'alias-x')
        self.assertEqual(request['model'], 'alias-x')
        self.assertEqual(request['reasoning_effort'], 'high')
        self.assertEqual(request['max_tokens'], 65536)
        self.assertEqual([m['role'] for m in request['messages']], ['system', 'user', 'assistant', 'tool'])
        self.assertIsNone(request['messages'][2]['content'])
        self.assertEqual(request['messages'][2]['tool_calls'][0]['id'], f'call_1__thought__{SIG}')
        self.assertEqual(request['messages'][3], {'role': 'tool', 'tool_call_id': f'call_1__thought__{SIG}',
                                                  'content': 'a.txt'})
        names = [t['function']['name'] for t in request['tools']]
        self.assertEqual(names, ['run_command', 'legacy'])
        self.assertNotIn('$schema', request['tools'][0]['function']['parameters'])
        self.assertEqual(request['tools'][1]['function']['parameters']['properties']['n']['type'], 'integer')

    def test_replayed_call_carries_thought_signature(self):
        sig = base64.b64encode(b'sig-bytes').decode()
        body = {'contents': [
            {'role': 'user', 'parts': [{'text': 'go'}]},
            {'role': 'model', 'parts': [
                {'functionCall': {'id': 'call_a__thought__' + sig, 'name': 'read_file', 'args': {}}},
                {'functionCall': {'id': 'call_b', 'name': 'read_file', 'args': {}}, 'thoughtSignature': sig},
                {'functionCall': {'id': 'call_c', 'name': 'read_file', 'args': {}}}]},
        ]}
        calls = bridge.gemini_to_chat(body, 'm')['messages'][-1]['tool_calls']
        self.assertEqual([c.get('provider_specific_fields') for c in calls],
                         [{'thought_signature': sig}, {'thought_signature': sig}, None])
        self.assertEqual(calls[0]['id'], 'call_a__thought__' + sig)

    def test_images_become_content_parts(self):
        body = {'contents': [{'role': 'user', 'parts': [
            {'text': 'look'}, {'inlineData': {'mimeType': 'image/png', 'data': 'AAAA'}}]}]}
        message = bridge.gemini_to_chat(body, 'a')['messages'][0]
        self.assertEqual(message['content'][1]['image_url']['url'], 'data:image/png;base64,AAAA')


class ResponseTranslation(unittest.TestCase):
    def test_stream(self):
        chunks = [
            {'model': 'a', 'choices': [{'index': 0, 'delta': {'reasoning_content': 'think'}}]},
            {'choices': [{'index': 0, 'delta': {'content': 'ok'}}]},
            {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': f'c__thought__{SIG}',
                                                                'function': {'name': 'f', 'arguments': '{"a"'}}]}}]},
            {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': ': 1}'}}]},
                          'finish_reason': 'tool_calls'}]},
            {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 7, 'total_tokens': 17,
                                      'completion_tokens_details': {'reasoning_tokens': 5}}},
        ]
        lines = []
        for chunk in chunks:
            lines += [b'data: ' + json.dumps(chunk).encode() + b'\n', b'\n']
        lines += [b'data: [DONE]\n', b'\n']
        out = []
        summary = bridge.chat_stream_to_gemini(bridge.sse_events(iter(lines)), out.append)
        self.assertEqual(out[0]['candidates'][0]['content']['parts'], [{'text': 'think', 'thought': True}])
        final = out[-1]['candidates'][0]
        call = final['content']['parts'][0]
        self.assertEqual(call['functionCall'], {'name': 'f', 'args': {'a': 1}, 'id': f'c__thought__{SIG}'})
        self.assertEqual(call['thoughtSignature'], SIG)
        self.assertEqual(final['finishReason'], 'STOP')
        self.assertEqual(out[-1]['usageMetadata'], {'promptTokenCount': 10, 'candidatesTokenCount': 2,
                                                    'thoughtsTokenCount': 5, 'totalTokenCount': 17})
        self.assertEqual(summary['tool_calls'], 1)
        merged = bridge.merge_gemini(out)
        self.assertEqual(len(merged['candidates'][0]['content']['parts']), 3)

    def test_stream_error_raises(self):
        lines = [b'data: {"error": {"message": "boom"}}\n', b'\n']
        with self.assertRaises(bridge.BridgeError):
            bridge.chat_stream_to_gemini(bridge.sse_events(iter(lines)), lambda _: None)

    def test_invalid_signature_dropped(self):
        self.assertIsNone(bridge._signature({'id': 'c__thought__not base64!'}))


if __name__ == '__main__':
    unittest.main()


class UpstreamRetry(unittest.TestCase):
    """Retry policy mirrors pi-gemini: 16 retries before any byte reaches agy."""

    def serve(self, statuses):
        import http.server
        import threading
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers.get('content-length') or 0))
                status = statuses[min(len(seen), len(statuses) - 1)]
                seen.append(status)
                body = b'data: [DONE]\n\n' if status == 200 else b'{"error":{"message":"busy"}}'
                self.send_response(status)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_address[1]}/v1', seen

    def setUp(self):
        self.old = bridge.RETRY_BASE_S
        bridge.RETRY_BASE_S = 0.0

    def tearDown(self):
        bridge.RETRY_BASE_S = self.old

    def test_transient_status_is_retried(self):
        url, seen = self.serve([503, 429, 200])
        b = bridge.Bridge(url, 'tok', 'gemini-4-argon', 'nonce')
        connection, response, retries = b.open_upstream(b'{}')
        connection.close()
        self.assertEqual((response.status, retries, seen), (200, 2, [503, 429, 200]))

    def test_client_error_is_not_retried(self):
        url, seen = self.serve([400])
        b = bridge.Bridge(url, 'tok', 'gemini-4-argon', 'nonce')
        with self.assertRaises(bridge.BridgeError) as caught:
            b.open_upstream(b'{}')
        self.assertEqual((caught.exception.status, seen), (400, [400]))

    def test_retry_budget_is_bounded(self):
        url, seen = self.serve([502])
        b = bridge.Bridge(url, 'tok', 'gemini-4-argon', 'nonce')
        with self.assertRaises(bridge.BridgeError):
            b.open_upstream(b'{}')
        self.assertEqual(len(seen), bridge.RETRY_MAX + 1)
