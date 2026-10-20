"""Exact LiteLLM proxy with a local, credential-free Meta Responses fixture."""
import asyncio
import http.server
import json
import os
from pathlib import Path
import select
import threading
import time

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['LITELLM_TELEMETRY'] = 'False'
import litellm
from litellm.proxy import proxy_server

root = Path('/fixture')
scope = json.loads((root / 'scope.json').read_text())
state = {'requests': [], 'observer_calls': 0, 'tool_replayed': False,
         'reasoning_replayed': False, 'tool_sent': False, 'held': 0, 'closed': 0,
         'mode': 'normal'}
lock = threading.Lock()


class MetaFixture(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        assert self.path == '/v1/responses'
        assert self.headers['Authorization'] == 'Bearer offline-provider-dummy'
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert body['model'] == 'fixture-only'
        with lock:
            state['requests'].append(body)
            number = len(state['requests'])
            state['observer_calls'] += int(any(
                child.get('name') == 'submit_reminder_decision'
                for tool in body.get('tools', []) for child in tool.get('tools', [])))
            state['tool_replayed'] |= any(
                item.get('type') == 'function_call_output' and item.get('call_id') == 'call_fixture'
                for item in body.get('input', []) if isinstance(item, dict))
            state['reasoning_replayed'] |= any(
                item.get('type') == 'reasoning' and item.get('encrypted_content') == 'offline-opaque-reasoning'
                for item in body.get('input', []) if isinstance(item, dict))
            shell = any(tool.get('name') == 'bash' or any(
                child.get('name') == 'bash' for child in tool.get('tools', []))
                for tool in body.get('tools', []))
            send_tool = shell and not state['tool_sent'] and 'FRONTIER_OFFLINE_TOOL_MARKER' in json.dumps(body)
            state['tool_sent'] |= send_tool
            hold = state['mode'] == 'hold'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        response = {'id': f'resp_fixture_{number}', 'object': 'response',
                    'created_at': 1, 'model': 'fixture-only', 'status': 'in_progress', 'output': []}

        def emit(event):
            self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
            self.wfile.flush()

        emit({'type': 'response.created', 'sequence_number': 1, 'response': response})
        if hold:
            with lock:
                state['held'] += 1
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if select.select([self.connection], [], [], .05)[0]:
                    if not self.connection.recv(1):
                        with lock:
                            state['closed'] += 1
                        return
            return
        if send_tool:
            emit({'type': 'response.output_item.done', 'sequence_number': 2,
                  'output_index': 0, 'item': {'type': 'reasoning', 'id': 'rs_fixture',
                  'summary': [], 'encrypted_content': 'offline-opaque-reasoning'}})
            emit({'type': 'response.function_call_arguments.done', 'sequence_number': 3,
                  'output_index': 1, 'item_id': 'fc_fixture', 'name': 'bash', 'call_id': 'call_fixture',
                  'arguments': json.dumps({'command': 'printf fixture-ok > official-tool-result.txt',
                                          'description': 'Write harmless fixture artifact'})})
        else:
            emit({'type': 'response.output_text.delta', 'sequence_number': 2,
                  'output_index': 0, 'item_id': f'msg_{number}', 'content_index': 0,
                  'delta': 'Fixture complete.'})
        emit({'type': 'response.completed', 'sequence_number': 4, 'response': {
            **response, 'status': 'completed', 'usage': {'input_tokens': 7,
            'output_tokens': 11, 'total_tokens': 18, 'input_tokens_details': {'cached_tokens': 2},
            'output_tokens_details': {'reasoning_tokens': 5}}}})


meta = http.server.ThreadingHTTPServer(('127.0.0.1', 0), MetaFixture)
threading.Thread(target=meta.serve_forever, daemon=True).start()
config = '''model_list:
  - model_name: fixture-only
    litellm_params:
      model: openai/fixture-only
      api_key: offline-provider-dummy
      api_base: http://127.0.0.1:%s/v1
litellm_settings:
  callbacks: [fixture_callbacks.recorder]
  num_retries: 0
general_settings:
  master_key: sk-offline-gateway-dummy
''' % meta.server_port
(root / 'config.yaml').write_text(config)
litellm.register_model({'openai/fixture-only': {'litellm_provider': 'openai',
    'mode': 'responses', 'supports_native_streaming': True}})


@proxy_server.app.get('/fixture/state')
async def get_state():
    with lock:
        return dict(state)


@proxy_server.app.post('/fixture/hold')
async def hold():
    with lock:
        state['mode'] = 'hold'
    return {'ok': True}


@proxy_server.app.post('/fixture/normal')
async def normal():
    with lock:
        state['mode'] = 'normal'
    return {'ok': True}


async def main():
    import uvicorn
    await proxy_server.initialize(config=str(root / 'config.yaml'), telemetry=False)
    await uvicorn.Server(uvicorn.Config(proxy_server.app, host='0.0.0.0',
                                      port=scope['gateway_port'], log_level='warning')).serve()


asyncio.run(main())
