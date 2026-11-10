#!/usr/bin/env python3
"""Run-local Gemini-API to OpenAI Chat Completions bridge for the agy CLI.

The official Antigravity CLI (agy) speaks only the native Gemini API
(``POST /v1beta/models/{model}:streamGenerateContent?alt=sse`` with an
``x-goog-api-key`` header).  The Frontier Driver model proxy serves only its
frozen OpenAI/Anthropic paths, so this loopback bridge translates each native
request into one streamed ``/v1/chat/completions`` request against the
run-local proxy and translates the stream back.

Invariants:
* every upstream request names the CandidateVersion's model alias, whatever
  model slug agy chose (main agent, title generator or safety monitors);
* the run-scoped proxy token stays in this process; agy authenticates to the
  bridge with a random per-run loopback nonce;
* tool-call ids are replayed verbatim.  LiteLLM embeds Gemini thought
  signatures in them (``call_x__thought__<b64>``); agy echoes ``functionCall.id``
  and ``functionResponse.id`` unchanged, so signatures survive the round trip;
* web search and interactive questions are not offered to the model.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import http.client
import http.server
import json
import re
import sys
import threading
import time
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit

THOUGHT_SEPARATOR = '__thought__'
# search_web reaches Google's search backend; ask_question waits for a human.
BLOCKED_TOOLS = frozenset({'search_web', 'ask_question'})
PATH_RE = re.compile(r'^/v1beta/models/([A-Za-z0-9._-]+):(streamGenerateContent|generateContent)$')
EFFORTS = {'MINIMAL': 'minimal', 'LOW': 'low', 'MEDIUM': 'medium', 'HIGH': 'high'}
STATUS = {400: 'INVALID_ARGUMENT', 401: 'UNAUTHENTICATED', 403: 'PERMISSION_DENIED',
          404: 'NOT_FOUND', 408: 'DEADLINE_EXCEEDED', 409: 'ABORTED',
          429: 'RESOURCE_EXHAUSTED', 500: 'INTERNAL', 502: 'UNAVAILABLE',
          503: 'UNAVAILABLE', 504: 'DEADLINE_EXCEEDED'}
FINISH = {'stop': 'STOP', 'tool_calls': 'STOP', 'function_call': 'STOP',
          'length': 'MAX_TOKENS', 'content_filter': 'SAFETY'}
UPSTREAM_TIMEOUT_S = 3600
RETRY_MAX = 16
RETRY_BASE_S = 2.0
RETRY_CAP_S = 60.0
RETRY_STATUS = frozenset({408, 409, 425, 429})
MAX_REQUEST_BYTES = 256 * 1024 * 1024


class BridgeError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# Request translation (Gemini -> Chat Completions)
# --------------------------------------------------------------------------

def _lower_schema(value: Any) -> Any:
    """Gemini OpenAPI ``parameters`` use upper-case type names."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == 'type' and isinstance(item, str):
                out[key] = item.lower()
            else:
                out[key] = _lower_schema(item)
        return out
    if isinstance(value, list):
        return [_lower_schema(item) for item in value]
    return value


def _tools(body: dict) -> list[dict]:
    tools = []
    for group in body.get('tools') or []:
        if not isinstance(group, dict):
            continue
        for declaration in group.get('functionDeclarations') or []:
            name = declaration.get('name')
            if not isinstance(name, str) or name in BLOCKED_TOOLS:
                continue
            if 'parametersJsonSchema' in declaration:
                parameters = dict(declaration['parametersJsonSchema'] or {})
            elif 'parameters' in declaration:
                parameters = _lower_schema(declaration['parameters'] or {})
            else:
                parameters = {'type': 'object', 'properties': {}}
            parameters.pop('$schema', None)
            function = {'name': name, 'parameters': parameters}
            if declaration.get('description'):
                function['description'] = declaration['description']
            tools.append({'type': 'function', 'function': function})
        # Server-side Gemini tools (googleSearch, urlContext, codeExecution, ...)
        # are deliberately not forwarded.
    return tools


def _media_part(blob: dict) -> dict | None:
    mime = blob.get('mimeType') or 'application/octet-stream'
    data = blob.get('data')
    if not isinstance(data, str):
        return None
    url = f'data:{mime};base64,{data}'
    if mime.startswith('image/'):
        return {'type': 'image_url', 'image_url': {'url': url}}
    return {'type': 'file', 'file': {'file_data': url}}


def _tool_content(response: Any) -> str:
    if isinstance(response, dict) and set(response) == {'output'} and isinstance(response['output'], str):
        return response['output']
    return json.dumps(response, ensure_ascii=False)


def _user_content(chunks: list[dict]) -> str | list[dict]:
    if all(chunk['type'] == 'text' for chunk in chunks):
        return '\n'.join(chunk['text'] for chunk in chunks)
    return chunks


def gemini_to_chat(body: dict, model_alias: str, stream: bool = True) -> dict:
    messages: list[dict] = []
    system = body.get('systemInstruction')
    if isinstance(system, dict):
        text = '\n'.join(part['text'] for part in system.get('parts') or []
                         if isinstance(part, dict) and isinstance(part.get('text'), str))
        if text:
            messages.append({'role': 'system', 'content': text})
    elif isinstance(system, str) and system:
        messages.append({'role': 'system', 'content': system})

    generated_ids = 0
    pending_ids: dict[str, list[str]] = {}
    for content in body.get('contents') or []:
        role = content.get('role') or 'user'
        texts: list[str] = []
        user_chunks: list[dict] = []
        tool_calls: list[dict] = []
        tool_messages: list[dict] = []
        trailing_media: list[dict] = []
        for part in content.get('parts') or []:
            if not isinstance(part, dict):
                continue
            if part.get('thought') is True:
                continue  # provider reasoning is never replayed as content
            if 'functionCall' in part:
                call = part['functionCall'] or {}
                call_id = call.get('id')
                if not isinstance(call_id, str) or not call_id:
                    generated_ids += 1
                    call_id = f'call_agy_{generated_ids}'
                    pending_ids.setdefault(call.get('name', ''), []).append(call_id)
                tool_call = {'id': call_id, 'type': 'function', 'function': {
                    'name': call.get('name', ''),
                    'arguments': json.dumps(call.get('args') or {}, ensure_ascii=False)}}
                # The official Gemini API rejects a replayed functionCall without its thought
                # signature, and LiteLLM only forwards one given as provider_specific_fields.
                signature = (_valid_signature(part.get('thoughtSignature'))
                             or _signature({'id': call_id}))
                if signature:
                    tool_call['provider_specific_fields'] = {'thought_signature': signature}
                tool_calls.append(tool_call)
            elif 'functionResponse' in part:
                response = part['functionResponse'] or {}
                call_id = response.get('id')
                if not isinstance(call_id, str) or not call_id:
                    queue = pending_ids.get(response.get('name', ''), [])
                    call_id = queue.pop(0) if queue else f'call_agy_orphan_{generated_ids}'
                tool_messages.append({'role': 'tool', 'tool_call_id': call_id,
                                      'content': _tool_content(response.get('response'))})
                for nested in response.get('parts') or []:
                    if isinstance(nested, dict) and isinstance(nested.get('inlineData'), dict):
                        media = _media_part(nested['inlineData'])
                        if media:
                            trailing_media.append(media)
            elif isinstance(part.get('text'), str):
                texts.append(part['text'])
                user_chunks.append({'type': 'text', 'text': part['text']})
            elif isinstance(part.get('inlineData'), dict):
                media = _media_part(part['inlineData'])
                if media:
                    user_chunks.append(media)
            elif isinstance(part.get('fileData'), dict):
                uri = part['fileData'].get('fileUri', '')
                user_chunks.append({'type': 'text', 'text': f'[file reference unavailable: {uri}]'})
        if role == 'model':
            if texts or tool_calls:
                message: dict[str, Any] = {'role': 'assistant', 'content': ''.join(texts) or None}
                if tool_calls:
                    message['tool_calls'] = tool_calls
                messages.append(message)
            messages.extend(tool_messages)
            if trailing_media:
                messages.append({'role': 'user', 'content': trailing_media})
        else:
            messages.extend(tool_messages)
            chunks = user_chunks + trailing_media
            if chunks:
                messages.append({'role': 'user', 'content': _user_content(chunks)})

    request: dict[str, Any] = {'model': model_alias, 'messages': messages, 'stream': stream}
    if stream:
        request['stream_options'] = {'include_usage': True}
    tools = _tools(body)
    if tools:
        request['tools'] = tools
        mode = (((body.get('toolConfig') or {}).get('functionCallingConfig') or {}).get('mode') or '').upper()
        if mode == 'ANY':
            request['tool_choice'] = 'required'
        elif mode == 'NONE':
            request['tool_choice'] = 'none'
    config = body.get('generationConfig') or {}
    if isinstance(config.get('temperature'), (int, float)):
        request['temperature'] = config['temperature']
    if isinstance(config.get('topP'), (int, float)):
        request['top_p'] = config['topP']
    if isinstance(config.get('maxOutputTokens'), int) and config['maxOutputTokens'] > 0:
        request['max_tokens'] = config['maxOutputTokens']
    if config.get('stopSequences'):
        request['stop'] = list(config['stopSequences'])[:4]
    level = ((config.get('thinkingConfig') or {}).get('thinkingLevel') or '').upper()
    if level in EFFORTS:
        request['reasoning_effort'] = EFFORTS[level]
    schema = config.get('responseJsonSchema') or config.get('responseSchema')
    if config.get('responseMimeType') == 'application/json':
        if isinstance(schema, dict):
            request['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'response', 'schema': _lower_schema(schema) if 'responseSchema' in config else schema}}
        else:
            request['response_format'] = {'type': 'json_object'}
    return request


# --------------------------------------------------------------------------
# Response translation (Chat Completions stream -> Gemini stream)
# --------------------------------------------------------------------------

def _valid_signature(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None
    return value


def _signature(call: dict) -> str | None:
    for holder in (call.get('provider_specific_fields'), call.get('function_psf')):
        if isinstance(holder, dict):
            found = _valid_signature(holder.get('thought_signature'))
            if found:
                return found
    call_id = call.get('id') or ''
    if THOUGHT_SEPARATOR in call_id:
        return _valid_signature(call_id.split(THOUGHT_SEPARATOR, 1)[1])
    return None


def _usage(usage: dict | None) -> dict:
    usage = usage or {}
    prompt = int(usage.get('prompt_tokens') or 0)
    completion = int(usage.get('completion_tokens') or 0)
    reasoning = int(((usage.get('completion_tokens_details') or {}).get('reasoning_tokens')) or 0)
    cached = int(((usage.get('prompt_tokens_details') or {}).get('cached_tokens')) or 0)
    total = int(usage.get('total_tokens') or prompt + completion)
    metadata = {'promptTokenCount': prompt,
                'candidatesTokenCount': max(completion - reasoning, 0),
                'totalTokenCount': total}
    if reasoning:
        metadata['thoughtsTokenCount'] = reasoning
    if cached:
        metadata['cachedContentTokenCount'] = cached
    return metadata


def _candidate(parts: list[dict], finish: str | None = None) -> dict:
    candidate: dict[str, Any] = {'content': {'role': 'model', 'parts': parts}, 'index': 0}
    if finish:
        candidate['finishReason'] = finish
    return {'candidates': [candidate]}


def sse_events(lines: Iterator[bytes]) -> Iterator[dict]:
    data: list[str] = []
    for raw in lines:
        line = raw.decode('utf-8', 'replace').rstrip('\r\n')
        if not line:
            if data:
                payload = '\n'.join(data)
                data = []
                if payload.strip() == '[DONE]':
                    return
                yield json.loads(payload)
            continue
        if line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    if data:
        payload = '\n'.join(data)
        if payload.strip() != '[DONE]':
            yield json.loads(payload)


def chat_stream_to_gemini(events: Iterator[dict], emit: Callable[[dict], None]) -> dict:
    """Translate streamed chat chunks; returns a summary for logging."""
    calls: dict[int, dict] = {}
    finish = None
    usage = None
    response_model = None
    for event in events:
        if isinstance(event.get('error'), dict) or 'error' in event and not event.get('choices'):
            error = event.get('error')
            message = error.get('message') if isinstance(error, dict) else str(error)
            raise BridgeError(502, f'upstream stream error: {message}')
        response_model = event.get('model') or response_model
        if event.get('usage'):
            usage = event['usage']
        for choice in event.get('choices') or []:
            delta = choice.get('delta') or choice.get('message') or {}
            reasoning = delta.get('reasoning_content') or delta.get('reasoning')
            if isinstance(reasoning, str) and reasoning:
                emit(_candidate([{'text': reasoning, 'thought': True}]))
            if isinstance(delta.get('content'), str) and delta['content']:
                emit(_candidate([{'text': delta['content']}]))
            for position, call in enumerate(delta.get('tool_calls') or []):
                index = call.get('index', position)
                slot = calls.setdefault(index, {'id': '', 'name': '', 'arguments': ''})
                if call.get('id'):
                    slot['id'] = call['id']
                function = call.get('function') or {}
                if function.get('name'):
                    slot['name'] = function['name']
                if isinstance(function.get('arguments'), str):
                    slot['arguments'] += function['arguments']
                elif isinstance(function.get('arguments'), dict):
                    slot['arguments'] = json.dumps(function['arguments'])
                if isinstance(call.get('provider_specific_fields'), dict):
                    slot['provider_specific_fields'] = call['provider_specific_fields']
                if isinstance(function.get('provider_specific_fields'), dict):
                    slot['function_psf'] = function['provider_specific_fields']
            if choice.get('finish_reason'):
                finish = choice['finish_reason']
    parts = []
    for index in sorted(calls):
        call = calls[index]
        try:
            args = json.loads(call['arguments']) if call['arguments'].strip() else {}
        except json.JSONDecodeError:
            args = {'__unparsed_arguments': call['arguments']}
        if not isinstance(args, dict):
            args = {'value': args}
        part: dict[str, Any] = {'functionCall': {'name': call['name'], 'args': args}}
        if call['id']:
            part['functionCall']['id'] = call['id']
        signature = _signature(call)
        if signature:
            part['thoughtSignature'] = signature
        parts.append(part)
    final = _candidate(parts, FINISH.get(finish or 'stop', 'OTHER'))
    final['usageMetadata'] = _usage(usage)
    if response_model:
        final['modelVersion'] = response_model
    emit(final)
    return {'finish': finish, 'tool_calls': len(parts), 'usage': final['usageMetadata']}


def merge_gemini(chunks: list[dict]) -> dict:
    parts: list[dict] = []
    for chunk in chunks:
        for candidate in chunk.get('candidates') or []:
            for part in (candidate.get('content') or {}).get('parts') or []:
                if parts and 'text' in part and 'text' in parts[-1] and \
                        bool(part.get('thought')) == bool(parts[-1].get('thought')):
                    parts[-1] = {**parts[-1], 'text': parts[-1]['text'] + part['text']}
                else:
                    parts.append(dict(part))
    last = chunks[-1] if chunks else {}
    result = _candidate(parts, ((last.get('candidates') or [{}])[0]).get('finishReason'))
    for key in ('usageMetadata', 'modelVersion'):
        if key in last:
            result[key] = last[key]
    return result


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

class Bridge:
    def __init__(self, upstream_base: str, proxy_token: str, model_alias: str,
                 nonce: str, log_path: str | None = None):
        parsed = urlsplit(upstream_base.rstrip('/'))
        if parsed.scheme != 'http' or not parsed.hostname:
            raise ValueError('upstream must be a plain-HTTP run-local endpoint')
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.path = parsed.path.rstrip('/') + '/chat/completions'
        self.token = proxy_token
        self.model = model_alias
        self.nonce = nonce
        self.log_path = log_path
        self.lock = threading.Lock()
        self.server: http.server.ThreadingHTTPServer | None = None

    def log(self, record: dict) -> None:
        if not self.log_path:
            return
        with self.lock, open(self.log_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')

    def start(self) -> int:
        bridge = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'

            def log_message(self, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                bridge.reject(self, 404, 'only generateContent is bridged')

            def do_POST(self) -> None:
                bridge.handle(self)

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.server.server_address[1]

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()

    @staticmethod
    def reject(handler: http.server.BaseHTTPRequestHandler, status: int, message: str) -> None:
        body = json.dumps({'error': {'code': status, 'message': message,
                                     'status': STATUS.get(status, 'UNKNOWN')}}).encode()
        try:
            handler.send_response(status)
            handler.send_header('Content-Type', 'application/json; charset=UTF-8')
            handler.send_header('Content-Length', str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except OSError:
            pass

    def handle(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        started = time.monotonic()
        path, _, query = handler.path.partition('?')
        record: dict[str, Any] = {'path': path}
        supplied = handler.headers.get('x-goog-api-key') or ''
        if not supplied:
            bearer = handler.headers.get('authorization') or ''
            supplied = bearer[7:] if bearer.lower().startswith('bearer ') else ''
        if not hmac.compare_digest(supplied.encode(), self.nonce.encode()):
            record['status'] = 401
            self.log(record)
            self.reject(handler, 401, 'bridge authentication failed')
            return
        match = PATH_RE.match(path)
        if not match:
            record['status'] = 404
            self.log(record)
            self.reject(handler, 404, f'unsupported Gemini operation: {path}')
            return
        record['requested_model'] = match.group(1)
        streaming = match.group(2) == 'streamGenerateContent'
        try:
            length = int(handler.headers.get('content-length') or 0)
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise BridgeError(400, 'request body is missing or too large')
            body = json.loads(handler.rfile.read(length))
            if not isinstance(body, dict):
                raise BridgeError(400, 'request body must be a JSON object')
            request = gemini_to_chat(body, self.model)
            record['reasoning_effort'] = request.get('reasoning_effort')
            record['tools'] = len(request.get('tools') or [])
            record['messages'] = len(request['messages'])
            summary = self.forward(handler, request, streaming)
            record.update(summary)
            record['status'] = 200
        except BridgeError as error:
            record['status'] = error.status
            record['error'] = str(error)[:500]
            if not getattr(handler, '_frontier_headers_sent', False):
                self.reject(handler, error.status, str(error))
        except (OSError, json.JSONDecodeError, http.client.HTTPException) as error:
            record['status'] = 503
            record['error'] = f'{type(error).__name__}: {error}'[:500]
            if not getattr(handler, '_frontier_headers_sent', False):
                self.reject(handler, 503, 'model bridge upstream failure')
        finally:
            record['seconds'] = round(time.monotonic() - started, 3)
            self.log(record)

    def open_upstream(self, payload: bytes) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse, int]:
        # Same policy as the pi-gemini Candidate: 16 retries, 2 s base, 60 s cap.
        # Only before any byte reaches agy, so an answer is never replayed.
        retries = 0
        while True:
            connection = http.client.HTTPConnection(self.host, self.port, timeout=UPSTREAM_TIMEOUT_S)
            try:
                connection.request('POST', self.path, body=payload, headers={
                    'Authorization': f'Bearer {self.token}',
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'})
                response = connection.getresponse()
                if response.status == 200:
                    return connection, response, retries
                detail = response.read(65536).decode('utf-8', 'replace')
                try:
                    parsed = json.loads(detail)
                    message = (parsed.get('error') or {}).get('message') or detail
                except (json.JSONDecodeError, AttributeError):
                    message = detail
                error: Exception = BridgeError(response.status, f'upstream HTTP {response.status}: {message}'[:4000])
                retryable = response.status in RETRY_STATUS or response.status >= 500
            except (OSError, http.client.HTTPException) as caught:
                error, retryable = caught, True
            connection.close()
            if not retryable or retries >= RETRY_MAX:
                raise error
            time.sleep(min(RETRY_BASE_S * (2 ** retries), RETRY_CAP_S))
            retries += 1

    def forward(self, handler: http.server.BaseHTTPRequestHandler, request: dict,
                streaming: bool) -> dict:
        payload = json.dumps(request, ensure_ascii=False).encode()
        connection, response, retries = self.open_upstream(payload)
        try:
            events = sse_events(iter(response.readline, b''))
            if streaming:
                handler.send_response(200)
                handler.send_header('Content-Type', 'text/event-stream')
                handler.send_header('Cache-Control', 'no-cache')
                handler.end_headers()
                handler._frontier_headers_sent = True  # type: ignore[attr-defined]

                def emit(chunk: dict) -> None:
                    handler.wfile.write(b'data: ' + json.dumps(chunk, ensure_ascii=False).encode() + b'\r\n\r\n')
                    handler.wfile.flush()

                try:
                    return chat_stream_to_gemini(events, emit)
                except BridgeError as error:
                    emit({'error': {'code': error.status, 'message': str(error),
                                    'status': STATUS.get(error.status, 'UNKNOWN')}})
                    raise
            chunks: list[dict] = []
            summary = chat_stream_to_gemini(events, chunks.append)
            body = json.dumps(merge_gemini(chunks), ensure_ascii=False).encode()
            handler.send_response(200)
            handler.send_header('Content-Type', 'application/json; charset=UTF-8')
            handler.send_header('Content-Length', str(len(body)))
            handler.end_headers()
            handler._frontier_headers_sent = True  # type: ignore[attr-defined]
            handler.wfile.write(body)
            return summary
        finally:
            connection.close()


if __name__ == '__main__':  # pragma: no cover - manual debugging aid
    sys.exit('import this module from the launcher')
