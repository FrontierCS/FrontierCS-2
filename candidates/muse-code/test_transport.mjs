import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import test from 'node:test';
import { createTransport } from './transport.mjs';

const catalog = { object: 'list', data: [{ id: 'fixture-only', object: 'model',
  metadata: { 'muse-code': { release_date: '2026-01-01', is_hidden: false,
    limit: { context: 1000, output: 100 } } } }] };

async function fixture(t, handler) {
  const driver = http.createServer(handler);
  driver.listen(0, '127.0.0.1');
  await once(driver, 'listening');
  const transport = createTransport({
    driverBaseUrl: `http://127.0.0.1:${driver.address().port}/v1`,
    agentToken: 'fixture-agent-capability', catalog,
  });
  const base = await transport.listen();
  t.after(async () => {
    await transport.close();
    driver.closeAllConnections();
    await new Promise(resolve => driver.close(resolve));
  });
  return base;
}

test('catalog is local and only the two exact paths are available', async t => {
  let calls = 0;
  const base = await fixture(t, () => { calls++; });
  assert.deepEqual(await (await fetch(base + '/muse-code/models')).json(), catalog);
  for (const path of ['/responses?model=other', '/v1/responses', '/models', '/responses/']) {
    assert.equal((await fetch(base + path, { method: 'POST', body: '{}' })).status, 404);
  }
  assert.equal(calls, 0);
});

test('raw request, SSE reasoning/tools and usage survive; all authority is replaced', async t => {
  const payload = '{ "model":"fixture-only", "input":[{"type":"function_call_output","output":"x"}],"stream":true }';
  const sse = 'data: {"type":"response.reasoning_text.delta","delta":"x"}\n\n' +
    'data: {"type":"response.function_call_arguments.done","arguments":"{}"}\n\n' +
    'data: {"type":"response.completed","response":{"usage":{"input_tokens":2,"output_tokens":3,"output_tokens_details":{"reasoning_tokens":1}}}}\n\n';
  const base = await fixture(t, async (req, res) => {
    assert.equal(req.url, '/v1/responses');
    assert.equal(req.headers['x-frontier-agent-token'], 'fixture-agent-capability');
    for (const key of ['authorization', 'cookie', 'x-api-key', 'x-litellm-api-key', 'x-frontier-upstream']) {
      assert.equal(req.headers[key], undefined);
    }
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    assert.equal(Buffer.concat(chunks).toString(), payload);
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end(sse);
  });
  const response = await fetch(base + '/responses', { method: 'POST', body: payload,
    headers: { 'content-type': 'application/json', authorization: 'Bearer dummy',
      cookie: 'secret=no', 'x-api-key': 'no', 'x-litellm-api-key': 'no',
      'x-frontier-agent-token': 'wrong', 'x-frontier-upstream': 'no' } });
  assert.equal(response.headers.get('content-type'), 'text/event-stream');
  assert.equal(await response.text(), sse);
});

test('Driver refusal is preserved without retry or synthetic completion', async t => {
  let calls = 0;
  const base = await fixture(t, (req, res) => {
    calls++;
    res.writeHead(429, { 'content-type': 'application/json', 'retry-after': '7' });
    res.end('{"error":"model_session_quota_exhausted"}');
  });
  const response = await fetch(base + '/responses', { method: 'POST', body: '{}' });
  assert.equal(response.status, 429);
  assert.equal(response.headers.get('retry-after'), '7');
  assert.equal(await response.text(), '{"error":"model_session_quota_exhausted"}');
  assert.equal(calls, 1);
});

test('first event arrives before completion and client cancel closes Driver stream', async t => {
  let finishDriver;
  const closed = new Promise(resolve => { finishDriver = resolve; });
  const base = await fixture(t, (req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write('data: {"type":"response.created"}\n\n');
    res.once('close', finishDriver);
  });
  const controller = new AbortController();
  const response = await fetch(base + '/responses', {
    method: 'POST', body: '{}', signal: controller.signal,
  });
  const reader = response.body.getReader();
  const first = await reader.read();
  assert.equal(new TextDecoder().decode(first.value), 'data: {"type":"response.created"}\n\n');
  controller.abort();
  await closed;
});

test('configuration rejects provider URLs and ambiguous catalogs', () => {
  for (const driverBaseUrl of ['https://api.meta.ai/v1', 'http://localhost:8082/v1',
    'http://127.0.0.1:8082/v1?key=x', 'http://u:p@127.0.0.1:8082/v1']) {
    assert.throws(() => createTransport({ driverBaseUrl, agentToken: 'x', catalog }));
  }
  assert.throws(() => createTransport({ driverBaseUrl: 'http://127.0.0.1:8082/v1',
    agentToken: 'x', catalog: { object: 'list', data: [] } }));
});
