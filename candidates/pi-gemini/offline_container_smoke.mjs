import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { readFile, mkdtemp } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const capability = 'offline-driver-capability';
const modelAlias = 'gemini-4-argon';
// LiteLLM's Chat Completions translation for Gemini carries thought
// signatures inside tool call ids; Pi must replay them byte-for-byte.
const signature = '__thought__' + Buffer.from('fixture-signature+/=').toString('base64');
const requests = [];
const headers = [];
const submissions = [];
const payloads = new Map();
const calls = [
  ['bash', { command: 'printf fixture-ok > /workspace/solution.txt && tar -cf /workspace/final.tar -C /workspace solution.txt' }],
  ['read', { path: '/workspace/solution.txt' }],
  ['bash', { command: 'frontier-submit train /workspace/final.tar && frontier-submit store-final /workspace/final.tar && frontier-submit final /workspace/final.tar' }],
];
const callId = index => `call_fixture_${index}${signature}`;
const frame = value => `data: ${JSON.stringify(value)}\n\n`;
const model = http.createServer(async (req, res) => {
  assert.equal(req.url, '/v1/chat/completions');
  assert.equal(req.headers.authorization, `Bearer ${capability}`);
  headers.push(Object.keys(req.headers).sort());
  for (const name of Object.keys(req.headers)) {
    assert.ok(!/^(?:x-goog-api-key|x-api-key|api-key|cookie|openai-organization|openai-project|forwarded|via|x-session-affinity|session_id)$/.test(name), name);
    assert.ok(!/^(?:x-forwarded-|x-litellm-|x-proxy-|proxy-)/.test(name), name);
  }
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = JSON.parse(Buffer.concat(chunks));
  assert.equal(body.model, modelAlias);
  assert.equal(body.stream, true);
  assert.equal(body.reasoning_effort, 'high');
  // Pi reserves the estimated input within the context window; the gateway
  // removes this client-side capacity hint before forwarding upstream.
  assert.ok(body.max_completion_tokens > 900000 && body.max_completion_tokens <= 1048576);
  assert.ok(!('store' in body), 'Gateway request must not ask for provider storage');
  assert.ok(!('thinking_budget' in body) && !('max_tokens' in body), 'No extra numeric caps');
  assert.deepEqual(body.stream_options, { include_usage: true });
  assert.ok(!body.messages.some(m => m.role === 'developer'), 'Gemini route uses system role');
  const index = requests.length;
  requests.push(body);
  assert.ok(index < 4, 'Unexpected continuation call');
  for (let previous = 0; previous < index; previous += 1) {
    const replay = body.messages.find(m => m.role === 'assistant'
      && m.tool_calls?.some(call => call.id === callId(previous)));
    assert.ok(replay, `tool call ${previous} id with thought signature must be replayed verbatim`);
    assert.ok(body.messages.some(m => m.role === 'tool' && m.tool_call_id === callId(previous)));
  }
  const call = calls[index];
  const toolNames = (body.tools ?? []).map(tool => tool.function.name).sort();
  assert.deepEqual(toolNames, ['bash', 'edit', 'read', 'write'], JSON.stringify(toolNames));
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  const chunk = (delta, finish_reason = null) => ({ id: `chatcmpl_fixture_${index}`,
    object: 'chat.completion.chunk', created: 1, model: modelAlias,
    choices: [{ index: 0, delta, finish_reason }] });
  res.write(frame(chunk({ role: 'assistant', reasoning_content: `Fixture thought ${index}.` })));
  if (call) {
    res.write(frame(chunk({ tool_calls: [{ index: 0, id: callId(index),
      type: 'function', function: { name: call[0], arguments: JSON.stringify(call[1]) } }] })));
    res.write(frame(chunk({}, 'tool_calls')));
  } else {
    res.write(frame(chunk({ content: 'Fixture complete.' })));
    res.write(frame(chunk({}, 'stop')));
  }
  res.write(frame({ ...chunk({}), choices: [], usage: { prompt_tokens: 75,
    completion_tokens: 24, total_tokens: 99,
    completion_tokens_details: { reasoning_tokens: 20, text_tokens: 4 } } }));
  res.end('data: [DONE]\n\n');
});
const driver = http.createServer(async (req, res) => {
  assert.equal(req.headers.authorization, `Bearer ${capability}`);
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const raw = Buffer.concat(chunks);
  let response;
  if (req.url === '/v1/payloads') {
    const digest = 'sha256:' + createHash('sha256').update(raw).digest('hex');
    assert.equal(req.headers['x-content-sha256'], digest);
    const id = 'payload_' + digest.slice(7, 27);
    payloads.set(id, digest);
    response = { payload_id: id };
  } else {
    const body = JSON.parse(raw);
    assert.ok(payloads.has(body.payload_id));
    submissions.push(req.url);
    if (req.url === '/v1/verifier-train') response = { verification_id: 'verification_train_fixture',
      state: 'running', remaining_train_calls: 1023 };
    else if (req.url === '/v1/submissions/checkpoint') response = {
      remaining_runtime_s: 17900, remaining_checkpoint_submissions: null };
    else if (req.url === '/v1/submissions') response = { verification_id: 'verification_final_fixture' };
    else throw new Error('Unexpected fixture Driver route');
  }
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify(response));
});
let child;
try {
  assert.notEqual(process.getuid(), 0);
  await assert.rejects(mkdtemp('/tmp/root-write-probe-'), { code: 'EROFS' });
  for (const [server, port] of [[model, 8082], [driver, 8080]]) {
    server.listen(port, '127.0.0.1');
    await once(server, 'listening');
  }
  const started = Date.now();
  child = spawn('python3', ['/opt/frontier/candidate/launch'], { cwd: '/workspace',
    env: { PATH: '/usr/local/bin:/usr/bin:/bin', LANG: 'C.UTF-8',
      FRONTIER_DRIVER_API_URL: 'http://127.0.0.1:8080',
      FRONTIER_DRIVER_TOKEN_FILE: '/var/run/frontier/token',
      FRONTIER_MODEL_API_BASE_URL: 'http://127.0.0.1:8082/v1',
      FRONTIER_MODEL_PROXY_TOKEN: capability, FRONTIER_CODEX_MODEL: modelAlias,
      GEMINI_API_KEY: 'must-not-leak', GOOGLE_GEMINI_BASE_URL: 'https://foreign.example',
      PI_OFFLINE: '0' },
    stdio: ['pipe', 'pipe', 'pipe'] });
  const stdout = [];
  const stderr = [];
  child.stdout.on('data', data => stdout.push(data));
  child.stderr.on('data', data => stderr.push(data));
  const watchdog = setTimeout(() => child.kill('SIGTERM'), 120000);
  const [code] = await once(child, 'close');
  clearTimeout(watchdog);
  const text = Buffer.concat(stdout).toString();
  const diagnostics = Buffer.concat(stderr).toString();
  assert.equal(code, 0, text + diagnostics);
  assert.equal(requests.length, 4, text + diagnostics);
  const events = text.split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line));
  assert.equal(events[0].type, 'session');
  assert.ok(events.some(event => event.type === 'agent_settled' && event.aborted === false));
  const assistant = events.filter(event => event.type === 'message_end'
    && event.message.role === 'assistant').map(event => event.message);
  assert.equal(assistant.length, 4);
  const usage = assistant.map(message => message.usage);
  for (const item of usage) {
    assert.equal(item.input, 75);
    assert.equal(item.output, 24);
    assert.equal(item.totalTokens, 99);
  }
  const toolResults = events.filter(event => event.type === 'tool_execution_end');
  assert.equal(toolResults.length, 3);
  assert.ok(toolResults.every(event => event.isError === false), JSON.stringify(toolResults));
  assert.match(JSON.stringify(toolResults[1].result), /fixture-ok/);
  assert.deepEqual(submissions, ['/v1/verifier-train', '/v1/submissions/checkpoint', '/v1/submissions']);
  assert.equal(await readFile('/workspace/solution.txt', 'utf8'), 'fixture-ok');
  const receipt = JSON.parse(await readFile('/workspace/.frontier/final-submission.json', 'utf8'));
  assert.equal(receipt.operation, 'final');
  assert.equal(receipt.state, 'accepted');
  for (const name of ['models.json', 'settings.json']) {
    const config = await readFile(`/workspace/.frontier-pi-home/agent/${name}`, 'utf8');
    assert.ok(!config.includes(capability) && !config.includes('must-not-leak'), name);
  }
  console.log(JSON.stringify({ status: 'PASS_OFFICIAL_PI_READONLY_WIRE_FIXTURE',
    model_requests: requests.length, reasoning_effort: requests[0].reasoning_effort,
    max_completion_tokens: requests[0].max_completion_tokens,
    thought_signature_ids_replayed: 3, request_headers: headers[0], submissions,
    native_cli_usage_per_response: usage[0], wall_ms: Date.now() - started,
    actual_provider_requests: 0, independent_judge_executed: false }));
} finally {
  if (child?.exitCode === null) child.kill('SIGKILL');
  for (const server of [model, driver]) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}
