// Run in an unprivileged, network-disabled container with a read-only root.
import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { readFile, mkdtemp } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const transient = process.env.INJECT_TRUNCATED_STREAM === '1';
const invalid = process.env.INJECT_INVALID_REQUEST === '1';
let injected = false;
let failedMessages;
const broken = process.env.EXPECT_BROKEN === '1';
const capability = 'offline-driver-capability';
const toolResults = [];
const submissions = [];
const payloads = new Map();
let requestCount = 0;
const calls = [
  ['bash', { command: 'printf fixture-ok > /workspace/solution.txt && tar -cf /workspace/final.tar -C /workspace solution.txt',
    description: 'Create the local selected answer archive' }],
  ['glob', { pattern: '*.txt', path: '/workspace' }],
  ['grep', { pattern: 'fixture-ok', path: '/workspace/solution.txt' }],
  ['bash', { command: 'frontier-submit train /workspace/final.tar && frontier-submit store-final /workspace/final.tar && frontier-submit final /workspace/final.tar',
    description: 'Verify save and submit the selected answer' }],
];
const frame = value => `data: ${JSON.stringify(value)}\n\n`;
const model = http.createServer(async (req, res) => {
  assert.equal(req.url, '/v1/chat/completions');
  assert.equal(req.headers.authorization, `Bearer ${capability}`);
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = JSON.parse(Buffer.concat(chunks));
  assert.equal(body.model, 'fixture-only');
  assert.equal(body.stream, true);
  const results = body.messages.filter(message => message.role === 'tool');
  toolResults.splice(0, toolResults.length, ...results);
  requestCount++;
  if ((transient || invalid) && !injected) {
    injected = true;
    failedMessages = body.messages;
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write(frame({ id: 'incomplete', object: 'chat.completion.chunk', model: 'fixture-only',
      choices: [{ index: 0, delta: { role: 'assistant', content: 'INCOMPLETE_MUST_NOT_BE_REPLAYED' }, finish_reason: null }] }));
    res.end(frame({ error: { code: invalid ? '400' : '500', message: invalid ? '400 invalid_request: unsupported parameter' :
      "litellm.APIConnectionError: APIConnectionError: OpenAIException - Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'>" } }));
    return;
  }
  if (transient && requestCount === 2) assert.deepEqual(body.messages, failedMessages, 'Partial failed response was replayed');
  const index = requestCount - 1 - Number(transient);
  assert.ok(requestCount <= 5 + Number(transient), 'Unexpected extra model continuation');
  const call = calls[index];
  if (call) assert.ok(body.tools.some(tool => tool.function.name === call[0]));
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  const chunk = (delta, finish_reason = null) => ({ id: `chatcmpl_fixture_${index}`,
    object: 'chat.completion.chunk', created: 1, model: 'fixture-only',
    choices: [{ index: 0, delta, finish_reason }] });
  res.write(frame(chunk({ role: 'assistant', reasoning_content: 'Offline fixture.' })));
  if (call) {
    res.write(frame(chunk({ tool_calls: [{ index: 0, id: `call_fixture_${index}`,
      type: 'function', function: { name: call[0], arguments: JSON.stringify(call[1]) } }] })));
    res.write(frame(chunk({}, 'tool_calls')));
  } else {
    res.write(frame(chunk({ content: 'Fixture complete.' })));
    res.write(frame(chunk({}, 'stop')));
  }
  res.write(frame({ ...chunk({}), choices: [], usage: { prompt_tokens: 5,
    completion_tokens: 3, total_tokens: 8, completion_tokens_details: { reasoning_tokens: 1 } } }));
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
  child = spawn('/opt/frontier/candidate/launch', [], { cwd: '/workspace',
    env: { PATH: '/usr/local/bin:/usr/bin:/bin', LANG: 'C.UTF-8',
      FRONTIER_DRIVER_API_URL: 'http://127.0.0.1:8080',
      FRONTIER_DRIVER_TOKEN_FILE: '/var/run/frontier/token',
      FRONTIER_MODEL_API_BASE_URL: 'http://127.0.0.1:8082/v1',
      FRONTIER_MODEL_PROXY_TOKEN: capability, FRONTIER_CODEX_MODEL: 'fixture-only',
      FRONTIER_MODEL_CONTEXT_TOKENS: '1048576', FRONTIER_MODEL_OUTPUT_TOKENS: '262144',
      FRONTIER_MODEL_REASONING_EFFORT: 'max', TMPDIR: '/tmp', TMP: '/tmp', TEMP: '/tmp',
      // The negative control isolates subprocess temp writes from native-cache startup.
      NARB_DISABLE_NATIVE_CACHE: broken ? '1' : '0' },
    stdio: ['ignore', 'pipe', 'pipe'] });
  const stdout = [];
  const stderr = [];
  child.stdout.on('data', data => stdout.push(data));
  child.stderr.on('data', data => stderr.push(data));
  const watchdog = setTimeout(() => child.kill('SIGTERM'), 60000);
  const [code] = await once(child, 'close');
  clearTimeout(watchdog);
  const output = Buffer.concat([...stderr, ...stdout]).toString();
  if (invalid) {
    assert.equal(code, 1, output);
    assert.equal(requestCount, 1, 'Invalid request was retried');
    assert.deepEqual(submissions, []);
    console.log('INVALID_REQUEST_NOT_RETRIED');
    process.exitCode = 0;
  } else {
  assert.equal(code, broken ? 2 : 0, output);
  assert.equal(toolResults.length, 4, output);
  if (broken) {
    assert.ok(output.includes('EROFS'), output);
    assert.deepEqual(submissions, []);
  } else {
    assert.ok(!output.includes('EROFS'), output);
    assert.ok(toolResults[1].content.includes('solution.txt'));
    assert.ok(toolResults[2].content.includes('fixture-ok'));
    assert.deepEqual(submissions, ['/v1/verifier-train', '/v1/submissions/checkpoint', '/v1/submissions']);
    const receipt = JSON.parse(await readFile('/workspace/.frontier/final-submission.json', 'utf8'));
    assert.equal(receipt.state, 'accepted');
    assert.equal(receipt.operation, 'final');
    assert.equal(receipt.verification_id, 'verification_final_fixture');
    assert.equal(await readFile('/workspace/solution.txt', 'utf8'), 'fixture-ok');
  }
  console.log(JSON.stringify({ status: broken ? 'READONLY_FAILURE_REPRODUCED' : 'READONLY_FIXTURE_PASSED',
    uid: process.getuid(), model_requests: requestCount, tool_results: toolResults.length,
    submissions, real_provider_requests: 0, independent_judge_executed: false }));
  }
} finally {
  if (child?.exitCode === null) child.kill('SIGKILL');
  for (const server of [model, driver]) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}
