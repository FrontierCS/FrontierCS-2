import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { readFile, readdir, mkdtemp } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const capability = 'offline-driver-capability';
const modelAlias = 'inkling-2-small';
const requests = [];
const headers = [];
const submissions = [];
const payloads = new Map();
const calls = [
  ['bash', { command: 'printf fixture-ok > /workspace/solution.txt && tar -cf /workspace/final.tar -C /workspace solution.txt',
    description: 'Write and pack the answer' }],
  ['read', { filePath: '/frontier/task/README.md' }],
  ['grep', { pattern: 'fixture-ok', path: '/workspace', include: '*.txt' }],
  ['skill', { name: 'frontier-submit' }],
  ['bash', { command: 'frontier-submit train /workspace/final.tar && frontier-submit store-final /workspace/final.tar && frontier-submit final /workspace/final.tar',
    description: 'Submit train, save and final' }],
];
const callId = index => `call_fixture_${index}`;
const thought = index => `Fixture thought ${index}.`;
const frame = value => `data: ${JSON.stringify(value)}\n\n`;
const model = http.createServer(async (req, res) => {
  try {
    assert.equal(req.method, 'POST');
    assert.equal(req.url, '/v1/chat/completions');
    assert.equal(req.headers.authorization, `Bearer ${capability}`);
    headers.push(Object.keys(req.headers).sort());
    for (const name of Object.keys(req.headers)) {
      assert.ok(!/^(?:x-api-key|api-key|cookie|openai-organization|openai-project|forwarded|via|session_id)$/.test(name), name);
      assert.ok(!/^(?:x-forwarded-|x-litellm-|x-proxy-|proxy-)/.test(name), name);
    }
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks));
    const index = requests.length;
    requests.push(body);
    assert.ok(index <= calls.length, 'Unexpected continuation call');
    assert.equal(body.model, modelAlias);
    assert.equal(body.stream, true);
    assert.deepEqual(body.stream_options, { include_usage: true });
    // OpenCode's per-response cap is raised to the model's full output limit;
    // Tinker clamps it to the remaining window.
    assert.equal(body.max_tokens, 262144);
    assert.ok(!('max_completion_tokens' in body) && !('store' in body), JSON.stringify(Object.keys(body)));
    assert.equal(body.messages[0].role, 'system');
    assert.ok(!body.messages.some(m => m.role === 'developer'));
    const user = body.messages.find(m => m.role === 'user');
    const userText = typeof user.content === 'string' ? user.content : user.content.map(p => p.text).join('');
    assert.ok(userText.startsWith('Solve the task in /frontier/task/README.md.'), userText.slice(0, 80));
    assert.ok(userText.includes('frontier-submit final exactly once'));
    const system = body.messages.filter(m => m.role === 'system').map(m => m.content).join('\n');
    assert.match(system, /frontier-submit/, 'skill catalog must list the Frontier skills');
    for (let previous = 0; previous < index; previous += 1) {
      const replay = body.messages.find(m => m.role === 'assistant'
        && m.tool_calls?.some(call => call.id === callId(previous)));
      assert.ok(replay, `tool call ${previous} must be replayed`);
      assert.equal(replay.reasoning_content, thought(previous), 'reasoning is replayed as reasoning_content');
      assert.ok(body.messages.some(m => m.role === 'tool' && m.tool_call_id === callId(previous)));
    }
    const toolNames = (body.tools ?? []).map(tool => tool.function.name).sort();
    for (const name of ['bash', 'edit', 'glob', 'grep', 'read', 'skill', 'write']) assert.ok(toolNames.includes(name), JSON.stringify(toolNames));
    for (const name of ['webfetch', 'websearch', 'question']) assert.ok(!toolNames.includes(name), JSON.stringify(toolNames));
    const call = calls[index];
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    const chunk = (delta, finish_reason = null) => ({ id: `chatcmpl_fixture_${index}`,
      object: 'chat.completion.chunk', created: 1, model: modelAlias,
      choices: [{ index: 0, delta, finish_reason }] });
    res.write(frame(chunk({ role: 'assistant', reasoning_content: thought(index) })));
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
      completion_tokens_details: { reasoning_tokens: 20 } } }));
    res.end('data: [DONE]\n\n');
  } catch (error) {
    console.error('model fixture:', error);
    res.writeHead(400, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: `fixture rejected request: ${error.message}`, type: 'invalid_request_error' } }));
  }
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
    env: { PATH: '/usr/local/bin:/usr/bin:/bin', LANG: 'C.UTF-8', PWD: '/',
      FRONTIER_DRIVER_API_URL: 'http://127.0.0.1:8080',
      FRONTIER_DRIVER_TOKEN_FILE: '/var/run/frontier/token',
      FRONTIER_MODEL_API_BASE_URL: 'http://127.0.0.1:8082/v1',
      FRONTIER_MODEL_PROXY_TOKEN: capability, FRONTIER_CODEX_MODEL: modelAlias,
      OPENAI_API_KEY: 'must-not-leak', TINKER_API_KEY: 'must-not-leak',
      OPENCODE_CONFIG_CONTENT: '{"model":"foreign/model"}', OPENCODE_DISABLE_AUTOCOMPACT: '1' },
    stdio: ['pipe', 'pipe', 'pipe'] });
  child.stdin.end();
  const stdout = [];
  const stderr = [];
  child.stdout.on('data', data => stdout.push(data));
  child.stderr.on('data', data => stderr.push(data));
  const watchdog = setTimeout(() => child.kill('SIGTERM'), 180000);
  const [code] = await once(child, 'close');
  clearTimeout(watchdog);
  const text = Buffer.concat(stdout).toString();
  const diagnostics = Buffer.concat(stderr).toString();
  assert.equal(code, 0, text + diagnostics);
  assert.equal(requests.length, calls.length + 1, text + diagnostics);
  const events = text.split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line));
  assert.ok(!events.some(event => event.type === 'error'), text);
  const toolResults = events.filter(event => event.type === 'tool_use').map(event => event.part);
  assert.deepEqual(toolResults.map(part => part.tool), calls.map(call => call[0]), text);
  assert.ok(toolResults.every(part => part.state.status === 'completed'), JSON.stringify(toolResults));
  assert.match(toolResults[1].state.output, /Offline fixture/);
  assert.match(toolResults[2].state.output, /solution\.txt/);
  assert.match(toolResults[3].state.output, /Frontier submission/);
  const steps = events.filter(event => event.type === 'step_finish').map(event => event.part);
  assert.equal(steps.length, calls.length + 1);
  for (const step of steps) {
    assert.equal(step.tokens.output, 24 - 20);
    assert.equal(step.tokens.reasoning, 20);
  }
  assert.ok(events.some(event => event.type === 'text' && event.part.text === 'Fixture complete.'));
  assert.deepEqual(submissions, ['/v1/verifier-train', '/v1/submissions/checkpoint', '/v1/submissions']);
  assert.equal(await readFile('/workspace/solution.txt', 'utf8'), 'fixture-ok');
  const receipt = JSON.parse(await readFile('/workspace/.frontier/final-submission.json', 'utf8'));
  assert.equal(receipt.operation, 'final');
  assert.equal(receipt.state, 'accepted');
  const home = '/workspace/.frontier-opencode-home';
  const config = await readFile(`${home}/opencode.json`, 'utf8');
  assert.ok(!config.includes(capability) && !config.includes('must-not-leak'));
  // No npm, models.dev, ripgrep or LSP downloads were attempted.
  const logDir = `${home}/data/opencode/log`;
  const logs = (await Promise.all((await readdir(logDir)).map(name => readFile(`${logDir}/${name}`, 'utf8')))).join('\n');
  for (const marker of ['dependency install failed', 'downloading ripgrep', 'Failed to fetch models.dev',
    'registry.npmjs.org', 'models.dev/api.json', capability, 'must-not-leak']) {
    assert.ok(!logs.includes(marker), marker);
  }
  assert.deepEqual(await readdir(`${home}/config/opencode/node_modules`), []);
  console.log(JSON.stringify({ status: 'PASS_OFFICIAL_OPENCODE_READONLY_WIRE_FIXTURE',
    model_requests: requests.length, max_tokens: requests[0].max_tokens,
    tools: requests[0].tools.map(tool => tool.function.name).sort(),
    reasoning_replayed_as: 'reasoning_content', request_headers: headers[0],
    request_body_keys: Object.keys(requests[0]).sort(), submissions,
    tool_calls: toolResults.map(part => part.tool), wall_ms: Date.now() - started,
    actual_provider_requests: 0, independent_judge_executed: false }));
} finally {
  if (child?.exitCode === null) child.kill('SIGKILL');
  for (const server of [model, driver]) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}
