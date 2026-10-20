import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const capability = 'offline-driver-capability';
const thinking = process.env.FIXTURE_QWEN_THINKING === '1';
const modelAlias = thinking ? 'qwen3.8-max-vl-thinking' : 'qwen3.8-max';
const nearLimit = process.env.FIXTURE_NEAR_CONTEXT_LIMIT === '1';
const clampControl = process.env.FIXTURE_CONTEXT_CLAMP_CONTROL === '1';
const lifetime = process.env.FIXTURE_STREAM_LIFETIME === '1';
const lifetimeControl = process.env.FIXTURE_STREAM_LIFETIME_CONTROL === '1';
assert.ok(!clampControl || nearLimit, 'Clamp control requires near-context mode');
assert.ok(!lifetimeControl || lifetime, 'Lifetime control requires stream mode');
assert.ok(!lifetime || !nearLimit, 'Stream and context fixtures are separate');
const wire = thinking ? { temperature: 1, max_tokens: null,
  reasoning_effort: 'high', enable_thinking: true } :
  { temperature: 1, max_tokens: 131072, thinking_budget: 131072,
    preserve_thinking: true, enable_thinking: true };
const requests = [];
const submissions = [];
const payloads = new Map();
const calls = [
  ['run_shell_command', { command: 'printf fixture-ok > /workspace/solution.txt && tar -cf /workspace/final.tar -C /workspace solution.txt',
    description: 'Create the local selected answer archive' }],
  ['read_file', { file_path: '/workspace/solution.txt' }],
  ['run_shell_command', { command: 'frontier-submit train /workspace/final.tar && frontier-submit store-final /workspace/final.tar && frontier-submit final /workspace/final.tar',
    description: 'Verify, save, and submit the selected answer' }],
];
const frame = value => `data: ${JSON.stringify(value)}\n\n`;
const model = http.createServer(async (req, res) => {
  assert.equal(req.url, '/v1/chat/completions');
  assert.equal(req.headers.authorization, `Bearer ${capability}`);
  for (const name of Object.keys(req.headers)) {
    assert.ok(!/^(?:x-request-id|x-correlation-id|traceparent|tracestate|baggage|sentry-trace|x-amzn-trace-id|cookie|x-api-key|api-key|openai-organization|openai-project|forwarded|via)$/.test(name));
    assert.ok(!/^(?:x-forwarded-|x-litellm-|x-proxy-|proxy-)/.test(name));
  }
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = JSON.parse(Buffer.concat(chunks));
  assert.equal(body.model, modelAlias);
  if (thinking) {
    for (const key of ['max_completion_tokens', 'max_output_tokens', 'thinking_budget']) {
      assert.ok(!(key in body), `Unexpected numeric cap ${key}`);
    }
  }
  assert.equal(body.stream, true);
  for (const [key, value] of Object.entries(wire)) {
    if (clampControl && key === 'max_tokens') assert.ok(body[key] > 0 && body[key] <= value, `native context control cap: ${body[key]}`);
    else assert.equal(body[key], value, key);
  }
  assert.ok(!('extra_body' in body), 'Extra parameters must be top-level wire keys');
  const index = requests.length;
  requests.push(body);
  assert.ok(index < (lifetimeControl ? 10 : 4), 'Unexpected continuation or background memory call');
  const call = calls[lifetimeControl ? 0 : index];
  if (call) assert.ok(body.tools.some(tool => tool.function.name === call[0]), JSON.stringify(body.tools.map(t => t.function.name)));
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  const chunk = (delta, finish_reason = null) => ({ id: `chatcmpl_fixture_${index}`,
    object: 'chat.completion.chunk', created: 1, model: modelAlias,
    choices: [{ index: 0, delta, finish_reason }] });
  res.write(frame(chunk({ role: 'assistant', reasoning_content: `Fixture thought ${index}.` })));
  if (lifetime) await new Promise(resolve => setTimeout(resolve, 100));
  if (call) {
    res.write(frame(chunk({ tool_calls: [{ index: 0, id: `call_fixture_${index}`,
      type: 'function', function: { name: call[0], arguments: JSON.stringify(call[1]) } }] })));
    res.write(frame(chunk({}, 'tool_calls')));
  } else {
    res.write(frame(chunk({ content: 'Fixture complete.' })));
    res.write(frame(chunk({}, 'stop')));
  }
  res.write(frame({ ...chunk({}), choices: [], usage: { prompt_tokens: 75,
    completion_tokens: 24, total_tokens: 99,
    completion_tokens_details: { reasoning_tokens: 20, text_tokens: 24 } } }));
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
  let launcher = '/opt/frontier/candidate/launch';
  if (nearLimit) {
    launcher = '/workspace/near-limit-launch.py';
    await writeFile(launcher, `#!/usr/bin/env python3
from importlib.machinery import SourceFileLoader
import importlib.util
loader = SourceFileLoader('fixture_qwen', '/opt/frontier/candidate/launch')
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
original = module.settings
def settings(*args):
    result = original(*args)
    result['modelProviders']['openai'][0]['generationConfig']['contextWindowSize'] = 150000
    ${clampControl ? "result['modelProviders']['openai'][0]['generationConfig']['extra_body'].pop('max_tokens')" : ''}
    return result
module.settings = settings
module.PROMPT += '\\nIgnore this padding: ' + 'padding ' * 15000
raise SystemExit(module.main())
`, { mode: 0o700 });
  }
  if (lifetime) {
    launcher = '/workspace/stream-lifetime-launch.py';
    await writeFile(launcher, `#!/usr/bin/env python3
from importlib.machinery import SourceFileLoader
import importlib.util
loader = SourceFileLoader('fixture_qwen', '/opt/frontier/candidate/launch')
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
original = module.settings
def settings(*args):
    result = original(*args)
    result['modelProviders']['openai'][0]['generationConfig']['maxRetries'] = 0
    return result
module.settings = settings
${lifetimeControl ? `original_env = module.prepare_environment
def environment():
    child = original_env()
    child['QWEN_STREAM_MAX_LIFETIME_MS'] = '1'
    return child
module.prepare_environment = environment` : ''}
raise SystemExit(module.main())
`, { mode: 0o700 });
  }
  child = spawn('python3', [launcher], { cwd: '/workspace',
    env: { PATH: '/usr/local/bin:/usr/bin:/bin', LANG: 'C.UTF-8',
      FRONTIER_DRIVER_API_URL: 'http://127.0.0.1:8080',
      FRONTIER_DRIVER_TOKEN_FILE: '/var/run/frontier/token',
      FRONTIER_MODEL_API_BASE_URL: 'http://127.0.0.1:8082/v1',
      FRONTIER_MODEL_PROXY_TOKEN: capability, FRONTIER_CODEX_MODEL: modelAlias,
      ...(lifetime ? { QWEN_STREAM_MAX_LIFETIME_MS: '1' } : {}) },
    stdio: ['ignore', 'pipe', 'pipe'] });
  const output = [];
  child.stdout.on('data', data => output.push(data));
  child.stderr.on('data', data => output.push(data));
  const watchdog = setTimeout(() => child.kill('SIGTERM'), 90000);
  const [code] = await once(child, 'close');
  clearTimeout(watchdog);
  const text = Buffer.concat(output).toString();
  if (lifetimeControl) {
    assert.notEqual(code, 0, text);
    assert.match(text, /Stream exceeded its 1ms upstream-wait cap/, text);
    assert.equal(submissions.length, 0, text);
    console.log(JSON.stringify({ status: 'PASS_OFFICIAL_QWEN_STREAM_LIFETIME_CONTROL',
      stream_delay_ms: 100, native_cap_ms: 1, model_requests: requests.length,
      actual_provider_requests: 0 }));
  } else {
    assert.equal(code, 0, text);
    assert.ok(!text.includes('EACCES'), text);
    assert.equal(requests.length, 4, text);
    if (clampControl) assert.ok(requests[0].max_tokens < wire.max_tokens,
      'The padded first request must exercise native output clamping');
    const foreground = requests[3];
    const toolResults = foreground.messages.filter(m => m.role === 'tool');
    assert.equal(toolResults.length, 3);
    assert.match(JSON.stringify(toolResults[1].content), /fixture-ok/);
    const events = text.split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line));
    const result = events.find(event => event.type === 'result');
    assert.deepEqual(result.usage, { input_tokens: 300, output_tokens: 96,
      cache_read_input_tokens: 0, total_tokens: 396 });
    assert.deepEqual(submissions, ['/v1/verifier-train', '/v1/submissions/checkpoint', '/v1/submissions']);
    assert.equal(await readFile('/workspace/solution.txt', 'utf8'), 'fixture-ok');
    const receipt = JSON.parse(await readFile('/workspace/.frontier/final-submission.json', 'utf8'));
    assert.equal(receipt.operation, 'final');
    assert.equal(receipt.state, 'accepted');
    const thoughts = foreground.messages.filter(m => m.role === 'assistant' && m.reasoning_content);
    assert.ok(thoughts.length >= 3, 'Reasoning must survive native tool continuation');
    console.log(JSON.stringify({ status: 'PASS_OFFICIAL_QWEN_READONLY_WIRE_FIXTURE',
      near_context_limit: nearLimit, configured_context: nearLimit ? 150000 : 1000000,
      preserved_wire_parameters: clampControl ? null : wire,
      context_clamp_control: clampControl,
      stream_lifetime_fixture: lifetime,
      observed_output_caps: requests.map(request => request.max_tokens), model_requests: requests.length,
      reasoning_continuations: thoughts.length, submissions,
      native_cli_usage: result.usage, output_includes_reasoning_subset: true,
      actual_provider_requests: 0, independent_judge_executed: false }));
  }
} finally {
  if (child?.exitCode === null) child.kill('SIGKILL');
  for (const server of [model, driver]) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}
