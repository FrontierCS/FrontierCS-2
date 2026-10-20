import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { readFile, mkdtemp } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const capability = 'offline-driver-capability';
const modelAlias = 'step-5-preview';
// Step's default tool profile with the launcher's --exclude-tools search_web.
const parentTools = ['agent_send', 'clarify_user', 'create_goal', 'cron_create', 'cron_delete',
  'cron_list', 'edit_file', 'enter_plan_mode', 'exit_plan_mode', 'find_files', 'find_tools',
  'get_goal', 'list_directory', 'read_file', 'run_command', 'search_files', 'subagent',
  'task_create', 'task_get', 'task_list', 'task_update', 'update_goal', 'workflow', 'write_file'];
const requests = [];
const headers = [];
const submissions = [];
const payloads = new Map();
// Request 3 is the subagent's own child process; it answers without tools.
const script = [
  ['run_command', { command: 'test -z "${OPENROUTER_STEPFUN_API_KEY:-}${STEP_API_KEY:-}" && printf fixture-ok > /workspace/solution.txt && tar -cf /workspace/final.tar -C /workspace solution.txt' }],
  ['read_file', { path: '/workspace/solution.txt' }],
  ['subagent', { agent: 'explore', task: 'Confirm that /workspace/solution.txt exists.' }],
  'Child confirms the fixture.',
  ['run_command', { command: 'frontier-submit train /workspace/final.tar && frontier-submit store-final /workspace/final.tar && frontier-submit final /workspace/final.tar' }],
  'Fixture complete.',
];
// INJECT_EMPTY_REPLY=1: the first reply carries reasoning only, no text or tool
// call, and finish_reason stop. Step ends that print run with exit 0; the
// launcher must resume the same session with --continue, after which the
// normal script runs to an accepted final. INJECT_EMPTY_REPLY=length instead
// makes the first reply a few words of text cut off with finish_reason length,
// which must resume the same way. INJECT_EMPTY_REPLY=always makes every reply
// reasoning-only: the launcher must stop after 1 + 20 runs and report failure
// without a final submission.
const injected = process.env.INJECT_EMPTY_REPLY ?? '';
assert.ok(['', '1', 'length', 'always'].includes(injected), injected);
const alwaysEmpty = injected === 'always';
const offset = injected === '1' || injected === 'length' ? 1 : 0;
const cutReply = '\u0000cut';
const maxResumes = 20;
const resumeText = 'Your previous reply was cut off before any answer or tool call.';
if (alwaysEmpty) script.splice(0, script.length, ...Array(1 + maxResumes).fill(''));
else if (offset) script.unshift(injected === 'length' ? cutReply : '');
const childIndex = 3 + offset;
const userTexts = body => body.messages.filter(m => m.role === 'user').map(m => JSON.stringify(m.content));
const callId = index => `call_fixture_${index}`;
const frame = value => `data: ${JSON.stringify(value)}\n\n`;
const model = http.createServer(async (req, res) => {
  assert.equal(req.url, '/v1/chat/completions');
  assert.equal(req.headers.authorization, `Bearer ${capability}`);
  headers.push(Object.keys(req.headers).sort());
  for (const name of Object.keys(req.headers)) {
    assert.ok(!/^(?:x-api-key|api-key|cookie|openai-organization|openai-project|forwarded|via|x-session-affinity|session_id|x-session-id)$/.test(name), name);
    assert.ok(!/^(?:x-forwarded-|x-litellm-|x-proxy-|proxy-)/.test(name), name);
  }
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = JSON.parse(Buffer.concat(chunks));
  assert.equal(body.model, modelAlias);
  assert.equal(body.stream, true);
  // thinkingFormat "openrouter": native OpenRouter reasoning object only.
  assert.deepEqual(body.reasoning, { effort: 'high' });
  assert.ok(!('reasoning_effort' in body), 'No duplicate reasoning control');
  // Step sends its declared output capacity; the gateway alias drops it.
  assert.ok(body.max_tokens > 0 && body.max_tokens <= 64000, String(body.max_tokens));
  assert.ok(!('max_completion_tokens' in body) && !('store' in body));
  assert.deepEqual(body.stream_options, { include_usage: true });
  assert.ok(!body.messages.some(m => m.role === 'developer'), 'Step dialect uses system role');
  const index = requests.length;
  requests.push(body);
  assert.ok(index < script.length, 'Unexpected continuation call');
  const toolNames = (body.tools ?? []).map(tool => tool.function.name).sort();
  if (!alwaysEmpty && index === childIndex) {
    assert.ok(!toolNames.includes('write_file') && !toolNames.includes('subagent'), JSON.stringify(toolNames));
    assert.ok(body.messages.some(m => m.role === 'user' && JSON.stringify(m.content).includes('solution.txt exists')));
  } else {
    assert.deepEqual(toolNames, parentTools, JSON.stringify(toolNames));
    // A resumed run replays the original task prompt as its first user message
    // and adds one resume prompt per resumed run after it.
    const users = userTexts(body);
    assert.equal(users[0], userTexts(requests[0])[0], 'Original task prompt must be replayed');
    assert.ok(users[0].includes('frontier-submit final'), users[0]);
    const resumes = alwaysEmpty ? index : (offset && index > 0 ? 1 : 0);
    assert.equal(users.filter(text => text.includes(resumeText)).length, resumes, JSON.stringify(users));
    if (resumes) assert.ok(users.slice(1).some(text => text.includes(resumeText)), 'Resume prompt follows the task prompt');
    const toolCalls = alwaysEmpty ? [] : [0, 1, 2, 4].map(value => value + offset);
    for (const previous of toolCalls.filter(value => value < index)) {
      const replay = body.messages.find(m => m.role === 'assistant'
        && m.tool_calls?.some(call => call.id === callId(previous)));
      assert.ok(replay, `tool call ${previous} must be replayed`);
      assert.equal(replay.reasoning_content, `Fixture thought ${previous}.`);
      assert.ok(body.messages.some(m => m.role === 'tool' && m.tool_call_id === callId(previous)));
    }
  }
  const step = script[index];
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  const chunk = (delta, finish_reason = null) => ({ id: `chatcmpl_fixture_${index}`,
    object: 'chat.completion.chunk', created: 1, model: modelAlias,
    choices: [{ index: 0, delta, finish_reason }] });
  res.write(frame(chunk({ role: 'assistant', reasoning_content: `Fixture thought ${index}.` })));
  if (Array.isArray(step)) {
    res.write(frame(chunk({ tool_calls: [{ index: 0, id: callId(index),
      type: 'function', function: { name: step[0], arguments: JSON.stringify(step[1]) } }] })));
    res.write(frame(chunk({}, 'tool_calls')));
  } else if (step === cutReply) {
    res.write(frame(chunk({ content: 'So turn-1 ' })));
    res.write(frame(chunk({}, 'length')));
  } else if (step) {
    res.write(frame(chunk({ content: step })));
    res.write(frame(chunk({}, 'stop')));
  } else {
    res.write(frame(chunk({}, 'stop')));
  }
  res.write(frame({ ...chunk({}), choices: [], usage: { prompt_tokens: 75,
    completion_tokens: 24, total_tokens: 99,
    completion_tokens_details: { reasoning_tokens: 20 } } }));
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
      OPENROUTER_STEPFUN_API_KEY: 'must-not-leak', STEP_API_KEY: 'must-not-leak',
      STEP_BASE_URL: 'https://foreign.example', STEP_PERMISSION_PRESET: 'ask' },
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
  const events = text.split('\n').filter(line => line.startsWith('{')).map(line => JSON.parse(line));
  const sessions = events.filter(event => event.type === 'session');
  assert.ok(sessions.every(event => typeof event.id === 'string' && event.id), JSON.stringify(sessions));
  const resumeNotes = diagnostics.match(/resuming after truncated reply/g) ?? [];
  if (alwaysEmpty) {
    assert.equal(code, 2, text + diagnostics);
    assert.equal(requests.length, 1 + maxResumes, text + diagnostics);
    assert.equal(sessions.length, 1 + maxResumes, text);
    assert.equal(new Set(sessions.map(event => event.id)).size, 1, JSON.stringify(sessions));
    assert.equal(resumeNotes.length, maxResumes, diagnostics);
    assert.match(diagnostics, /no accepted final submission receipt/);
    assert.deepEqual(submissions, []);
    await assert.rejects(readFile('/workspace/.frontier/final-submission.json'), { code: 'ENOENT' });
    console.log(JSON.stringify({ status: 'PASS_EVERY_REPLY_EMPTY_STOPS_AFTER_BOUNDED_RESUMES',
      exit_code: code, agent_runs: sessions.length, model_requests: requests.length,
      session_id_reused: true, last_request_roles: requests.at(-1).messages.map(m => m.role), submissions, actual_provider_requests: 0 }));
    process.exitCode = 0;
  } else {
    assert.equal(code, 0, text + diagnostics);
    assert.equal(requests.length, script.length, text + diagnostics);
    assert.equal(events[0].type, 'session');
    assert.equal(sessions.length, 1 + offset, text);
    assert.equal(new Set(sessions.map(event => event.id)).size, 1, JSON.stringify(sessions));
    assert.equal(resumeNotes.length, offset, diagnostics);
    assert.equal(events.at(-1).type, 'agent_settled');
    const assistant = events.filter(event => event.type === 'message_end'
      && event.message.role === 'assistant').map(event => event.message);
    assert.equal(assistant.length, 5 + offset);
    if (offset) {
      assert.equal(assistant[0].stopReason, injected === 'length' ? 'length' : 'stop');
      assert.ok(assistant[0].content.every(block => block.type !== 'toolCall'), JSON.stringify(assistant[0]));
      const text = assistant[0].content.filter(block => block.type === 'text').map(block => block.text).join('');
      assert.equal(text.trim(), injected === 'length' ? 'So turn-1' : '', JSON.stringify(assistant[0]));
    }
    assert.equal(assistant.at(-1).stopReason, 'stop');
    const usage = assistant.map(message => message.usage);
    for (const item of usage) {
      assert.equal(item.input, 75);
      assert.equal(item.output, 24);
      assert.equal(item.totalTokens, 99);
    }
    const toolResults = events.filter(event => event.type === 'tool_execution_end');
    assert.deepEqual(toolResults.map(event => event.toolName),
      ['run_command', 'read_file', 'subagent', 'run_command']);
    assert.ok(toolResults.every(event => event.isError === false), JSON.stringify(toolResults));
    assert.match(JSON.stringify(toolResults[1].result), /fixture-ok/);
    assert.match(JSON.stringify(toolResults[2].result), /Child confirms the fixture/);
    assert.deepEqual(submissions, ['/v1/verifier-train', '/v1/submissions/checkpoint', '/v1/submissions']);
    assert.equal(await readFile('/workspace/solution.txt', 'utf8'), 'fixture-ok');
    const receipt = JSON.parse(await readFile('/workspace/.frontier/final-submission.json', 'utf8'));
    assert.equal(receipt.operation, 'final');
    assert.equal(receipt.state, 'accepted');
    for (const name of ['models.json', 'config.toml', 'auth.json']) {
      const config = await readFile(`/workspace/.frontier-stepcode-home/.stepcode/${name}`, 'utf8');
      assert.ok(!config.includes(capability) && !config.includes('must-not-leak'), name);
    }
    console.log(JSON.stringify({ status: offset ? 'PASS_EMPTY_REPLY_RESUMES_SAME_SESSION'
      : 'PASS_OFFICIAL_STEPCODE_READONLY_WIRE_FIXTURE',
      exit_code: code, agent_runs: sessions.length, model_requests: requests.length,
      subagent_child_requests: 1,
      reasoning: requests[0].reasoning, max_tokens: requests[0].max_tokens,
      reasoning_content_replayed: 4, resume_prompt_replayed: offset > 0,
      ...(offset ? { resumed_request_roles: requests[1].messages.map(m => m.role) } : {}), request_headers: headers[0], submissions,
      native_cli_usage_per_response: usage[0], wall_ms: Date.now() - started,
      actual_provider_requests: 0, independent_judge_executed: false }));
  }
} finally {
  if (child?.exitCode === null) child.kill('SIGKILL');
  for (const server of [model, driver]) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}
