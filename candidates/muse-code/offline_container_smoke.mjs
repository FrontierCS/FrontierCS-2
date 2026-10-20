// Run only in the network-disabled, unprivileged offline fixture container.
import assert from 'node:assert/strict';
import http from 'node:http';
import { once } from 'node:events';
import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

let modelRequests = 0;
let observerRequests = 0;
let toolSent = false;
let toolResultSeen = false;
const submissions = [];
const gpuCalls = [];
const payloads = new Map();
const frame = x => `data: ${JSON.stringify(x)}\n\n`;
const model = http.createServer(async (req, res) => {
  assert.equal(req.url, '/v1/responses');
  assert.equal(req.headers['x-frontier-agent-token'], 'offline-driver-capability');
  assert.equal(req.headers.authorization, undefined);
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = JSON.parse(Buffer.concat(chunks));
  assert.equal(body.model, 'fixture-only');
  modelRequests++;
  const tools = body.tools?.flatMap(t => t.tools || [t]) || [];
  const main = tools.some(t => t.name === 'bash');
  observerRequests += Number(tools.some(t => t.name === 'submit_reminder_decision'));
  toolResultSeen ||= body.input?.some?.(i => i.type === 'function_call_output' && i.call_id === 'call_fixture');
  const id = `resp_fixture_${modelRequests}`;
  const response = { id, object: 'response', model: 'fixture-only', status: 'in_progress', output: [] };
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  res.write(frame({ type: 'response.created', sequence_number: 1, response }));
  if (main && !toolSent) {
    toolSent = true;
    res.write(frame({ type: 'response.function_call_arguments.done', sequence_number: 2,
      output_index: 0, item_id: 'fc_fixture', name: 'bash', call_id: 'call_fixture',
      arguments: JSON.stringify({ description: 'Exercise selected final submission fixture',
        command: 'frontier-gpu --version > /workspace/gpu-version.txt && frontier-gpu quota > /workspace/gpu-quota.json && frontier-gpu list > /workspace/gpu-list.json && printf fixture-ok > /workspace/solution.txt && tar -cf /workspace/final.tar -C /workspace solution.txt && frontier-submit train /workspace/final.tar && frontier-submit store-final /workspace/final.tar && frontier-submit final /workspace/final.tar' }) }));
  } else {
    res.write(frame({ type: 'response.output_text.delta', sequence_number: 2,
      output_index: 0, item_id: `msg_${modelRequests}`, content_index: 0, delta: 'Fixture complete.' }));
  }
  res.end(frame({ type: 'response.completed', sequence_number: 3, response: { ...response,
    status: 'completed', usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5,
      output_tokens_details: { reasoning_tokens: 1 } } } }));
});
const driver = http.createServer(async (req, res) => {
  assert.equal(req.headers.authorization, 'Bearer offline-driver-capability');
  if (req.method === 'GET' && ['/v1/run', '/v1/gpu-runners'].includes(req.url)) {
    gpuCalls.push(req.url);
    res.writeHead(200, {'content-type':'application/json'});
    return res.end(JSON.stringify(req.url === '/v1/run' ? {ssh_sessions:{scope:'agent',limit:1024,used:0,remaining:1024,verifier_uses_separate_budget:true}} : {runners:[]}));
  }
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const raw = Buffer.concat(chunks);
  let response;
  if (req.method === 'POST' && req.url === '/v1/payloads') {
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
      state: 'passed', remaining_train_calls: 49, raw_score: 1, metrics: { fixture: true } };
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
  for (const [server, port] of [[model, 8082], [driver, 8080]]) {
    server.listen(port, '127.0.0.1');
    await once(server, 'listening');
  }
  child = spawn('/opt/frontier/candidate/launch', [], { cwd: '/workspace',
    env: { PATH: '/usr/local/bin:/usr/bin:/bin', LANG: 'C.UTF-8',
      FRONTIER_DRIVER_API_URL: 'http://127.0.0.1:8080', FRONTIER_DRIVER_TOKEN_FILE: '/fixture/token',
      FRONTIER_MODEL_API_BASE_URL: 'http://127.0.0.1:8082/v1', FRONTIER_MODEL_PROXY_TOKEN: 'offline-driver-capability',
      FRONTIER_CODEX_MODEL: 'fixture-only' }, stdio: ['ignore', 'pipe', 'pipe'] });
  const stdout = [];
  const stderr = [];
  child.stdout.on('data', data => stdout.push(data));
  child.stderr.on('data', data => stderr.push(data));
  const watchdog = setTimeout(() => child.kill('SIGTERM'), 60000);
  const [code] = await once(child, 'close');
  clearTimeout(watchdog);
  assert.equal(code, 0, Buffer.concat(stderr).toString() + Buffer.concat(stdout).toString());
  assert.deepEqual(submissions, ['/v1/verifier-train', '/v1/submissions/checkpoint', '/v1/submissions']);
  assert.ok(toolSent && toolResultSeen && observerRequests > 0);
  assert.deepEqual(gpuCalls, ['/v1/run','/v1/gpu-runners']);
  assert.equal((await readFile('/workspace/gpu-version.txt','utf8')).trim(),'frontier.gpu-helper/v3');
  assert.equal(JSON.parse(await readFile('/workspace/gpu-quota.json','utf8')).remaining,1024);
  const receipt = JSON.parse(await readFile('/workspace/.frontier/final-submission.json', 'utf8'));
  assert.equal(receipt.state, 'accepted');
  assert.equal(receipt.verification_id, 'verification_final_fixture');
  console.log(JSON.stringify({ status: 'OFFLINE_IMAGE_FIXTURE_PASSED', uid: process.getuid(),
    model_requests: modelRequests, observer_requests: observerRequests, submissions, gpu_calls:gpuCalls,
    real_provider_requests: 0, independent_judge_executed: false }));
} finally {
  if (child?.exitCode === null) child.kill('SIGKILL');
  for (const server of [model, driver]) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}
