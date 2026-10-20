import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createTransport } from './transport.mjs';

test('pinned official Muse performs a tool turn through dummy-only Driver transport', {
  skip: !process.env.MUSE_TEST_BINARY,
  timeout: 90000,
}, async t => {
  const binary = process.env.MUSE_TEST_BINARY;
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(binary)) hash.update(chunk);
  assert.equal(hash.digest('hex'), 'dfb3096c91f4767c4d98006460800b7ba906a0b1a408280a926a8dc19a1af64f');
  const root = await mkdtemp(path.join(os.tmpdir(), 'frontier-muse-offline-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const home = path.join(root, 'home');
  const workspace = path.join(root, 'workspace');
  await mkdir(path.join(home, '.config/muse'), { recursive: true });
  await mkdir(workspace);
  let requestCount = 0;
  let toolSent = false;
  let toolResultSeen = false;
  let observerRequests = 0;
  let encryptedReasoningSeen = false;
  let holdResponses = false;
  let heldResponseStarted;
  let heldResponseClosed;
  const frames = object => `data: ${JSON.stringify(object)}\n\n`;
  const driver = http.createServer(async (req, res) => {
    assert.equal(req.url, '/v1/responses');
    assert.equal(req.headers['x-frontier-agent-token'], 'offline-driver-capability');
    assert.equal(req.headers.authorization, undefined);
    const parts = [];
    for await (const part of req) parts.push(part);
    const raw = Buffer.concat(parts).toString();
    const body = JSON.parse(raw);
    assert.equal(body.model, 'fixture-only');
    requestCount++;
    const hasShell = body.tools?.some(tool => tool.name === 'bash' || tool.tools?.some(child => child.name === 'bash'));
    if (hasShell && raw.includes('FRONTIER_OFFLINE_TOOL_MARKER')) {
      assert.equal(body.reasoning?.effort, 'max');
    }
    if (body.tools?.some(tool => tool.tools?.some(child => child.name === 'submit_reminder_decision'))) observerRequests++;
    const sendTool = !toolSent && hasShell && raw.includes('FRONTIER_OFFLINE_TOOL_MARKER');
    if (sendTool && process.env.MUSE_TEST_REQUEST_CAPTURE) {
      await writeFile(process.env.MUSE_TEST_REQUEST_CAPTURE, raw, { mode: 0o600 });
    }
    toolResultSeen ||= Array.isArray(body.input) && body.input.some(item => item.type === 'function_call_output' && item.call_id === 'call_fixture');
    encryptedReasoningSeen ||= Array.isArray(body.input) && body.input.some(item =>
      item.type === 'reasoning' && item.encrypted_content === 'offline-opaque-reasoning');
    const id = `resp_fixture_${requestCount}`;
    const response = { id, object: 'response', model: 'fixture-only', status: 'in_progress', output: [] };
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write(frames({ type: 'response.created', sequence_number: 1, response }));
    if (holdResponses) {
      res.once('close', () => heldResponseClosed());
      heldResponseStarted();
      return;
    }
    if (sendTool) {
      toolSent = true;
      res.write(frames({ type: 'response.output_item.done', sequence_number: 2,
        output_index: 0, item: { type: 'reasoning', id: 'rs_fixture', summary: [],
          encrypted_content: 'offline-opaque-reasoning' } }));
      res.write(frames({ type: 'response.function_call_arguments.done', sequence_number: 3,
        output_index: 1, item_id: 'fc_fixture', name: 'bash', call_id: 'call_fixture',
        arguments: JSON.stringify({ command: 'printf fixture-ok > official-tool-result.txt', description: 'Write harmless fixture artifact' }) }));
    } else {
      res.write(frames({ type: 'response.output_text.delta', sequence_number: 2,
        output_index: 0, item_id: `msg_${requestCount}`, content_index: 0, delta: 'Fixture complete.' }));
    }
    res.end(frames({ type: 'response.completed', sequence_number: 4,
      response: { ...response, status: 'completed', usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5,
        output_tokens_details: { reasoning_tokens: 1 } } } }));
  });
  driver.listen(0, '127.0.0.1');
  await once(driver, 'listening');
  // This explicit fixture catalog is never a production model definition.
  const catalog = { object: 'list', data: [{ id: 'fixture-only', object: 'model',
    metadata: { 'muse-code': { release_date: '2026-01-01', is_hidden: false,
      limit: { context: 1000000, output: 1024 } } } }] };
  const transport = createTransport({ driverBaseUrl: `http://127.0.0.1:${driver.address().port}/v1`,
    agentToken: 'offline-driver-capability', catalog });
  const base = await transport.listen();
  t.after(async () => {
    await transport.close();
    driver.closeAllConnections();
    await new Promise(resolve => driver.close(resolve));
  });
  await writeFile(path.join(home, '.config/muse/settings.json'), JSON.stringify({
    schema_version: 1, endpoint_transport: { base_url: base, auth: 'bearer' },
  }), { mode: 0o600 });
  await writeFile(path.join(home, '.config/muse/auth.json'), JSON.stringify({
    schema_version: 1, providers: { meta: { api_key: 'offline-dummy-no-authority' } },
  }), { mode: 0o600 });
  const child = spawn(binary, ['exec', '--json', '--model', 'fixture-only', '--reasoning-effort', 'max',
    '--workspace', workspace, '--disable-sandbox', '--disable-approval',
    '--disable-web-tools', '--no-foreign-personal-context', '--trust-workspace',
    'FRONTIER_OFFLINE_TOOL_MARKER: write the requested harmless fixture file then finish.'], {
    cwd: workspace, env: { HOME: home, PATH: '/usr/bin:/bin', LANG: 'C.UTF-8' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  t.after(() => { if (child.exitCode === null) child.kill('SIGKILL'); });
  const stdout = [];
  const stderr = [];
  child.stdout.on('data', b => stdout.push(b));
  child.stderr.on('data', b => stderr.push(b));
  const [code] = await once(child, 'close');
  assert.equal(code, 0, Buffer.concat(stderr).toString() + Buffer.concat(stdout).toString());
  assert.ok(toolSent, 'real host must advertise and call its own shell tool');
  assert.ok(toolResultSeen, 'real host must replay the tool result');
  assert.ok(encryptedReasoningSeen, 'real host must replay opaque encrypted reasoning');
  assert.equal(await readFile(path.join(workspace, 'official-tool-result.txt'), 'utf8'), 'fixture-ok');
  assert.ok(requestCount >= 2);
  assert.ok(observerRequests > 0, 'official background observers must share the adapter');
  t.diagnostic(`official host requests: ${requestCount}; observer requests: ${observerRequests}; tool replay verified; dummy credentials only`);
  holdResponses = true;
  const started = new Promise(resolve => { heldResponseStarted = resolve; });
  const closed = new Promise(resolve => { heldResponseClosed = resolve; });
  const cancelled = spawn(binary, ['exec', '--json', '--model', 'fixture-only', '--reasoning-effort', 'max',
    '--workspace', workspace, '--disable-sandbox', '--disable-approval',
    '--disable-web-tools', '--no-foreign-personal-context', 'Wait on the offline cancellation fixture.'], {
    cwd: workspace, env: { HOME: home, PATH: '/usr/bin:/bin', LANG: 'C.UTF-8' }, stdio: 'ignore',
  });
  t.after(() => { if (cancelled.exitCode === null) cancelled.kill('SIGKILL'); });
  const cancelledExit = once(cancelled, 'close');
  await started;
  cancelled.kill('SIGINT');
  assert.equal((await cancelledExit)[0], 130);
  await closed;
});
