// Invoked by the ignored real Driver router integration test; no paid API.
import assert from 'node:assert/strict';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createTransport } from '../transport.mjs';

const root = process.argv[2];
const scope = JSON.parse(await readFile(`${root}/scope.json`));
const binary = process.env.MUSE_TEST_BINARY;
assert.equal(createHash('sha256').update(await readFile(binary)).digest('hex'),
  'dfb3096c91f4767c4d98006460800b7ba906a0b1a408280a926a8dc19a1af64f');
const image = 'ghcr.io/berriai/litellm-non_root@sha256:0aa398456068b6a91d05eb0ae0004d3ca5cec7c7608a901c73d3cdc6833be974';
assert.equal(execFileSync('docker', ['image', 'inspect', image, '--format', '{{.Id}}']).toString().trim(),
  'sha256:0aa398456068b6a91d05eb0ae0004d3ca5cec7c7608a901c73d3cdc6833be974');
const name = `frontier-muse-offline-e2e-${process.pid}`;
const network = `${name}-net`;
const subnet = scope.gateway_host.replace(/\.2$/, '.0/24');
execFileSync('docker', ['network', 'create', '--internal', '--subnet', subnet, network]);
const catalog = { object: 'list', data: [{ id: 'fixture-only', object: 'model',
  metadata: { 'muse-code': { release_date: '2026-01-01', is_hidden: false,
    limit: { context: 1000000, output: 1024 } } } }] };
const children = [];
const transports = [];
const gateway = `http://${scope.gateway_host}:${scope.gateway_port}`;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(check, description, seconds = 30) {
  for (let attempt = 0; attempt < seconds * 10; attempt++) {
    try { const result = await check(); if (result) return result; } catch {}
    await sleep(100);
  }
  throw new Error(`Timed out: ${description}`);
}
async function state() { return (await fetch(`${gateway}/fixture/state`)).json(); }
async function host(route, prompt) {
  const home = `${root}/home-${route}`;
  const workspace = `${root}/workspace-${route}`;
  await mkdir(`${home}/.config/muse`, { recursive: true });
  await mkdir(workspace, { recursive: true });
  const transport = createTransport({ driverBaseUrl: scope[route].url,
    agentToken: scope[route].token, catalog });
  transports.push(transport);
  const base = await transport.listen();
  await writeFile(`${home}/.config/muse/settings.json`, JSON.stringify({ schema_version: 1,
    endpoint_transport: { base_url: base, auth: 'bearer' } }), { mode: 0o600 });
  await writeFile(`${home}/.config/muse/auth.json`, JSON.stringify({ schema_version: 1,
    providers: { meta: { api_key: 'offline-dummy-no-authority' } } }), { mode: 0o600 });
  const child = spawn(binary, ['exec', '--json', '--model', 'fixture-only', '--workspace', workspace,
    '--disable-sandbox', '--disable-approval', '--disable-web-tools', '--no-foreign-personal-context',
    '--trust-workspace', prompt], { cwd: workspace, env: { HOME: home, PATH: '/usr/bin:/bin', LANG: 'C.UTF-8' },
    stdio: ['ignore', 'pipe', 'pipe'] });
  children.push(child);
  const output = [];
  child.stdout.on('data', value => output.push(value));
  child.stderr.on('data', value => output.push(value));
  return { child, exit: once(child, 'close'), output, workspace, transport };
}
try {
  execFileSync('docker', ['run', '-d', '--name', name, '--network', network,
    '--ip', scope.gateway_host, '--cap-drop', 'ALL',
    '--security-opt', 'no-new-privileges', '-e', 'PYTHONPATH=/fixture', '-e', 'LITELLM_LOCAL_MODEL_COST_MAP=True',
    '-v', `${root}:/fixture`, '--entrypoint', '/app/.venv/bin/python', image, '/fixture/gateway.py']);
  await poll(() => state(), 'offline exact-image proxy');
  const normal = await host('normal', 'FRONTIER_OFFLINE_TOOL_MARKER: write the requested harmless fixture file then finish.');
  const deadline = new Promise((_, reject) => {
    const timer = setTimeout(() => reject(new Error('Muse normal run timed out')), 60000);
    timer.unref();
  });
  const exit = await Promise.race([normal.exit, deadline]);
  assert.equal(exit[0], 0, Buffer.concat(normal.output).toString());
  assert.equal(await readFile(`${normal.workspace}/official-tool-result.txt`, 'utf8'), 'fixture-ok');
  const complete = await state();
  assert.ok(complete.tool_replayed && complete.reasoning_replayed && complete.observer_calls > 0);
  await poll(async () => {
    const response = await fetch(`${gateway}/health/liveliness`);
    return response.ok;
  }, 'proxy ready after official host');
  await writeFile(`${root}/normal-state.json`, JSON.stringify(complete));
  await fetch(`${gateway}/fixture/hold`, { method: 'POST' });
  const cancel = await host('cancel', 'Wait on the offline cancellation fixture.');
  await poll(async () => (await state()).held > 0, 'official host held response');
  cancel.child.kill('SIGINT');
  assert.equal((await cancel.exit)[0], 130);
  // Driver cancellation must reach the real proxy and its local provider.
  const cancellation = await poll(async () => {
    const observed = await state(); return observed.closed > 0 && observed;
  }, 'provider stream cancellation', 10);
  await fetch(`${gateway}/fixture/normal`, { method: 'POST' });
  const quota = createTransport({ driverBaseUrl: scope.quota.url, agentToken: scope.quota.token, catalog });
  transports.push(quota);
  const quotaUrl = await quota.listen();
  const beforeQuota = (await state()).requests.length;
  const body = { model: 'fixture-only', input: [], stream: true };
  const first = await fetch(`${quotaUrl}/responses`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  assert.equal(first.status, 200); await first.text();
  const rejected = await fetch(`${quotaUrl}/responses`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const rejectedBody = await rejected.text();
  await writeFile(`${root}/quota-rejection.json`, JSON.stringify({status: rejected.status, body: rejectedBody}));
  // The hard audit session cap is a fail-closed traffic audit error in 9a,
  // distinct from the model quota's HTTP 429. It must never forward a call.
  assert.equal(rejected.status, 500);
  assert.equal((await state()).requests.length, beforeQuota + 1, 'quota rejection must not contact provider');
  const report = { official_host_exit: 0, normal_requests: complete.requests.length,
    observer_requests: complete.observer_calls, namespace_tools: true,
    tool_replayed: complete.tool_replayed, encrypted_reasoning_replayed: complete.reasoning_replayed,
    cancelled_host_exit: 130, provider_stream_closed: cancellation.closed > 0,
    audit_session_cap_rejection_status: rejected.status, quota_provider_requests: 1, exact_litellm_image: image,
    provider: 'local Meta fixture', paid_model_calls: 0, dummy_credentials_only: true };
  await writeFile(`${root}/host-report.json`, JSON.stringify(report));
  console.log(JSON.stringify(report));
} finally {
  for (const child of children) if (child.exitCode === null) child.kill('SIGKILL');
  for (const transport of transports) await transport.close();
  try { await writeFile(`${root}/gateway.log`, execFileSync('docker', ['logs', name])); } catch {}
  try { execFileSync('docker', ['rm', '-f', name], { stdio: 'ignore' }); } catch {}
  execFileSync('docker', ['network', 'rm', network], { stdio: 'ignore' });
}
