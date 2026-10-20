import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, rm, chmod, lstat, readdir, symlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { cleanupRuntime, main, prepare, PROMPT, selectModel } from './launcher.mjs';
import { createTransport } from './transport.mjs';

async function inputs(t) {
  const temp = await mkdtemp(path.join(os.tmpdir(), 'muse-launcher-fixture-'));
  t.after(() => rm(temp, { recursive: true, force: true }));
  const root = path.join(temp, 'candidate');
  const workspace = path.join(temp, 'workspace');
  const taskReadme = path.join(temp, 'README.md');
  await mkdir(workspace);
  for (const file of ['muse-code/muse', 'bin/frontier-submit', 'bin/frontier-gpu',
    'bin/frontier-gpu-connect-proxy', 'skills/frontier-submit/SKILL.md', 'skills/frontier-gpu/SKILL.md']) {
    await mkdir(path.dirname(path.join(root, file)), { recursive: true });
    await writeFile(path.join(root, file), 'fixture');
  }
  await writeFile(taskReadme, 'fixture task');
  await writeFile(path.join(root, 'muse-code/model-catalog.json'), JSON.stringify({
    object: 'list', data: [{ id: 'fixture-only', object: 'model',
      metadata: { 'muse-code': { release_date: '2026-01-01', is_hidden: false,
        limit: { context: 1000, output: 100 } } } }],
  }));
  return { root, workspace, taskReadme, env: { FRONTIER_DRIVER_API_URL: 'http://10.1.2.3:8080',
    FRONTIER_MODEL_API_BASE_URL: 'http://10.1.2.3:8082/v1', FRONTIER_MODEL_PROXY_TOKEN: 'fixture-capability',
    FRONTIER_DRIVER_TOKEN_FILE: '/var/run/frontier/token', FRONTIER_CODEX_MODEL: 'fixture-only',
    PATH: '/usr/bin:/bin', META_API_KEY: 'fixture-never-forward', AWS_SECRET_ACCESS_KEY: 'fixture-never-forward',
    CODEX_API_KEY: 'fixture-never-forward', ANTHROPIC_AUTH_TOKEN: 'fixture-never-forward',
    OPENAI_BASE_URL: 'https://provider.invalid', MUSE_CONFIG: '/developer/private',
    LD_LIBRARY_PATH: '/wrong/lib', NODE_OPTIONS: '--inspect', HTTP_PROXY: 'http://proxy.invalid' } };
}

test('fresh private config, task prompt and governed capabilities without provider environment', async t => {
  const config = await inputs(t);
  const prepared = await prepare(config);
  assert.equal(prepared.token, 'fixture-capability');
  assert.equal(prepared.childEnv.FRONTIER_DRIVER_TOKEN_FILE, '/var/run/frontier/token');
  for (const key of ['META_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'CODEX_API_KEY', 'ANTHROPIC_AUTH_TOKEN',
    'OPENAI_BASE_URL', 'MUSE_CONFIG', 'FRONTIER_MODEL_PROXY_TOKEN', 'LD_LIBRARY_PATH', 'NODE_OPTIONS', 'HTTP_PROXY']) {
    assert.equal(prepared.childEnv[key], undefined);
  }
  assert.ok(prepared.childEnv.HOME.startsWith(config.workspace));
  assert.ok(prepared.args.includes('--disable-web-tools'));
  assert.ok(prepared.args.includes('--no-foreign-personal-context'));
  assert.ok(!prepared.args.includes('--max-model-steps'));
  assert.equal(await readFile(path.join(prepared.runtime, 'prompt.txt'), 'utf8'), PROMPT);
  assert.match(PROMPT, /must save.*answer/s);
  assert.match(PROMPT, /store-final early/);
  assert.match(PROMPT, /independent final/);
  // Same train policy as the shared skill: keep working, then read the verification.
  assert.match(PROMPT, /keep working while it runs/);
  assert.match(PROMPT, /frontier-submit train/);
  assert.match(PROMPT, /frontier-submit\s+status <verification_id>/);
  assert.doesNotMatch(PROMPT, /Wait for long-running train/);
});

test('provider bypass, cross-run endpoints, wrong alias and unsupported effort fail closed', async t => {
  const config = await inputs(t);
  for (const update of [
    { FRONTIER_MODEL_DIRECT_BASE_URL: 'https://api.meta.ai/v1' },
    { FRONTIER_MODEL_API_BASE_URL: 'http://10.1.2.4:8082/v1' },
    { FRONTIER_CODEX_MODEL: 'unreviewed-model' },
    { FRONTIER_CODEX_REASONING_EFFORT: 'none' },
  ]) {
    await assert.rejects(prepare({ ...config, env: { ...config.env, ...update } }));
  }
  await rm(path.join(config.root, 'muse-code/model-catalog.json'));
  await assert.rejects(prepare(config), 'missing real metadata cannot produce a default catalog');
});

// Every production model of this overlay: [alias, CandidateVersion effort].
const MODELS = [['muse-spark-1.3', 'high'], ['muse-wonder-1.0', 'max']];

async function production(t, alias, effort) {
  const config = await inputs(t);
  const catalog = await readFile(new URL('./model-catalog.json', import.meta.url), 'utf8');
  await writeFile(path.join(config.root, 'muse-code/model-catalog.json'), catalog);
  config.env.FRONTIER_CODEX_MODEL = alias;
  if (effort === undefined) delete config.env.FRONTIER_CODEX_REASONING_EFFORT;
  else config.env.FRONTIER_CODEX_REASONING_EFFORT = effort;
  return config;
}

for (const [alias, effort] of MODELS) {
  test(`${alias}: host is served only its own minimal row and the effort unchanged`, async t => {
    const prepared = await prepare(await production(t, alias, effort));
    // Exactly what the single-row image served: no other model, notes or invented metadata.
    assert.equal(JSON.stringify(prepared.catalog), JSON.stringify({ object: 'list',
      data: [{ id: alias, object: 'model', created: 0, owned_by: 'meta' }] }));
    const { args } = prepared;
    assert.equal(args[args.indexOf('--model') + 1], alias);
    assert.equal(args[args.indexOf('--reasoning-effort') + 1], effort);
    const transport = createTransport({ driverBaseUrl: 'http://10.1.2.3:8082/v1',
      agentToken: prepared.token, catalog: prepared.catalog });
    const base = await transport.listen();
    t.after(() => transport.close());
    assert.deepEqual(await (await fetch(base + '/muse-code/models')).json(), prepared.catalog);
  });
}

test('efforts pass through; only a model with a documented range is narrowed', async t => {
  for (const effort of ['minimal', 'low', 'medium', 'high', 'xhigh', 'max']) {
    const { args } = await prepare(await production(t, 'muse-wonder-1.0', effort));
    assert.equal(args[args.indexOf('--reasoning-effort') + 1], effort);
  }
  await assert.rejects(prepare(await production(t, 'muse-wonder-1.0', 'ultra')));
  // Spark keeps its original behaviour: default high and the full generic list.
  for (const [effort, expected] of [[undefined, 'high'], ['ultra', 'ultra'], ['max', 'max']]) {
    const { args } = await prepare(await production(t, 'muse-spark-1.3', effort));
    assert.equal(args[args.indexOf('--reasoning-effort') + 1], expected);
  }
  await assert.rejects(prepare(await production(t, 'muse-spark-1.2', 'high')));
});

test('ambiguous or malformed catalogs fail closed', () => {
  const row = { id: 'm', object: 'model' };
  assert.deepEqual(selectModel({ object: 'list', data: [row, { id: 'n', object: 'model' }] }, 'm', 'high'),
    { object: 'list', data: [row] });
  for (const file of [null, { object: 'list' }, { object: 'page', data: [row] },
    { object: 'list', data: [row, row] }, { object: 'list', data: [{ id: 'm', object: 'other' }] },
    { object: 'list', data: [row], frontier: { m: { reasoning_effort_range: ['max', 'low'] } } },
    { object: 'list', data: [row], frontier: { m: { reasoning_effort_range: 'max' } } },
    { object: 'list', data: [row], frontier: { m: { reasoning_effort_range: ['low', 'medium'] } } }]) {
    assert.throws(() => selectModel(file, 'm', 'high'));
  }
});

test('reviewed gateway routes cover exactly the catalog models', async () => {
  const catalog = JSON.parse(await readFile(new URL('./model-catalog.json', import.meta.url), 'utf8'));
  const routes = await readFile(new URL('./gateway-models.yaml', import.meta.url), 'utf8');
  const names = [...routes.matchAll(/^  - model_name: (\S+)$/gm)].map(match => match[1]);
  assert.deepEqual(names, catalog.data.map(row => row.id));
  assert.deepEqual(names, MODELS.map(([alias]) => alias));
  for (const name of names) assert.ok(routes.includes(`      model: openai/${name}\n`));
  assert.equal(routes.match(/supports_native_streaming: true/g).length, names.length);
  assert.doesNotMatch(routes, /api_key: (?!os\.environ\/)/);
});

test('nonroot cleanup removes a real read-only module cache without following skill symlinks', {
  skip: process.getuid?.() === 0 ? 'Requires nonroot filesystem permissions' : false,
}, async t => {
  const config = await inputs(t);
  const prepared = await prepare(config);
  const module = path.join(prepared.childEnv.HOME, 'go/pkg/mod/example.org/module@v1.0.0');
  await mkdir(module, { recursive: true });
  await writeFile(path.join(module, 'go.mod'), 'module example.org/module\n', { mode: 0o444 });
  await chmod(module, 0o555);
  const outside = path.join(config.root, 'outside-cache');
  await mkdir(outside, { mode: 0o555 });
  await symlink(outside, path.join(prepared.runtime, 'outside-link'));
  await assert.rejects(rm(prepared.runtime, { recursive: true, force: true }), { code: 'EACCES' });
  await cleanupRuntime(prepared.runtime);
  await assert.rejects(lstat(prepared.runtime), { code: 'ENOENT' });
  assert.equal((await lstat(outside)).mode & 0o777, 0o555);
  assert.equal(await readFile(path.join(config.root, 'skills/frontier-submit/SKILL.md'), 'utf8'), 'fixture');
});

async function readonlyCacheExecutable(config, { final, exitCode }) {
  await writeFile(path.join(config.root, 'muse-code/muse'), `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
const moduleCache = path.join(process.env.HOME, 'go/pkg/mod/example.org/module@v1.0.0');
fs.mkdirSync(moduleCache, { recursive: true });
fs.writeFileSync(path.join(moduleCache, 'go.mod'), 'module example.org/module\\n', { mode: 0o444 });
fs.chmodSync(moduleCache, 0o555);
if (${JSON.stringify(final)}) {
  const receipt = path.join(process.cwd(), '.frontier/final-submission.json');
  fs.mkdirSync(path.dirname(receipt), { recursive: true, mode: 0o700 });
  fs.writeFileSync(receipt, JSON.stringify({ schema_version: 'frontier.candidate-submit-result/v1',
    operation: 'final', state: 'accepted', payload_id: 'payload_fixture',
    verification_id: 'verification_fixture' }), { mode: 0o600 });
}
process.exit(${exitCode});
`, { mode: 0o755 });
  await chmod(path.join(config.root, 'muse-code/muse'), 0o755);
  config.env.PATH = process.env.PATH;
}

test('successful final keeps its exit status while its read-only runtime is removed', async t => {
  const config = await inputs(t);
  await readonlyCacheExecutable(config, { final: true, exitCode: 0 });
  assert.equal(await main(config), 0);
  assert.ok(!(await readdir(config.workspace)).some(name => name.startsWith('.frontier-muse-')));
  const receipt = JSON.parse(await readFile(path.join(config.workspace, '.frontier/final-submission.json'), 'utf8'));
  assert.equal(receipt.state, 'accepted');
});

test('read-only cache cleanup does not turn a genuine no-final Agent failure into success', async t => {
  const config = await inputs(t);
  await readonlyCacheExecutable(config, { final: false, exitCode: 7 });
  assert.equal(await main(config), 7);
  assert.ok(!(await readdir(config.workspace)).some(name => name.startsWith('.frontier-muse-')));
});
