import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { lstat, readFile, mkdir, mkdtemp, writeFile, symlink, rm, readdir, chmod } from 'node:fs/promises';
import path from 'node:path';
import { isIP } from 'node:net';
import { pathToFileURL } from 'node:url';
import { createTransport } from './transport.mjs';

export const PROMPT = `Solve the evaluation task in /frontier/task/README.md.
Before any task work, read the complete task README and the installed skill at
/opt/frontier/candidate/skills/frontier-submit/SKILL.md. The task README determines
the exact train/final archive format. Work in /workspace. Use frontier-submit
train for feedback within the task's quotas. You must save a usable answer that
you select with frontier-submit store-final early, and update that selected
answer after meaningful improvements. Save count is unlimited when the command
reports remaining_checkpoint_submissions: null. Saving is not judging.
When ready, use frontier-submit final once as your last platform action, then
finish without further tool calls. Your final text is not a submitted answer.
The Agent window is task-controlled (normally 18000 seconds); independent final
judging does not use that window. No train score automatically selects a final.
Train verification runs independently; you may keep working while it runs.
Start it with frontier-submit train and read it with frontier-submit
status <verification_id>. When Muse returns a session_id for a long-running
command, use bash_input for that existing shell session; do not resubmit an
archive to check on a verification. Do not replace the submission
client or inspect Driver tokens or the model adapter's private configuration.
If the task supports Agent GPUs, read the installed frontier-gpu skill and use
the Candidate-owned GPU v3 client within task budgets. Check frontier-gpu quota
before debugging; do not mix GPU clients. Web search is disabled. Use task files
and local tools. Read any task-specific instructions before choosing an approach.
`;

// Ordered weakest to strongest, as accepted by the official host's --reasoning-effort.
const EFFORTS = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'];

// The image may carry several reviewed rows. The host is served exactly the one
// row of this Candidate's immutable alias, never the other models and never the
// platform-only `frontier` notes. An optional documented effort range for that
// model is enforced here; a model without one keeps the generic list.
export function selectModel(file, alias, effort) {
  const rows = file?.object === 'list' && Array.isArray(file.data)
    ? file.data.filter(row => row?.id === alias) : [];
  if (rows.length !== 1 || rows[0].object !== 'model') {
    throw new Error('Missing authenticated model catalog row for the immutable model alias');
  }
  const range = file.frontier?.[alias]?.reasoning_effort_range;
  if (range !== undefined) {
    const [low, high] = Array.isArray(range) && range.length === 2 ? range.map(e => EFFORTS.indexOf(e)) : [-1, -1];
    if (low < 0 || high < low) throw new Error('Invalid reviewed reasoning effort range');
    const index = EFFORTS.indexOf(effort);
    if (index < low || index > high) throw new Error('Reasoning effort is outside the model\'s documented range');
  }
  return { object: 'list', data: rows };
}

function endpoint(value, port, pathname) {
  const u = new URL(value);
  if (u.protocol !== 'http:' || !isIP(u.hostname) || u.port !== String(port) ||
      u.pathname.replace(/\/$/, '') !== pathname || u.username || u.password || u.search || u.hash) {
    throw new Error('Expected the fixed run-local Driver endpoint');
  }
  return u;
}

export async function prepare({ env = process.env, workspace = '/workspace',
  root = '/opt/frontier/candidate', taskReadme = '/frontier/task/README.md' } = {}) {
  if (env.FRONTIER_MODEL_DIRECT_BASE_URL) throw new Error('Muse requires the governed Driver model route');
  const driver = endpoint(env.FRONTIER_DRIVER_API_URL, 8080, '');
  const model = endpoint(env.FRONTIER_MODEL_API_BASE_URL, 8082, '/v1');
  if (driver.hostname !== model.hostname) throw new Error('Driver/model endpoint IP mismatch');
  const token = env.FRONTIER_MODEL_PROXY_TOKEN;
  if (typeof token !== 'string' || !token || token.length > 16384 || /[\r\n]/.test(token)) {
    throw new Error('Missing or invalid Agent-scoped Driver capability');
  }
  const alias = env.FRONTIER_CODEX_MODEL;
  if (typeof alias !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,255}$/.test(alias)) {
    throw new Error('Missing or invalid immutable model alias');
  }
  // Passed to the official host unchanged; there is no platform-side remapping.
  const effort = env.FRONTIER_CODEX_REASONING_EFFORT || 'high';
  if (!EFFORTS.includes(effort)) throw new Error('Unsupported Meta reasoning effort');
  const binary = path.join(root, 'muse-code/muse');
  const catalogPath = path.join(root, 'muse-code/model-catalog.json');
  for (const file of [binary, taskReadme, catalogPath,
    ...['frontier-submit', 'frontier-gpu'].map(skill => path.join(root, 'skills', skill, 'SKILL.md')),
    ...['frontier-submit', 'frontier-gpu', 'frontier-gpu-connect-proxy'].map(tool => path.join(root, 'bin', tool))]) {
    if (!(await lstat(file)).isFile()) throw new Error('Required Candidate input must be a regular file');
  }
  const catalog = selectModel(JSON.parse(await readFile(catalogPath, 'utf8')), alias, effort);
  const runtime = await mkdtemp(path.join(workspace, '.frontier-muse-'));
  try {
    const home = path.join(runtime, 'home');
    const config = path.join(home, '.config/muse');
    await mkdir(path.join(config, 'skills'), { recursive: true, mode: 0o700 });
    await mkdir(path.join(runtime, 'tmp'), { mode: 0o700 });
    for (const skill of ['frontier-submit', 'frontier-gpu']) {
      await symlink(path.join(root, 'skills', skill), path.join(config, 'skills', skill));
    }
    const childEnv = { ...env };
    for (const key of Object.keys(childEnv)) {
      if (/^(META_|MUSE_|OPENAI_|ANTHROPIC_|CODEX_|KIMI_|LITELLM_|AWS_|AZURE_|GOOGLE_|HF_)/.test(key) ||
          /(?:API_KEY|AUTH_TOKEN|SECRET|PASSWORD)$/.test(key) ||
          ['LD_LIBRARY_PATH', 'LD_PRELOAD', 'NODE_OPTIONS', 'PYTHONPATH', 'PYTHONSTARTUP',
            'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy',
            'FRONTIER_MODEL_PROXY_TOKEN', 'FRONTIER_MODEL_DIRECT_BASE_URL'].includes(key)) delete childEnv[key];
    }
    Object.assign(childEnv, { HOME: home, XDG_CONFIG_HOME: path.join(home, '.config'),
      XDG_CACHE_HOME: path.join(runtime, 'cache'), XDG_DATA_HOME: path.join(runtime, 'data'),
      TMPDIR: path.join(runtime, 'tmp'), PATH: path.join(root, 'bin') + ':' + (env.PATH || '/usr/bin:/bin') });
    const prompt = path.join(runtime, 'prompt.txt');
    await writeFile(prompt, PROMPT, { mode: 0o600, flag: 'wx' });
    return { binary, runtime, workspace, config, childEnv, catalog, token, modelUrl: model.href.replace(/\/$/, ''),
      args: ['exec', '--json', '--model', alias, '--reasoning-effort', effort,
        '--workspace', workspace, '--prompt-file', prompt, '--disable-sandbox',
        '--disable-approval', '--disable-web-tools', '--no-foreign-personal-context', '--trust-workspace'] };
  } catch (error) {
    await cleanupRuntime(runtime);
    throw error;
  }
}

export async function cleanupRuntime(runtime) {
  try {
    await rm(runtime, { recursive: true, force: true });
    return;
  } catch (error) {
    if (!['EACCES', 'EPERM'].includes(error.code)) throw error;
  }
  // Go module caches contain read-only directories. Repair only this private
  // runtime's directories, never skill symlinks pointing outside it.
  async function writableDirectories(directory) {
    const stat = await lstat(directory);
    if (!stat.isDirectory()) return;
    await chmod(directory, (stat.mode & 0o777) | 0o700);
    for (const name of await readdir(directory)) {
      await writableDirectories(path.join(directory, name));
    }
  }
  await writableDirectories(runtime);
  await rm(runtime, { recursive: true, force: true });
}

async function acceptedFinal(workspace) {
  try {
    const file = path.join(workspace, '.frontier/final-submission.json');
    const stat = await lstat(file);
    if (!stat.isFile() || (stat.mode & 0o077) || stat.size < 1 || stat.size > 16384) return false;
    const value = JSON.parse(await readFile(file, 'utf8'));
    return Object.keys(value).sort().join(',') === 'operation,payload_id,schema_version,state,verification_id' &&
      value.schema_version === 'frontier.candidate-submit-result/v1' && value.operation === 'final' &&
      value.state === 'accepted' && typeof value.payload_id === 'string' && value.payload_id.startsWith('payload_') &&
      typeof value.verification_id === 'string' && value.verification_id.startsWith('verification_');
  } catch { return false; }
}

export async function main(options) {
  const prepared = await prepare(options);
  const transport = createTransport({ driverBaseUrl: prepared.modelUrl,
    agentToken: prepared.token, catalog: prepared.catalog });
  let child;
  let childClosed;
  let cancelled = false;
  const stop = signal => {
    cancelled = true;
    if (child?.pid) {
      try { process.kill(-child.pid, signal); } catch (error) { if (error.code !== 'ESRCH') throw error; }
    }
  };
  const onTerm = () => stop('SIGTERM');
  const onInt = () => stop('SIGINT');
  process.on('SIGTERM', onTerm);
  process.on('SIGINT', onInt);
  try {
    const base = await transport.listen();
    await writeFile(path.join(prepared.config, 'settings.json'), JSON.stringify({ schema_version: 1,
      endpoint_transport: { base_url: base, auth: 'bearer' } }), { mode: 0o600, flag: 'wx' });
    await writeFile(path.join(prepared.config, 'auth.json'), JSON.stringify({ schema_version: 1,
      providers: { meta: { api_key: 'frontier-local-dummy-no-provider-authority' } } }), { mode: 0o600, flag: 'wx' });
    if (cancelled) return 143;
    child = spawn(prepared.binary, prepared.args, { cwd: prepared.workspace, env: prepared.childEnv,
      stdio: ['inherit', 'pipe', 'pipe'], detached: true });
    // Give Muse independent blocking descriptors; Node forwards log backpressure.
    child.stdout.pipe(process.stdout, { end: false });
    child.stderr.pipe(process.stderr, { end: false });
    childClosed = once(child, 'close').catch(() => {});
    const [code, signal] = await once(child, 'exit');
    if (cancelled) return signal === 'SIGINT' ? 130 : 143;
    if (code === 0 || await acceptedFinal(prepared.workspace)) return 0;
    return code || 1;
  } finally {
    process.off('SIGTERM', onTerm);
    process.off('SIGINT', onInt);
    if (child?.pid) {
      try { process.kill(-child.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
    }
    if (childClosed) await childClosed;
    await transport.close();
    await cleanupRuntime(prepared.runtime);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then(code => { process.exitCode = code; }).catch(() => {
    // Never render token-bearing network/config errors into solver-visible logs.
    process.stderr.write('Muse Candidate startup or supervision failed; verify immutable inputs and Driver endpoints.\n');
    process.exitCode = 1;
  });
}
