"""Signed fixed-testbed shell service with a separate unprivileged command UID.

The root-owned service verifies each scheduler request. Model-requested shell and
patch commands run through a fixed no-new-privileges launcher without auth env.
The Driver starts one such container per (instance, variant) beside the GPU
parent; after its timed batch the scheduler asks it to shut down.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import locale
import multiprocessing
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from instance_manifest import expected_groups
from . import sidecar_env
from .swe_case_failure import CaseDeadlineExceeded, failure
from .tool_auth import ToolAuthError, ToolAuthVerifier


TESTBED = Path('/testbed')
MAX_COMMAND_SECONDS = 60
# Patch extraction is the harness's own read, not a model command. In a fresh
# container every index entry is stat-dirty (layer extraction gives new ctime
# and inode numbers), so the first diff re-reads the whole tree. Run
# run_01M4XC2Y2G4XXK2S50WQ3PPE14 measured that cold read at 56.04 s
# (django reference, complete) and >60.39 s (django vanilla, timed out) under
# node contention, against ~0.5 s warm: 180 s is 3x the largest observation.
MAX_PATCH_SECONDS = 180
# Bounded preparation outside the timed episode. /ready requests can return
# preparing while this runs; a failed refresh must never release the barrier.
INDEX_REFRESH_SECONDS = 600
READY_WAIT_SECONDS = 50
COMMAND_LAUNCHER = Path('/opt/frontier/judge/serving_eval/swe_tool_exec.py')
WORKER_PYTHON = Path('/opt/frontier/worker-python/bin/python')


IDENTITY_FIELDS = frozenset({
    'operation_id', 'group_id', 'placement_id', 'instance_id', 'child_id', 'phase',
})
SHUTDOWN_UID = 10001


def bootstrap_identity(instance_id, output, identity=None):
    """Map only the Driver's declared-sidecar identity; the instance is baked in argv.

    ``identity`` is the dictionary ``sidecar_env.install`` returned; when absent
    the Driver variable is read directly (the key material is left untouched).
    """
    raw = sidecar_env.read_identity() if identity is None else identity
    if instance_id not in expected_groups('eval')['swe/public']:
        raise ValueError('invalid Driver public identity or fixed instance')
    if raw['phase'] not in {'train', 'eval'}:
        raise ValueError('SWE testbeds exist only in Verifier phases')
    identity = {
        'operation_id': raw['operation_id'], 'group_id': raw['operation_id'],
        'placement_id': raw['operation_id'], 'instance_id': instance_id,
        'child_id': raw['name'], 'phase': raw['phase'],
    }
    with Path(output).open('x') as handle:
        json.dump(identity, handle)
        handle.write('\n')
    return identity


def kill_uid_processes(uid=SHUTDOWN_UID):
    """SIGKILL every process the model-command uid still owns."""
    killed = 0
    for _ in range(3):
        found = False
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != uid:
                    continue
                os.kill(int(entry.name), signal.SIGKILL)
                found = True
                killed += 1
            except (OSError, ProcessLookupError):
                continue
        if not found:
            break
        time.sleep(0.2)
    return killed


class OutputLimitExceeded(RuntimeError):
    pass


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _command_argv(command):
    return [str(WORKER_PYTHON), str(COMMAND_LAUNCHER), '/bin/bash', '-lc', command]


def _open_private_dir(parent_fd, name):
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        os.chown(name, 10001, 10001, dir_fd=parent_fd, follow_symlinks=False)
    except FileExistsError:
        pass
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                         dir_fd=parent_fd)
    info = os.fstat(descriptor)
    if info.st_uid != 10001 or info.st_gid != 10001:
        os.close(descriptor)
        raise PermissionError('private tool runtime directory identity changed')
    os.fchmod(descriptor, 0o700)
    return descriptor


def _ensure_private_runtime_dirs():
    """Restore image-owned scratch dirs without following mutable symlinks."""
    tmp = os.open('/tmp', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    runtime = None
    cache = None
    try:
        runtime = _open_private_dir(tmp, 'frontier-tool')
        cache = _open_private_dir(runtime, 'cache')
    finally:
        for descriptor in (cache, runtime, tmp):
            if descriptor is not None:
                os.close(descriptor)


def _run_shell(command, timeout, max_output_bytes, *, argv=None, cwd=None):
    """Keep original text/output semantics, with bounded capture and group cleanup."""
    process = None
    completed = False
    try:
        if argv is None: _ensure_private_runtime_dirs()
        process = subprocess.Popen(
            _command_argv(command) if argv is None else argv, cwd=TESTBED if cwd is None else cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        streams = [bytearray(), bytearray()]
        size = 0
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            for index, pipe in enumerate((process.stdout, process.stderr)):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, index)
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return 124, 'command timed out'
                events = selector.select(min(remaining, 0.05))
                for key, _ in events:
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    size += len(data)
                    if size > max_output_bytes:
                        raise OutputLimitExceeded('command output exceeds publisher limit')
                    streams[key.data].extend(data)
            encoding = locale.getpreferredencoding(False)
            output = ''.join(
                bytes(part).decode(encoding).replace('\r\n', '\n').replace('\r', '\n')
                for part in streams
            )
            completed = True
            return process.returncode, output
    except OutputLimitExceeded:
        raise
    except Exception as exc:
        return 1, 'sandbox error: ' + type(exc).__name__
    finally:
        if process is not None:
            # Successful commands may leave background services for later tools.
            # Timeout/error cleanup cannot contain descendants that detach.
            if not completed:
                _kill_group(process)
            process.wait()
            process.stdout.close()
            process.stderr.close()


def _patch_diagnostics_worker(connection):
    counts = {'regular': 0, 'directory': 0, 'symlink': 0, 'other': 0}
    try:
        for current, directories, files in os.walk(TESTBED, followlinks=False):
            if Path(current) == TESTBED:
                directories[:] = [name for name in directories if name != '.git']
            counts['directory'] += len(directories)
            for name in files:
                try:
                    mode = (Path(current) / name).lstat().st_mode
                    if stat.S_ISLNK(mode):
                        counts['symlink'] += 1
                    elif stat.S_ISREG(mode):
                        counts['regular'] += 1
                    else:
                        counts['other'] += 1
                except OSError:
                    counts['other'] += 1
            if sum(counts.values()) >= 1_000_000:
                break
        uid_processes = 0
        for entry in Path('/proc').iterdir():
            try:
                uid_processes += entry.name.isdigit() and entry.stat().st_uid == 10001
            except OSError:
                pass
        connection.send({'worktree_counts': counts,
                         'count_capped': sum(counts.values()) >= 1_000_000,
                         'index_lock_present': (TESTBED / '.git/index.lock').exists(),
                         'uid_processes': uid_processes})
    finally:
        connection.close()


def _bounded_patch_diagnostics():
    parent, child = multiprocessing.get_context('spawn').Pipe(duplex=False)
    process = multiprocessing.get_context('spawn').Process(
        target=_patch_diagnostics_worker, args=(child,), daemon=True)
    process.start()
    child.close()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join()
        parent.close()
        return {'diagnostics_timed_out': True}
    try:
        try:
            return parent.recv() if parent.poll() else {'diagnostics_failed': True}
        except EOFError:
            return {'diagnostics_failed': True}
    finally:
        parent.close()


def _refresh_index(*, runner=None, testbed=None, timeout=INDEX_REFRESH_SECONDS):
    """Refresh the testbed index's cached stat data once, as the tool uid.

    ``update-index --refresh`` only rewrites stat fields of entries whose
    content still equals the indexed blob; no blob, mode or path changes, so
    every later diff has identical content. It runs before /ready, outside any
    timed episode, so the per-container full-tree read is not paid inside
    /patch (or the model's first Git command). On failure or
    timeout the index is left as the image shipped it (a lock left by a killed
    refresh is removed; nothing else uses the testbed yet).
    """
    runner = _run_shell if runner is None else runner
    testbed = TESTBED if testbed is None else Path(testbed)
    lock = testbed / '.git/index.lock'
    if lock.exists():
        return {'refreshed': False, 'reason': 'index_lock_present', 'seconds': 0.0}
    started = time.monotonic()
    try:
        code, _ = runner('git update-index -q --refresh >/dev/null 2>&1', timeout, 65536)
    except OutputLimitExceeded:
        code = 1
    if code != 0 and lock.exists():
        lock.unlink()
    return {'refreshed': code == 0, 'exit_code': code, 'seconds': time.monotonic() - started}


def _read_patch(timeout, max_output_bytes, *, runner=None, testbed=None):
    """Preserve the original patch and timeout semantics while attributing failures."""
    runner = _run_shell if runner is None else runner
    testbed = TESTBED if testbed is None else Path(testbed)
    if (testbed / '.git/index.lock').exists():
        return 1, '', {'stage': 'git_index_lock', 'stage_seconds': 0.0,
                       **_bounded_patch_diagnostics()}
    # Add only paths that are currently untracked to a disposable copy of the
    # index, then emit one coherent diff for every tracked and new worktree edit.
    command = r'''set -e
paths=$(mktemp)
index=$(mktemp)
trap 'rm -f "$paths" "$index"' EXIT
git ls-files --others --exclude-standard -z >"$paths"
cp "$(git rev-parse --git-path index)" "$index"
xargs -0 -r env GIT_INDEX_FILE="$index" git --literal-pathspecs add --intent-to-add -- <"$paths"
env GIT_INDEX_FILE="$index" git --no-pager diff HEAD'''
    started = time.monotonic()
    code, output = runner(command, timeout, max_output_bytes)
    diff_seconds = time.monotonic() - started
    diagnostics = {} if code == 0 else _bounded_patch_diagnostics()
    return code, output, {'stage': 'complete' if code == 0 else 'git_diff',
                          'stage_seconds': diff_seconds, **diagnostics}


class ToolService:
    routes = frozenset({('GET', '/ready'), ('POST', '/execute'), ('POST', '/patch'), ('POST', '/shutdown')})

    def __init__(self, identity, *, max_request_bytes, max_output_bytes, auth=None,
                 shell_runner=None, patch_reader=None, testbed=None, exit_hook=None,
                 index_refresh=None):
        if (not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS
                or any(not isinstance(value, str) or not value for value in identity.values())):
            raise ValueError('expected exactly the six nonempty prepared identity fields')
        if identity['instance_id'] not in expected_groups('eval')['swe/public']:
            raise ValueError('instance is not in the frozen SWE inventory')
        for value in (max_request_bytes, max_output_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('publisher byte limits must be positive integers')
        self.shell_runner = _run_shell if shell_runner is None else shell_runner
        self.patch_reader = _read_patch if patch_reader is None else patch_reader
        self.testbed = TESTBED if testbed is None else Path(testbed)
        self.identity = dict(identity)
        self.auth = auth if auth is not None else ToolAuthVerifier.from_environment()
        context = self.auth.context
        if (context['operation_id'] != self.identity['operation_id']
                or context['sidecar_id'] != self.identity['child_id']):
            raise ValueError('tool authentication context differs from sidecar identity')
        self.max_request_bytes = max_request_bytes
        self.max_output_bytes = max_output_bytes
        self.command_lock = threading.Lock()
        self.exit_hook = self._exit if exit_hook is None else exit_hook
        self.shutting_down = False
        self.index_refresh = index_refresh
        self.refresh_receipt = None
        self.prepared = threading.Event()
        if index_refresh is None:
            self.prepared.set()

    def start_index_refresh(self):
        """Refresh once in the background; /ready and commands wait for it."""
        if self.prepared.is_set():
            return

        def worker():
            try:
                self.refresh_receipt = self.index_refresh()
            except Exception as exc:  # noqa: BLE001 - best effort, recorded
                self.refresh_receipt = {'refreshed': False, 'error_type': type(exc).__name__}
            finally:
                self.prepared.set()

        threading.Thread(target=worker, daemon=True).start()

    def preparation_succeeded(self):
        return self.prepared.is_set() and (self.index_refresh is None or
                                          isinstance(self.refresh_receipt, dict)
                                          and self.refresh_receipt.get('refreshed') is True)

    @staticmethod
    def _exit():
        kill_uid_processes()
        os._exit(0)

    def handle(self, method, path, body):
        if (method, path) == ('GET', '/ready'):
            prepared = self.prepared.wait(READY_WAIT_SECONDS)
            result = {'ready': self.preparation_succeeded() and self.testbed.is_dir()
                      and (self.testbed / '.git').exists()}
            if self.index_refresh is not None:
                result.update(preparation_complete=prepared, index_refresh=self.refresh_receipt)
            return 200, result
        if path == '/shutdown':
            return self.shutdown(body)
        return self.execute(body, patch=path == '/patch')

    def shutdown(self, body):
        """Confirm, then end the container: the model's processes die with it."""
        if not isinstance(body, dict) or set(body) != {'identity', 'timeout_seconds'}:
            return 400, {'error': 'invalid request fields'}
        if body['identity'] != self.identity:
            return 409, {'error': 'prepared identity mismatch'}
        if not self.shutting_down:
            self.shutting_down = True
            timer = threading.Timer(0.5, self.exit_hook)
            timer.daemon = True
            timer.start()
        return 200, {'shutdown': True}

    def execute(self, body, *, patch=False):
        expected = {'identity', 'timeout_seconds'} if patch else {'identity', 'timeout_seconds', 'command'}
        if not isinstance(body, dict) or set(body) != expected:
            return 400, {'error': 'invalid request fields'}
        if body['identity'] != self.identity:
            return 409, {'error': 'prepared identity mismatch'}
        timeout = body['timeout_seconds']
        limit = MAX_PATCH_SECONDS if patch else MAX_COMMAND_SECONDS
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not 0 < timeout <= limit):
            return 400, {'error': f'timeout must be positive and at most {limit} seconds'}
        if not patch and not isinstance(body['command'], str):
            return 400, {'error': 'command must be a string'}
        if self.shutting_down:
            return 409, {'error': 'testbed is shutting down'}
        if not self.prepared.wait(READY_WAIT_SECONDS):
            return 409, {'error': 'testbed is still preparing'}
        if not self.preparation_succeeded():
            return 503, {'error': 'testbed index preparation failed',
                         'index_refresh': self.refresh_receipt}
        if not self.command_lock.acquire(blocking=False):
            return 409, {'error': 'testbed command already running'}
        try:
            if patch:
                code, output, diagnostics = self.patch_reader(timeout, self.max_output_bytes)
                if self.refresh_receipt is not None:
                    diagnostics = {**diagnostics, 'index_refresh': self.refresh_receipt}
                return 200, {'exit_code': code, 'patch': output, 'diagnostics': diagnostics}
            code, output = self.shell_runner(body['command'], timeout, self.max_output_bytes)
            return 200, {'exit_code': code, 'output': output}
        except BaseException as error:
            reason = ('output_limit' if isinstance(error, OutputLimitExceeded)
                      else 'deadline' if isinstance(error, CaseDeadlineExceeded) else 'boundary_error')
            return (503 if isinstance(error, OutputLimitExceeded) else 422), {
                'error': 'SWE case execution failed',
                'case_failure': failure('patch' if patch else 'execute', error, reason=reason)}
        finally:
            self.command_lock.release()


def make_server(service, *, listen_profile, host, port):
    address = ipaddress.IPv4Address(host)
    if listen_profile not in {'loopback', 'bridge'}:
        raise ValueError('unknown listen profile')
    if listen_profile == 'loopback' and not address.is_loopback:
        raise ValueError('loopback profile requires a loopback IPv4 address')
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError('invalid port')

    class Handler(BaseHTTPRequestHandler):
        # Close each connection, as in the existing urllib client. Request bytes
        # are bounded separately from shell output and are never logged here.
        protocol_version = 'HTTP/1.0'

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, format, *args):
            pass

        def respond(self, status, result):
            control = {'identity': service.identity, 'error': 'SWE response failed',
                       'case_failure': {**failure('correctness', RuntimeError()), 'error_type': 'E' * 128}}
            metadata_bytes = len(json.dumps(control).encode())
            budget = service.max_request_bytes + service.max_output_bytes + metadata_bytes
            try:
                encoded = json.dumps({'identity': service.identity, **result}, allow_nan=False).encode()
                # The fixed request allocation funds identity/protocol metadata;
                # the separate output allocation funds command/test output.
                if len(encoded) > budget:
                    raise OutputLimitExceeded('sidecar response exceeds funded transport buffers')
            except BaseException as error:
                status = 422
                encoded = json.dumps({'identity': service.identity, 'error': 'SWE response failed',
                    'case_failure': failure('serialize', error,
                        reason='output_limit' if isinstance(error, OutputLimitExceeded)
                        else 'boundary_error')}, allow_nan=False).encode()
                if len(encoded) > budget:
                    self.close_connection = True
                    return
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def authenticate(self, method, body):
            try:
                service.auth.verify(self.headers, method, self.path, body)
                return True
            except ToolAuthError:
                self.respond(401, {'error': 'tool request authentication failed'})
                return False

        def dispatch(self, method, body):
            try:
                status, result = service.handle(method, self.path, body)
            except BaseException as error:
                status, result = 422, {'error': 'SWE request failed',
                    'case_failure': failure('correctness' if self.path == '/correctness'
                                            else 'patch' if self.path == '/patch' else 'execute', error)}
            self.respond(status, result)

        def do_GET(self):
            if ('GET', self.path) not in service.routes:
                self.respond(404, {'error': 'unknown sidecar route'})
                return
            if not self.authenticate('GET', b''):
                return
            self.dispatch('GET', None)

        def do_POST(self):
            if ('POST', self.path) not in service.routes:
                self.respond(404, {'error': 'unknown sidecar route'})
                return
            lengths = self.headers.get_all('Content-Length', [])
            if self.headers.get('Transfer-Encoding') or len(lengths) != 1:
                self.respond(400, {'error': 'one Content-Length is required'})
                return
            try:
                length = int(lengths[0])
            except ValueError:
                self.respond(400, {'error': 'invalid Content-Length'})
                return
            if length <= 0 or length > service.max_request_bytes:
                self.respond(413, {'error': 'request exceeds publisher limit or is empty'})
                return
            try:
                data = self.rfile.read(length)
                if len(data) != length:
                    raise ValueError('incomplete body')
                if not self.authenticate('POST', data):
                    return
                body = json.loads(data)
            except (ValueError, UnicodeError, TimeoutError):
                self.respond(400, {'error': 'invalid JSON body'})
                return
            self.dispatch('POST', body)

    return ThreadingHTTPServer((str(address), port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    identity_source = parser.add_mutually_exclusive_group(required=True)
    identity_source.add_argument('--identity-file', type=Path)
    identity_source.add_argument('--driver-identity-file', type=Path, help='New public identity file generated from Driver env')
    parser.add_argument('--instance-id', help='Publisher-fixed instance for Driver bootstrap')
    parser.add_argument('--listen-profile', required=True, choices=['loopback', 'bridge'])
    parser.add_argument('--host', required=True, help='Publisher-selected IPv4 address')
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--max-request-bytes', required=True, type=int)
    parser.add_argument('--max-output-bytes', required=True, type=int)
    args = parser.parse_args()
    if args.driver_identity_file is not None:
        if not args.instance_id:
            parser.error('Driver bootstrap requires the publisher-fixed instance')
        driver_identity = sidecar_env.install('verifier')
        identity = bootstrap_identity(args.instance_id, args.driver_identity_file, driver_identity)
    else:
        if args.instance_id is not None:
            parser.error('instance is contained in the supplied identity file')
        identity = json.loads(args.identity_file.read_text())
    service = ToolService(
        identity,
        max_request_bytes=args.max_request_bytes, max_output_bytes=args.max_output_bytes,
        auth=ToolAuthVerifier.from_environment(), index_refresh=_refresh_index,
    )
    sidecar_env.scrub()
    service.start_index_refresh()
    with make_server(service, listen_profile=args.listen_profile, host=args.host, port=args.port) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
