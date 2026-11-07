"""Real localhost/shell/Git checks against the existing prepared-testbed client."""
import copy
import json
import os
import shlex
import signal
import sys
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

import shutil

import batch_coordinator as coordinator
from batch_coordinator import BackendError, PreparedGroup, PreparedTestbed, TestbedClient
from serving_eval import sidecar_env
from serving_eval import swe_tool_service as service
from serving_eval.tool_auth import ToolAuthError, ToolAuthSigner, ToolAuthVerifier, _key_id
from serving_eval.swe_tool_exec import clean_environment


class ToolServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cwd_patch = patch.object(service, 'TESTBED', self.root)
        self.cwd_patch.start()
        self.addCleanup(self.cwd_patch.stop)
        self.launch_patch = patch.object(service, '_command_argv', lambda command: ['bash', '-lc', command])
        self.launch_patch.start()
        self.private_dirs = patch.object(service, "_ensure_private_runtime_dirs", lambda: None)
        self.private_dirs.start()
        self.addCleanup(self.private_dirs.stop)
        self.addCleanup(self.launch_patch.stop)
        for command in (
            ['git', 'init', '-q'],
            ['git', 'config', 'user.name', 'Fixture'],
            ['git', 'config', 'user.email', 'fixture@example.invalid'],
        ):
            subprocess.run(command, cwd=self.root, check=True, capture_output=True)
        (self.root / 'tracked.txt').write_text('original\n')
        subprocess.run(['git', 'add', '.'], cwd=self.root, check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-qm', 'fixture'], cwd=self.root, check=True, capture_output=True)
        group = PreparedGroup('op', 'group', 'placement', 'train', 'swe', 'public',
                              'http://127.0.0.1:1', 'sha256:model')
        descriptor = PreparedTestbed('sympy__sympy-21847', 'http://127.0.0.1:1', 'child', 'train')
        private = Ed25519PrivateKey.generate()
        private_text = private.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()).decode()
        public_text = private.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode()
        key_id = _key_id(public_text)
        scheduler_context = {'schema_version': 'frontier.tool-auth-context/v1', 'run_id': 'run',
            'attempt_id': 'attempt', 'operation_id': 'op', 'sidecar_id': 'scheduler'}
        verifier_context = {**scheduler_context, 'sidecar_id': 'child'}
        self.signer = ToolAuthSigner(private_text, public_text, key_id, scheduler_context)
        self.verifier = ToolAuthVerifier(public_text, key_id, verifier_context)
        self.client = TestbedClient(group, descriptor, [], signer=self.signer)
        self.exits = []
        self.service = service.ToolService(self.client.identity, max_request_bytes=16384,
                                           max_output_bytes=65536, auth=self.verifier,
                                           exit_hook=lambda: self.exits.append('exit'))
        self.server = service.make_server(self.service, listen_profile='loopback', host='127.0.0.1', port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01})
        self.thread.start()
        self.addCleanup(self.close_server)
        self.client.endpoint = 'http://127.0.0.1:' + str(self.server.server_port)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, route, body, *, headers=None, raw=None):
        data = json.dumps(body).encode() if raw is None else raw
        signed = self.signer.headers(self.client.identity['child_id'], 'POST', route, data)
        signed['Content-Type'] = 'application/json'
        if headers:
            signed.update(headers)
        request = Request(self.client.endpoint + route, data=data, headers=signed)
        try:
            response = build_opener(ProxyHandler({})).open(request, timeout=2)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, json.loads(response.read())

    def body(self, command='true', timeout=1):
        return {'identity': copy.deepcopy(self.client.identity), 'command': command, 'timeout_seconds': timeout}


    def test_auth_binds_target_method_path_body_time_and_nonce(self):
        body = self.body('touch authenticated')
        data = json.dumps(body).encode()
        headers = self.signer.headers(self.client.identity['child_id'], 'POST', '/execute', data)
        request = Request(self.client.endpoint + '/execute', data=data, headers={**headers, 'Content-Type': 'application/json'})
        with build_opener(ProxyHandler({})).open(request, timeout=2) as response:
            self.assertEqual(response.status, 200)
        self.assertTrue((self.root / 'authenticated').exists())
        replay = Request(self.client.endpoint + '/execute', data=data, headers={**headers, 'Content-Type': 'application/json'})
        with self.assertRaises(HTTPError) as caught:
            build_opener(ProxyHandler({})).open(replay, timeout=2)
        self.assertEqual(caught.exception.code, 401)
        for route, changed in (('/patch', data), ('/execute', data + b' ')):
            fresh = self.signer.headers(self.client.identity['child_id'], 'POST', '/execute', data)
            request = Request(self.client.endpoint + route, data=changed,
                              headers={**fresh, 'Content-Type': 'application/json'})
            with self.subTest(route=route, changed=changed != data), self.assertRaises(HTTPError) as caught:
                build_opener(ProxyHandler({})).open(request, timeout=2)
            self.assertEqual(caught.exception.code, 401)

        private = Ed25519PrivateKey.generate()
        private_text = private.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()).decode()
        public_text = private.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode()
        key_id = _key_id(public_text)
        context = {'schema_version': 'frontier.tool-auth-context/v1', 'run_id': 'run', 'attempt_id': 'attempt',
                   'operation_id': 'op', 'sidecar_id': 'scheduler'}
        stale = ToolAuthSigner(private_text, public_text, key_id, context, now=lambda: 100)
        verifier = ToolAuthVerifier(public_text, key_id, {**context, 'sidecar_id': 'child'}, now=lambda: 131)
        with self.assertRaisesRegex(ToolAuthError, 'expired'):
            verifier.verify(stale.headers('child', 'POST', '/execute', data), 'POST', '/execute', data)
        wrong_target = ToolAuthVerifier(public_text, key_id, {**context, 'sidecar_id': 'other'}, now=lambda: 100)
        with self.assertRaisesRegex(ToolAuthError, 'signature'):
            wrong_target.verify(stale.headers('child', 'POST', '/execute', data), 'POST', '/execute', data)
        wrong_field = {**context, 'schema': context['schema_version']}
        del wrong_field['schema_version']
        with self.assertRaisesRegex(ToolAuthError, 'context'):
            ToolAuthSigner(private_text, public_text, key_id, wrong_field)


    def test_command_environment_drops_auth_and_python_injection(self):
        source = {
            'PATH': '/opt/miniconda3/bin:/usr/bin',
            'TZ': 'Etc/UTC',
            'FRONTIER_TOOL_AUTH_PUBLIC_KEY_OPENSSH': 'public',
            'FRONTIER_TOOL_AUTH_CONTEXT_JSON': 'context',
            'PYTHONPATH': '/testbed/attacker',
            'LD_PRELOAD': '/testbed/attacker.so',
            'BASH_ENV': '/testbed/attacker',
        }
        cleaned = clean_environment(source)
        self.assertEqual(cleaned['PATH'], source['PATH'])
        self.assertEqual(cleaned['HOME'], '/home/frontier-tool')
        self.assertEqual(cleaned['TMPDIR'], '/tmp/frontier-tool')
        self.assertNotIn('PYTHONPATH', cleaned)
        self.assertNotIn('LD_PRELOAD', cleaned)
        self.assertNotIn('BASH_ENV', cleaned)
        self.assertFalse(any(key.startswith('FRONTIER_TOOL_AUTH_') for key in cleaned))

    def test_ready_and_original_output_exit_and_newline_semantics(self):
        self.client.ready()
        result = self.client.run("printf 'err\\r\\n' >&2; printf 'out\\r\\n'; exit 7", timeout=1)
        self.assertEqual(result, (7, 'out\nerr\n'))
        self.assertEqual(self.client.trace[-1]['response']['identity'], self.client.identity)

    def test_fresh_shell_fixed_cwd_and_persistent_files(self):
        code, _ = self.client.run('export TOOL_SERVICE_TEST_VALUE=yes; touch persisted; cd /', timeout=1)
        self.assertEqual(code, 0)
        code, output = self.client.run('pwd; printf "%s\\n" "${TOOL_SERVICE_TEST_VALUE-unset}"; test -f persisted', timeout=1)
        self.assertEqual((code, output), (0, str(self.root) + '\nunset\n'))

    def test_background_http_service_survives_between_commands(self):
        (self.root / 'served.txt').write_text('background server persists')
        (self.root / 'background.py').write_text(
            "from http.server import HTTPServer, SimpleHTTPRequestHandler\n"
            "from pathlib import Path\n"
            "server = HTTPServer(('127.0.0.1', 0), SimpleHTTPRequestHandler)\n"
            "Path('server.port').write_text(str(server.server_port))\n"
            "server.serve_forever()\n"
        )
        python = shlex.quote(sys.executable)
        code, output = self.client.run(
            python + ' background.py </dev/null >server.log 2>&1 & printf "%s" "$!" >server.pid', timeout=1)
        self.assertEqual((code, output), (0, ''))
        pid = int((self.root / 'server.pid').read_text())

        def cleanup():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.addCleanup(cleanup)
        probe = (
            "from pathlib import Path; import time; "
            "from urllib.request import build_opener, ProxyHandler; "
            "deadline = time.monotonic() + 2\n"
            "while not Path('server.port').exists() and time.monotonic() < deadline: time.sleep(.01)\n"
            "port = Path('server.port').read_text(); "
            "print(build_opener(ProxyHandler({})).open('http://127.0.0.1:' + port + '/served.txt').read().decode())"
        )
        self.assertEqual(self.client.run(python + ' -c ' + shlex.quote(probe), timeout=3),
                         (0, 'background server persists\n'))

    def test_complete_output_remains_for_conversation_loop_to_clip(self):
        self.assertEqual(self.client.run("printf '%05000d' 1", timeout=1), (0, '0' * 4999 + '1'))

    def test_patch_stages_tracked_and_new_files(self):
        self.client.run("printf 'changed\\n' > tracked.txt; printf 'new\\n' > added.txt", timeout=1)
        index_before = (self.root / '.git' / 'index').read_bytes()
        result = self.client.read_patch()
        self.assertIn('+changed', result)
        self.assertIn('+new', result)
        self.assertEqual((self.root / '.git' / 'index').read_bytes(), index_before)

    def test_patch_treats_names_literally_and_does_not_readd_head_paths(self):
        (self.root / "literal[1]").write_text("new\n")
        (self.root / "removed-from-index").write_text("tracked\n")
        subprocess.run(["git", "add", "removed-from-index"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "edge paths"], cwd=self.root, check=True)
        subprocess.run(["git", "rm", "--cached", "removed-from-index"], cwd=self.root,
                       check=True, capture_output=True)
        (self.root / "removed-from-index").write_text("changed\n")

        result = self.client.read_patch()

        self.assertIn("diff --git a/literal[1] b/literal[1]", result)
        self.assertEqual(result.count("diff --git a/removed-from-index b/removed-from-index"), 1)
        removed = result.split(
            "diff --git a/removed-from-index b/removed-from-index", 1)[1].split(
                "diff --git ", 1)[0]
        self.assertNotIn("deleted file mode", removed)
        self.assertNotIn("new file mode", removed)
        self.assertIn("+changed", removed)
        with tempfile.TemporaryDirectory() as target:
            subprocess.run(["git", "clone", "-q", str(self.root), target], check=True)
            subprocess.run(["git", "apply", "--check", "-"], cwd=target, input=result,
                           text=True, check=True, capture_output=True)

    def test_patch_preserves_index_deletions_symlinks_and_ignores(self):
        (self.root / "deleted.txt").write_text("delete\n")
        subprocess.run(["git", "add", "deleted.txt"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "deleted"], cwd=self.root, check=True)
        (self.root / "deleted.txt").unlink()
        (self.root / "staged.txt").write_text("staged\n")
        subprocess.run(["git", "add", "staged.txt"], cwd=self.root, check=True)
        (self.root / "staged.txt").write_text("staged\nunstaged\n")
        (self.root / "added-link").symlink_to("tracked.txt")
        (self.root / "added-binary").write_bytes(b"\x00\x01\x02")
        (self.root / ".gitignore").write_text("ignored\n")
        (self.root / "ignored").write_text("ignored\n")

        result = self.client.read_patch()

        self.assertIn("deleted file mode", result)
        self.assertIn("new file mode 120000", result)
        self.assertIn("Binary files /dev/null and b/added-binary differ", result)
        self.assertIn("+unstaged", result)
        self.assertNotIn("diff --git a/ignored b/ignored", result)
        staged = subprocess.run(["git", "show", ":staged.txt"], cwd=self.root,
                                check=True, capture_output=True, text=True).stdout
        self.assertEqual(staged, "staged\n")

    def test_patch_failure_is_not_empty_success(self):
        (self.root / '.git' / 'index.lock').touch()
        with self.assertRaisesRegex(BackendError, 'patch extraction failed'):
            self.client.read_patch()

    def test_timeout_cleans_ordinary_descendants_and_keeps_transport_grace(self):
        with patch('batch_coordinator.TOOL_TRANSPORT_GRACE_SECONDS', 0.5):
            result = self.client.run('(sleep 0.3; touch escaped-marker) & wait', timeout=0.05)
        self.assertEqual(result, (124, 'command timed out'))
        time.sleep(0.4)
        self.assertFalse((self.root / 'escaped-marker').exists())
        self.assertEqual(self.client.run('printf recovered', timeout=1), (0, 'recovered'))

    def test_identity_and_request_shape_validation(self):
        body = self.body('touch wrong-identity')
        body['identity']['instance_id'] = 'wrong'
        self.assertEqual(self.request('/execute', body)[0], 409)
        self.assertFalse((self.root / 'wrong-identity').exists())
        for key, value in [('timeout_seconds', True), ('timeout_seconds', 61),
                           ('timeout_seconds', 0), ('timeout_seconds', float('nan')),
                           ('timeout_seconds', 10 ** 400),
                           ('command', 42), ('cwd', '/')]:
            body = self.body()
            body[key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(self.request('/execute', body)[0], 400)

    def test_signed_shutdown_confirms_then_ends_the_container_and_refuses_new_commands(self):
        self.assertEqual(self.exits, [])
        self.client.shutdown()
        self.assertEqual(self.request('/execute', self.body('touch after-shutdown'))[0], 409)
        self.assertFalse((self.root / 'after-shutdown').exists())
        self.client.shutdown()  # idempotent while the exit timer runs
        time.sleep(0.8)
        self.assertEqual(self.exits, ['exit'])
        body = {'identity': dict(self.client.identity, instance_id='other'), 'timeout_seconds': 5}
        self.assertEqual(self.request('/shutdown', body)[0], 409)

    def test_no_correctness_or_lifecycle_routes(self):
        for route in ('/correctness', '/score', '/prepare', '/execute?instance=other'):
            with self.subTest(route=route):
                self.assertEqual(self.request(route, self.body())[0], 404)
        self.assertFalse(hasattr(self.client, 'run_correctness'))
        with self.assertRaises(BackendError):
            self.client._call('/correctness', {'timeout_seconds': 1})

    def test_publisher_bounds_fail_visibly_and_release_command_lock(self):
        self.service.max_output_bytes = 32
        with self.assertRaises(BackendError):
            self.client.run("printf '%0100d' 1", timeout=1)
        self.assertEqual(self.client.run('printf ok', timeout=1), (0, 'ok'))
        self.service.max_request_bytes = 16
        self.assertEqual(self.request('/execute', self.body())[0], 413)

    def test_concurrent_mutation_is_rejected(self):
        with self.service.command_lock:
            self.assertEqual(self.request('/execute', self.body('touch concurrent'))[0], 409)
            self.assertEqual(self.request('/patch', {'identity': self.client.identity, 'timeout_seconds': 1})[0], 409)
        self.assertFalse((self.root / 'concurrent').exists())

    def test_driver_declared_identity_maps_to_the_prepared_testbed_identity(self):
        public = {'run_id': 'run', 'attempt_id': 'attempt', 'operation_id': 'runner',
                  'phase': 'eval', 'name': 'swe-012-candidate', 'profile_digest': 'sha256:' + '1' * 64}
        destination = self.root / 'driver-identity.json'
        with patch.dict(os.environ, {sidecar_env.IDENTITY_ENV: json.dumps(public)}):
            identity = service.bootstrap_identity(self.client.identity['instance_id'], destination)
            self.assertEqual(identity, {'operation_id': 'runner', 'group_id': 'runner', 'placement_id': 'runner',
                                        'instance_id': self.client.identity['instance_id'],
                                        'child_id': 'swe-012-candidate', 'phase': 'eval'})
            self.assertEqual(set(identity), service.IDENTITY_FIELDS)
            self.assertEqual(json.loads(destination.read_text()), identity)
            with self.assertRaises(FileExistsError):
                service.bootstrap_identity(self.client.identity['instance_id'], destination)
        for key, value in (('token', 'rejected'), ('provider_id', 'rejected'), ('phase', 'agent')):
            with patch.dict(os.environ, {sidecar_env.IDENTITY_ENV: json.dumps({**public, key: value})}):
                with self.assertRaises(ValueError):
                    service.bootstrap_identity(self.client.identity['instance_id'], self.root / key)
            self.assertFalse((self.root / key).exists())
        with patch.dict(os.environ, {sidecar_env.IDENTITY_ENV: json.dumps(public)}):
            with self.assertRaises(ValueError):
                service.bootstrap_identity('unknown__instance-1', self.root / 'unknown')

    def test_fixed_startup_identity_and_explicit_listen_profile(self):
        invalid = dict(self.client.identity, instance_id='unknown')
        with self.assertRaises(ValueError):
            service.ToolService(invalid, max_request_bytes=1, max_output_bytes=1)
        with self.assertRaises(ValueError):
            service.ToolService(dict(self.client.identity, token='not-permitted'), max_request_bytes=1, max_output_bytes=1)
        with self.assertRaises(ValueError):
            service.make_server(self.service, listen_profile='loopback', host='0.0.0.0', port=0)
        with service.make_server(self.service, listen_profile='bridge', host='0.0.0.0', port=0):
            pass

    def test_patch_bound_is_separate_from_the_original_60_second_command_bound(self):
        self.assertEqual(service.MAX_COMMAND_SECONDS, 60)
        self.assertEqual(service.MAX_PATCH_SECONDS, coordinator.PATCH_TIMEOUT_SECONDS)
        self.assertLess(service.INDEX_REFRESH_SECONDS, coordinator.TESTBED_READY_SECONDS)
        self.assertLess(service.READY_WAIT_SECONDS, 60)  # coordinator /ready transport timeout
        identity = self.client.identity
        self.assertEqual(self.request('/patch', {'identity': identity, 'timeout_seconds': 180})[0], 200)
        self.assertEqual(self.request('/patch', {'identity': identity, 'timeout_seconds': 181})[0], 400)
        self.assertEqual(self.request('/execute', self.body(timeout=61))[0], 400)
        self.client.read_patch()
        self.assertEqual(self.client.trace[-1]['request'], {'timeout_seconds': 180})

    def test_one_timed_out_diff_is_retried_once_and_other_failures_are_not(self):
        calls = []

        def reader(outcomes):
            def read(timeout, max_output_bytes):
                calls.append(timeout)
                code, stage = outcomes[len(calls) - 1]
                return code, 'command timed out' if code else 'diff\n', {'stage': stage}
            return read

        self.service.patch_reader = reader([(124, 'git_diff'), (0, 'complete')])
        self.assertEqual(self.client.read_patch(), 'diff\n')
        self.assertEqual(calls, [180, 180])
        self.assertEqual([event['path'] for event in self.client.trace[-2:]], ['/patch', '/patch'])
        self.assertTrue(all('error_origin' not in event for event in self.client.trace))
        for outcomes, expected in (([(124, 'git_diff'), (124, 'git_diff')], 2),
                                   ([(1, 'git_index_lock')], 1), ([(1, 'git_diff')], 1)):
            calls.clear()
            self.service.patch_reader = reader(outcomes)
            with self.subTest(outcomes=outcomes), self.assertRaisesRegex(BackendError, 'patch extraction failed'):
                self.client.read_patch()
            self.assertEqual(len(calls), expected)

    def test_patch_retry_is_idempotent_on_a_real_repository(self):
        self.client.run("printf 'changed\\n' > tracked.txt; printf 'new\\n' > added.txt", timeout=1)
        index = (self.root / '.git/index').read_bytes()
        first = service._read_patch(5, 65536)
        second = service._read_patch(5, 65536)
        self.assertEqual(first[:2], second[:2])
        self.assertEqual(first[0], 0)
        self.assertEqual((self.root / '.git/index').read_bytes(), index)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()),
                         ['.git', 'added.txt', 'tracked.txt'])

    def test_index_refresh_removes_stale_stat_work_and_keeps_the_patch_identical(self):
        # A container testbed is a layer extraction: same content and mtime,
        # new inode/ctime, so every index entry is stat-dirty until refreshed.
        for index in range(40):
            directory = self.root / f'pkg{index % 4}'
            directory.mkdir(exist_ok=True)
            (directory / f'module{index}.py').write_text(f'value = {index}\n')
        subprocess.run(['git', 'add', '.'], cwd=self.root, check=True)
        subprocess.run(['git', 'commit', '-qm', 'tree'], cwd=self.root, check=True)
        (self.root / 'pkg1/module1.py').write_text('value = -1\n')
        (self.root / 'pkg2/module2.py').unlink()
        (self.root / 'pkg3/new.py').write_text('new = True\n')
        extracted = Path(self.temp.name + '-extracted')
        shutil.copytree(self.root, extracted, symlinks=True)
        self.addCleanup(shutil.rmtree, extracted, True)

        def git(*args):
            return subprocess.run(['git', *args], cwd=extracted, check=True,
                                  capture_output=True, text=True).stdout

        with patch.object(service, 'TESTBED', extracted):
            stale = git('diff-files', '--name-only').split()
            self.assertGreaterEqual(len(stale), 41)
            staged = git('ls-files', '-s')
            before = service._read_patch(5, 1 << 20)
            receipt = service._refresh_index(testbed=extracted, timeout=10)
            after = service._read_patch(5, 1 << 20)
        self.assertTrue(receipt['refreshed'], receipt)
        self.assertEqual(git('diff-files', '--name-only').split(), ['pkg1/module1.py', 'pkg2/module2.py'])
        self.assertEqual(git('ls-files', '-s'), staged)
        self.assertEqual(before[0], 0)
        self.assertEqual(before[:2], after[:2])
        for marker in ('+value = -1', 'deleted file mode', '+new = True'):
            self.assertIn(marker, after[1])
        self.assertFalse((extracted / '.git/index.lock').exists())

    def test_index_refresh_is_bounded_and_never_leaves_or_steals_a_lock(self):
        lock = self.root / '.git/index.lock'

        def killed(command, timeout, max_output_bytes):
            self.assertEqual(timeout, service.INDEX_REFRESH_SECONDS)
            lock.touch()  # a SIGKILLed git cannot remove its own lock
            return 124, 'command timed out'

        receipt = service._refresh_index(runner=killed, testbed=self.root)
        self.assertEqual((receipt['refreshed'], receipt['exit_code']), (False, 124))
        self.assertFalse(lock.exists())
        lock.touch()
        receipt = service._refresh_index(runner=killed, testbed=self.root)
        self.assertEqual(receipt['reason'], 'index_lock_present')
        self.assertTrue(lock.exists())

    def test_ready_and_commands_wait_for_the_startup_refresh(self):
        gate = threading.Event()

        def refresh():
            gate.wait(5)
            return {'refreshed': True, 'exit_code': 0, 'seconds': 0.0}

        prepared = service.ToolService(self.client.identity, max_request_bytes=16384,
                                       max_output_bytes=65536, auth=self.verifier,
                                       index_refresh=refresh, exit_hook=lambda: None)
        prepared.start_index_refresh()
        with patch.object(service, 'READY_WAIT_SECONDS', 0.05):
            status, result = prepared.handle('GET', '/ready', None)
            self.assertEqual((status, result['ready'], result['preparation_complete']),
                             (200, False, False))
            self.assertEqual(prepared.handle('POST', '/execute', self.body('touch early'))[0], 409)
        self.assertFalse((self.root / 'early').exists())
        results = []
        waiter = threading.Thread(target=lambda: results.append(
            prepared.handle('POST', '/execute', self.body('printf waited'))))
        waiter.start()
        time.sleep(0.1)
        self.assertEqual(results, [])
        gate.set()
        waiter.join(5)
        self.assertEqual(results, [(200, {'exit_code': 0, 'output': 'waited'})])
        status, result = prepared.handle('GET', '/ready', None)
        self.assertEqual((status, result['ready'], result['preparation_complete']),
                         (200, True, True))
        status, result = prepared.handle('POST', '/patch', {'identity': self.client.identity,
                                                            'timeout_seconds': 5})
        self.assertEqual((status, result['exit_code']), (200, 0))
        self.assertEqual(result['diagnostics']['index_refresh']['refreshed'], True)

    def test_failed_or_invalid_refresh_never_releases_commands_or_ready(self):
        for receipt in ({'refreshed': False, 'exit_code': 124}, None):
            with self.subTest(receipt=receipt):
                prepared = service.ToolService(
                    self.client.identity, max_request_bytes=16384,
                    max_output_bytes=65536, auth=self.verifier,
                    index_refresh=lambda: receipt,
                    shell_runner=lambda *args: self.fail('command ran before preparation'),
                    patch_reader=lambda *args: self.fail('patch read before preparation'))
                prepared.start_index_refresh()
                self.assertTrue(prepared.prepared.wait(2))
                status, result = prepared.handle('GET', '/ready', None)
                self.assertEqual((status, result['ready'], result['preparation_complete']),
                                 (200, False, True))
                self.assertEqual(prepared.execute(self.body())[0], 503)
                self.assertEqual(prepared.execute({'identity': self.client.identity,
                                                  'timeout_seconds': 1}, patch=True)[0], 503)

    def test_readiness_client_waits_outside_episode_and_stops_on_prepare_failure(self):
        with patch.object(self.client, '_call', side_effect=[
                {'ready': False, 'preparation_complete': False}, {'ready': True}]) as call:
            with patch.object(coordinator.time, 'sleep'):
                self.client.ready()
            self.assertEqual(call.call_count, 2)
        with patch.object(self.client, '_call', return_value={
                'ready': False, 'preparation_complete': True}):
            with self.assertRaisesRegex(BackendError, 'index preparation failed'):
                self.client.ready()
        with patch.object(self.client, '_call', return_value={
                'ready': False, 'preparation_complete': False}):
            with patch.object(coordinator.time, 'monotonic', side_effect=[0, 661]):
                with self.assertRaisesRegex(BackendError, 'not ready'):
                    self.client.ready()


if __name__ == '__main__':
    unittest.main()
