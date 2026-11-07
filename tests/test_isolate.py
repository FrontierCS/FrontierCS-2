"""The shared isolate launcher (tasks/shared/lib/isolate.py): escape tests and limits.

Each escape test plays the submitted code: it runs inside an instance and tries to reach
something the judge holds. Every test runs once per backend this host can build (the classes
end in _landlock / _userns) and is skipped for the others. landlock needs Linux x86_64 with
Landlock ABI >= 6 (the EKS nodes run 6.12); to run it as the EKS Verifier pod runs (uid 65532,
every capability dropped, no_new_privs, read-only root, the runtime's default seccomp profile):

  docker run --rm --user 65532:65532 --cap-drop ALL --security-opt no-new-privileges \
    --read-only --tmpfs /tmp:exec -v "$PWD:/src:ro" -w /src \
    <frontier-verifier-base> python3 -m unittest tests.test_isolate

userns needs unprivileged user namespaces (Verifier GPU runners: Modal, gVisor). A local
approximation adds `--security-opt seccomp=unconfined --security-opt apparmor=unconfined` to
the command above; the real check runs on a runner (scripts/maintenance/isolate_runner.py).
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import dataclasses
import glob
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks/shared/lib"))
import isolate  # noqa: E402
from isolate_layers import Connect, KeptFile, Pinned, Recording, Undeclared  # noqa: E402

PYTHON = sys.executable


class Backend:
    backend = "auto"   # set per generated class

    def python(self, code: str, **kwargs) -> isolate.Result:
        kwargs.setdefault("timeout_seconds", 30)
        return self.run_([PYTHON, "-I", "-S", "-c", code], **kwargs)

    def setUp(self):
        # this class's backend only (the launcher otherwise takes the first this host can build)
        only = mock.patch.object(isolate, "BACKENDS", (self.backend,))
        only.start()
        self.addCleanup(only.stop)

    def run_(self, argv, **kwargs) -> isolate.Result:
        kwargs.setdefault("timeout_seconds", 30)
        result = isolate.run(argv, **kwargs)
        self.assertEqual(result.backend, self.backend)
        return result

    def start_(self, argv, **kwargs) -> isolate.Instance:
        kwargs.setdefault("timeout_seconds", 30)
        return isolate.start(argv, **kwargs)


class EscapeTests(Backend):
    def setUp(self):
        super().setUp()
        self.judge = Path(tempfile.mkdtemp(prefix="judge-state-"))
        self.secret = self.judge / "reference.patch"
        self.secret.write_text("the reference answer")

    def tearDown(self):
        isolate._remove(self.judge)

    def test_self_test_passes(self):
        report = isolate._probe(self.backend)
        self.assertTrue(all(report.values()), report)

    def test_cannot_read_judge_files(self):
        result = self.python(f"""
import os
for action in (lambda: open({str(self.secret)!r}).read(), lambda: os.listdir({str(self.judge)!r}),
               lambda: open('/proc/self/root' + {str(self.secret)!r}).read()):
    try:
        print('LEAK', action())
    except OSError as error:
        print('denied', error.errno)
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(b"LEAK", result.stdout)
        self.assertEqual(result.stdout.count(b"denied"), 3)

    def test_symlink_in_tree_does_not_reach_the_judge(self):
        tree = Path(tempfile.mkdtemp(prefix="tree-"))
        (tree / "answers").symlink_to(self.secret)
        try:
            result = self.python("""
try:
    print('LEAK', open('answers').read())
except OSError:
    print('denied')
""", tree=tree)
            self.assertEqual(result.stdout.strip(), b"denied", result.stderr)
            with self.assertRaises(isolate.IsolateError):
                self.python("pass", read_paths=[tree / "answers"])
        finally:
            isolate._remove(tree)

    def test_read_paths_are_read_only_and_bounded(self):
        shared = self.judge / "public"
        shared.mkdir()
        (shared / "graph.txt").write_text("1 2")
        result = self.python(f"""
print(open({str(shared / 'graph.txt')!r}).read())
for action in (lambda: open({str(shared / 'graph.txt')!r}, 'a').write('x'),
               lambda: open({str(self.secret)!r}).read()):
    try:
        action(); print('LEAK')
    except OSError:
        print('denied')
""", read_paths=[shared])
        self.assertEqual(result.stdout.split(b"\n")[:3], [b"1 2", b"denied", b"denied"],
                         result.stderr)
        for bad in ("/", "/tmp", "/opt", "relative", "/opt/frontier", str(self.judge / "absent")):
            with self.subTest(bad=bad), self.assertRaises(isolate.IsolateError):
                self.python("pass", read_paths=[bad])

    def test_cannot_write_outside_private_dirs(self):
        result = self.python(f"""
for path in ({str(self.judge / 'planted')!r}, '/tmp/planted', '/dev/shm/planted', '/var/tmp/x'):
    try:
        open(path, 'w').write('x'); print('WROTE', path)
    except OSError:
        print('denied')
""")
        self.assertNotIn(b"WROTE", result.stdout, result.stdout)
        self.assertFalse((self.judge / "planted").exists())

    def test_environment_is_rebuilt(self):
        os.environ["FRONTIER_VERIFIER_TOKEN"] = "driver-capability"
        os.environ["FRONTIER_SUBMISSION_ROLE"] = "final"
        try:
            result = self.python("import os, json; print(json.dumps(dict(os.environ)))",
                            env={"OMP_NUM_THREADS": "2"})
        finally:
            del os.environ["FRONTIER_VERIFIER_TOKEN"], os.environ["FRONTIER_SUBMISSION_ROLE"]
        seen = json.loads(result.stdout)
        self.assertEqual(set(seen) - {"PWD"},
                         {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "OMP_NUM_THREADS"})
        self.assertNotIn(b"driver-capability", result.stdout)

    def test_cannot_read_judge_environment_or_memory(self):
        result = self.python(f"""
import ctypes, os
judge = {os.getpid()}
# (userns: the instance has its own PID namespace, where the judge's pid is another process)
def is_judge():
    try:
        return open('/proc/%d/cmdline' % judge, 'rb').read() == {Path('/proc/self/cmdline').read_bytes()!r}
    except OSError:
        return False
for path in ('/proc/%d/environ' % judge, '/proc/%d/mem' % judge, '/proc/%d/cwd/x' % judge):
    try:
        open(path, 'rb').read(16); print('LEAK' if is_judge() else 'other', path)
    except OSError:
        print('denied')
libc = ctypes.CDLL(None, use_errno=True)
print('ptrace', libc.ptrace(16, judge, 0, 0), ctypes.get_errno())   # PTRACE_ATTACH
try:
    os.kill(judge, 0); print('LEAK signal' if is_judge() else 'other')
except OSError:
    print('denied signal')
""")
        self.assertNotIn(b"LEAK", result.stdout, result.stdout)
        self.assertIn(b"ptrace -1", result.stdout)

    def test_nothing_persists_across_runs(self):
        first = self.python("""
import os, subprocess, sys
open(os.path.join(os.environ['TMPDIR'], 'cache'), 'w').write('answers')
open(os.path.join(os.environ['HOME'], '.cache'), 'w').write('answers')
daemon = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600.4242)'],
                          start_new_session=True)
""")
        self.assertEqual(first.returncode, 0, first.stderr)
        time.sleep(0.2)
        survivors = []
        for entry in os.listdir("/proc"):   # by command line: the instance may have its own pids
            try:
                if b"600.4242" in Path(f"/proc/{entry}/cmdline").read_bytes() and \
                        Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
                    survivors.append(entry)
            except (OSError, IndexError):
                continue
        self.assertEqual(survivors, [], "a background process outlived its instance")
        second = self.python("""
import glob, os
print(sorted(glob.glob('/tmp/ts-*/r/*/*')),
      os.listdir(os.environ['TMPDIR']), os.listdir(os.environ['HOME']))
""")
        self.assertEqual(second.stdout.strip(), b"[] [] []", second.stderr)

    def test_sysv_and_posix_ipc_denied(self):
        result = self.python("""
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
print(libc.shmget(0x1234, 4096, 0o1666), libc.semget(0x1234, 1, 0o1666),
      libc.msgget(0x1234, 0o1666))
""")
        self.assertEqual(result.stdout.split(), [b"-1", b"-1", b"-1"], result.stderr)

    def test_no_network(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        unix_dir = Path(tempfile.mkdtemp(prefix="judge-socket-"))
        unix_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix_listener.bind(str(unix_dir / "validator.sock"))
        unix_listener.listen(1)
        abstract = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        abstract.bind(f"\0judge-{os.getpid()}")
        abstract.listen(1)
        try:
            code = f"""
import socket
attempts = [(socket.AF_INET, socket.SOCK_STREAM, ('127.0.0.1', {listener.getsockname()[1]})),
            (socket.AF_INET, socket.SOCK_DGRAM, ('1.1.1.1', 53)),
            (socket.AF_INET6, socket.SOCK_STREAM, ('::1', 80)),
            (socket.AF_NETLINK, socket.SOCK_RAW, None),
            (socket.AF_UNIX, socket.SOCK_STREAM, {str(unix_dir / 'validator.sock')!r}),
            (socket.AF_UNIX, socket.SOCK_STREAM, '\\0judge-{os.getpid()}')]
for family, kind, address in attempts:
    try:
        s = socket.socket(family, kind)
        if address: s.connect(address)
        print('CONNECTED', family)
    except OSError:
        print('denied')
a, b = socket.socketpair(); a.send(b'ok'); print(b.recv(2).decode())
"""
            # devices=["nvidia"] also allows socket(AF_UNIX) (libcuda): still nothing to reach
            results = [self.python(code), self.python(code, devices=["nvidia"])]
        finally:
            listener.close()
            unix_listener.close()
            abstract.close()
            isolate._remove(unix_dir)
        for result in results:
            self.assertNotIn(b"CONNECTED", result.stdout, result.stdout)
            self.assertEqual(result.stdout.split()[-1], b"ok", result.stderr)

    def test_no_namespaces_or_mounts(self):
        result = self.python("""
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
print(libc.unshare(0x10000000), libc.mount(b'none', b'/tmp', b'tmpfs', 0, None))
""")
        self.assertEqual(result.stdout.split(), [b"-1", b"-1"], result.stderr)

    def test_tree_is_a_copy_and_inputs_are_read_only(self):
        tree = Path(tempfile.mkdtemp(prefix="tree-"))
        (tree / "main.c").write_text("int main(){}")
        try:
            result = self.python("""
import os
open('main.c', 'a').write('// edited')
try:
    open(os.path.join(os.environ['HOME'], '..', 'in', 'workload'), 'a').write('x'); print('WROTE')
except OSError:
    print('read-only')
print(open(os.path.join(os.environ['HOME'], '..', 'in', 'workload')).read())
""", tree=tree, inputs={"workload": b"n=5"})
            self.assertEqual(result.stdout.split(), [b"read-only", b"n=5"], result.stderr)
            self.assertEqual((tree / "main.c").read_text(), "int main(){}")
        finally:
            isolate._remove(tree)

    def test_no_fork_profile_keeps_threads(self):
        result = self.python("""
import os, threading
seen = []
t = threading.Thread(target=lambda: seen.append(1)); t.start(); t.join()
print('thread', len(seen))
try:
    os.fork(); print('FORKED')
except OSError:
    print('denied')
""", syscalls="no_fork")
        self.assertEqual(result.stdout.split(), [b"thread", b"1", b"denied"], result.stderr)

    def test_strict_profile_has_no_processes(self):
        result = self.python("import os\ntry:\n os.fork(); print('FORKED')\n"
                        "except OSError:\n print('denied')", syscalls="strict")
        self.assertEqual(result.stdout.strip(), b"denied", result.stderr)


    def test_gpu_devices_only_when_requested(self):
        nodes = [p for p in glob.glob("/dev/nvidia*") if not os.path.isdir(p)]
        if not nodes:
            self.skipTest("no /dev/nvidia* here")
        code = f"""
import os
null = os.stat('/dev/null').st_rdev
for path in {nodes!r}:
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        print('closed'); continue
    print('open' if os.fstat(fd).st_rdev != null else 'closed'); os.close(fd)
print(os.environ.get('CUDA_VISIBLE_DEVICES'))
"""
        saved = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            hidden = self.python(code)
            granted = self.python(code, devices=["nvidia"])
        finally:
            if saved is None:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
        self.assertEqual(hidden.stdout.split(), [b"closed"] * len(nodes) + [b"None"],
                         hidden.stderr)
        self.assertIn(b"open", granted.stdout, granted.stderr)
        self.assertEqual(granted.stdout.split()[-1], b"0")

    def test_cuda_when_requested(self):
        """With a CUDA torch in the judge's interpreter and a GPU: the instance computes on it."""
        try:
            import torch
        except ImportError:
            self.skipTest("no torch here")
        if not torch.cuda.is_available():
            self.skipTest("no CUDA device here")
        code = ("import torch\nx = torch.ones(1024, device='cuda')\n"
                "print(torch.cuda.get_device_name(0), int(x.sum().item()))")
        # without -S: torch is in the interpreter's site-packages (a venv on the runners)
        result = self.run_([PYTHON, "-I", "-c", code], devices=["nvidia"], timeout_seconds=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertTrue(result.stdout.strip().endswith(b"1024"), result.stdout)
        denied = self.run_([PYTHON, "-I", "-c", "import torch\nprint(torch.cuda.is_available())"],
                           timeout_seconds=300)
        self.assertEqual(denied.stdout.strip(), b"False", denied.stderr[-2000:])


class UsernsTests(Backend):
    backend = "userns"

    def test_runs_as_its_own_uid_without_capabilities(self):
        result = self.python("import os\nprint(os.getuid())\n"
                             "print([l.split()[1] for l in open('/proc/self/status')"
                             " if l.startswith('CapEff')][0])")
        self.assertEqual(result.stdout.split(), [str(isolate.USERNS_UID).encode(),
                                                 b"0000000000000000"], result.stderr)

    def test_loopback_is_private(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            result = self.python(f"""
import socket
server = socket.socket(); server.bind(('127.0.0.1', 0)); server.listen(1)
client = socket.create_connection(server.getsockname()); peer, _ = server.accept()
client.send(b'ok'); print(peer.recv(2).decode())
try:
    socket.create_connection(('127.0.0.1', {listener.getsockname()[1]}), timeout=2)
    print('CONNECTED')
except OSError:
    print('denied')
""", network="loopback")
        finally:
            listener.close()
        self.assertEqual(result.stdout.split(), [b"ok", b"denied"], result.stderr)

    def test_layers_are_refused(self):
        with self.assertRaisesRegex(isolate.IsolateError, "landlock backend only"):
            self.python("pass", layers=[isolate.PeakRss()])

    def test_dev_shm_is_private_and_counted(self):
        result = self.python("open('/dev/shm/x', 'wb').write(b'1' * 4096); print('ok')",
                             dev_shm=True)
        self.assertEqual(result.stdout.strip(), b"ok", result.stderr)
        self.assertFalse(os.path.exists("/dev/shm/x"))
        hog = self.python("import time\nf = open('/dev/shm/hog', 'wb')\n"
                          "while True: f.write(b'1' * (16 << 20)); f.flush()",
                          dev_shm=True, memory_bytes=256 << 20)
        self.assertEqual(hog.limit, "memory")



class AccountTests(Backend):
    def test_the_command_has_an_account(self):
        # torch._inductor and others call getpass.getuser(): the instance's uid needs an entry
        # (userns: the launcher's USERNS_USER; landlock: the judge's own account).
        result = self.python("import getpass, os, pwd\n"
                             "entry = pwd.getpwuid(os.getuid())\n"
                             "print(entry.pw_name, getpass.getuser(), entry.pw_dir == os.environ['HOME'])")
        self.assertEqual(result.returncode, 0, result.stderr)
        name, user, home = result.stdout.split()
        self.assertEqual(name, user)
        if self.backend == "userns":
            self.assertEqual((name, home), (isolate.USERNS_USER.encode(), b"True"))


class LimitTests(Backend):
    def test_wall_time_kills_the_whole_tree(self):
        started = time.monotonic()
        result = self.python("import subprocess, sys, time\n"
                        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                        "time.sleep(60)", timeout_seconds=2)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.limit, "time")
        self.assertLess(time.monotonic() - started, 20)

    def test_command_seconds_times_the_command_only(self):
        result = self.python("import time; time.sleep(0.3)", tree=None)
        self.assertGreaterEqual(result.command_seconds, 0.3)
        self.assertLess(result.command_seconds, result.wall_seconds)

    def test_command_seconds_starts_at_exec(self):
        """A no-op binary: the launcher's fork and the instance setup are not inside the time;
        it stays close to a bare exec -> reap on the same pinned CPU."""
        true = shutil.which("true")
        cpu = min(os.sched_getaffinity(0))

        def bare():
            reader, writer = os.pipe()

            def stamp():
                os.sched_setaffinity(0, {cpu})
                os.write(writer, struct.pack("=q", time.perf_counter_ns()))

            process = subprocess.Popen([true], preexec_fn=stamp)
            os.close(writer)
            os.wait4(process.pid, 0)
            ended = time.perf_counter_ns()
            process.returncode = 0
            started = struct.unpack("=q", os.read(reader, 8))[0]
            os.close(reader)
            return (ended - started) / 1e9

        previous = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {cpu})
        try:
            sandboxed = min(self.run_([true]).command_seconds for _ in range(15))
        finally:
            os.sched_setaffinity(0, previous)
        self.assertLess(sandboxed, min(bare() for _ in range(15)) + 0.0005)

    def test_command_keeps_the_judges_cpus(self):
        cpu = min(os.sched_getaffinity(0))
        previous = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {cpu})
        try:
            result = self.python("import os; print(sorted(os.sched_getaffinity(0)))")
        finally:
            os.sched_setaffinity(0, previous)
        self.assertEqual(result.stdout.decode().strip(), str([cpu]))

    def test_unix_socket_below_tmpdir_fits_under_the_verifier_scratch(self):
        # The Verifier's TMPDIR is /frontier/scratch/frontier-verify-XXXXXXXX/tmp (46 bytes); a
        # multiprocessing manager binds <TMPDIR>/pymp-XXXXXXXX/listener-XXXXXXXX inside it.
        scratch = Path(tempfile.mkdtemp(prefix="verifier-scratch-"))
        judge_tmp = scratch / ("x" * max(1, 46 - len(str(scratch)) - 1))
        judge_tmp.mkdir()
        saved, tempfile.tempdir = tempfile.tempdir, str(judge_tmp)
        try:
            result = self.python("""
import multiprocessing
manager = multiprocessing.Manager()
print(manager.list([1, 2])[1])
manager.shutdown()
""", unix_sockets=True)
        finally:
            tempfile.tempdir = saved
            isolate._remove(scratch)
        self.assertEqual((result.returncode, result.stdout), (0, b"2\n"), result.stderr)

    def test_memory_limit(self):
        result = self.python("x = []\nwhile True: x.append(bytearray(16 << 20))",
                        memory_bytes=200 << 20)
        self.assertEqual(result.limit, "memory")

    def test_cpu_limit(self):
        result = self.python("while True: pass", cpu_seconds=1)
        self.assertIn(result.returncode, (-24, -9))   # SIGXCPU, then SIGKILL at the hard limit
        self.assertIsNone(result.limit)

    def test_output_limit(self):
        result = self.python("import sys\nwhile True: sys.stdout.write('x' * 65536)",
                        output_bytes=4096)
        self.assertEqual(result.limit, "output")
        self.assertLessEqual(len(result.stdout), 4096)

    def test_file_size_limit(self):
        # (Python itself ignores SIGXFSZ; a native writer is killed by it)
        result = self.run_(["/bin/sh", "-c", "head -c 10000 /dev/zero >&2"], file_bytes=4096)
        self.assertEqual(result.returncode, -25)      # SIGXFSZ
        self.assertEqual(len(result.stderr), 4096)

    def test_no_file_size_limit_unless_asked(self):
        # .NET sizes an anonymous memfd far beyond any file it writes (double-mapped code).
        # (libc's memfd_create: some judge interpreters lack os.memfd_create)
        result = self.python("import ctypes, os\nfd = ctypes.CDLL(None).memfd_create(b'x', 0)\n"
                        "os.ftruncate(fd, 64 << 30)\nprint('ok')")
        self.assertEqual(result.stdout.strip(), b"ok", result.stderr)

    def test_process_limit_defaults_high_and_is_overridable(self):
        read = "import resource; print(resource.getrlimit(resource.RLIMIT_NPROC))"
        self.assertEqual(self.python(read).stdout.strip(), b"(65536, 65536)")
        self.assertEqual(self.python(read, processes=512).stdout.strip(), b"(512, 512)")

    def test_missing_command_is_exit_127(self):
        result = self.run_(["no-such-command-here"])
        self.assertEqual(result.returncode, 127)

    def test_forged_setup_failure_is_the_commands_result(self):
        # what the launcher once reported its own setup failures as: exit 70 + this stderr line
        result = self.python(FORGED_SETUP_FAILURE)
        self.assertEqual(result.returncode, 70)
        self.assertTrue(result.stderr.startswith(b"isolate: OSError"), result.stderr)

    def test_setup_failure_raises(self):
        for cwd in ("{work}/missing", "{in}/file"):
            with self.subTest(cwd=cwd), \
                    self.assertRaisesRegex(isolate.IsolateError, "instance setup failed"):
                self.python("pass", cwd=cwd, inputs={"file": b"x"})

    def test_bad_arguments(self):
        for kwargs in ({"timeout_seconds": 0}, {"memory_bytes": 0}, {"syscalls": "none"},
                       {"unix_sockets": 1}, {"read_paths": "/usr"}, {"outputs": ["../x"]},
                       {"devices": ["tpu"]}, {"devices": "nvidia"}, {"network": "internet"},
                       {"layers": [object()]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(isolate.IsolateError):
                self.python("pass", **kwargs)


FORGED_SETUP_FAILURE = (
    "import os\nos.write(2, b'isolate: OSError: [Errno 13] seccomp\\n')\nos._exit(70)")


ECHO_WORKER = """
import os, sys
for line in sys.stdin.buffer:
    if line == b"secret?\\n":
        try:
            reply = open(sys.argv[1], "rb").read()
        except OSError:
            reply = b"denied"
        sys.stdout.buffer.write(reply + b"\\n")
    elif line == b"env?\\n":
        names = sorted(k for k in os.environ if k.startswith("FRONTIER_"))
        sys.stdout.buffer.write(repr(names).encode() + b"\\n")
    else:
        sys.stdout.buffer.write(line.upper())
    sys.stdout.buffer.flush()
"""


class WorkerTests(Backend):
    """start(): a long-lived worker behind pipes, inside the same boundary as run()."""

    def setUp(self):
        super().setUp()
        self.judge = Path(tempfile.mkdtemp(prefix="judge-state-"))
        self.secret = self.judge / "reference.patch"
        self.secret.write_text("the reference answer")

    def tearDown(self):
        isolate._remove(self.judge)

    def worker(self, **kwargs):
        kwargs.setdefault("timeout_seconds", 30)
        return self.start_([PYTHON, "-I", "-S", "-c", ECHO_WORKER, str(self.secret)], **kwargs)

    def test_round_trips_inside_the_boundary(self):
        os.environ["FRONTIER_ISOLATE_CANARY"] = "1"
        try:
            with self.worker() as worker:
                for request, reply in ((b"hello\n", b"HELLO\n"), (b"secret?\n", b"denied\n"),
                                       (b"env?\n", b"[]\n"), (b"again\n", b"AGAIN\n")):
                    worker.stdin.write(request)
                    self.assertEqual(worker.stdout.readline(), reply)
        finally:
            del os.environ["FRONTIER_ISOLATE_CANARY"]

    def test_close_kills_the_worker_and_its_children(self):
        marker = self.judge / "survivor"
        worker = self.start_([PYTHON, "-I", "-S", "-c", f"""
import os, subprocess, sys, time
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5); open({str(marker)!r}, 'w')"],
                 start_new_session=True)
print("up", flush=True)
time.sleep(60)
"""], timeout_seconds=60)
        self.assertEqual(worker.stdout.readline(), b"up\n")
        started = time.monotonic()
        result = worker.close()
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(result.returncode, -9)
        self.assertIsNone(result.limit)
        self.assertIs(worker.close(), result)
        time.sleep(6)
        self.assertFalse(marker.exists())

    def test_lifetime_limit_ends_the_worker(self):
        with self.worker(timeout_seconds=1) as worker:
            self.assertEqual(worker.stdout.read(), b"")    # EOF once the lifetime is over
            result = worker.close()
        self.assertTrue(result.timed_out)

    def test_worker_exit_is_seen_as_eof(self):
        worker = self.start_([PYTHON, "-I", "-S", "-c",
                              "import sys; sys.stderr.write('bye'); sys.exit(3)"])
        self.assertEqual(worker.stdout.read(), b"")
        result = worker.close()
        self.assertEqual((result.returncode, result.stderr), (3, b"bye"))

    def test_directories_are_removed(self):
        before = set(Path(tempfile.gettempdir()).glob(isolate.INSTANCE_PREFIX + "*"))
        with self.worker() as worker:
            worker.stdin.write(b"x\n")
            worker.stdout.readline()
        self.assertEqual(set(Path(tempfile.gettempdir()).glob(isolate.INSTANCE_PREFIX + "*")), before)

    def test_bad_arguments(self):
        with self.assertRaises(isolate.IsolateError):
            isolate.start([PYTHON], timeout_seconds=0)

    def test_forged_setup_failure_is_the_commands_result(self):
        worker = self.start_([PYTHON, "-I", "-S", "-c", FORGED_SETUP_FAILURE])
        self.assertEqual(worker.stdout.read(), b"")
        result = worker.close()
        self.assertEqual(result.returncode, 70)
        self.assertTrue(result.stderr.startswith(b"isolate: OSError"), result.stderr)

    def test_setup_failure_raises(self):
        before = set(Path(tempfile.gettempdir()).glob(isolate.INSTANCE_PREFIX + "*"))
        worker = self.worker(cwd="{work}/missing")
        self.assertEqual(worker.stdout.read(), b"")
        with self.assertRaisesRegex(isolate.IsolateError, "instance setup failed"):
            worker.close()
        self.assertEqual(set(Path(tempfile.gettempdir()).glob(isolate.INSTANCE_PREFIX + "*")), before)

    def test_missing_command_is_exit_127(self):
        result = self.start_(["no-such-command-here"]).close()
        self.assertEqual(result.returncode, 127)


HOSTILE_TREE = """
import os
work = os.getcwd()
os.symlink('/', os.path.join(work, 'to-root'))
os.symlink({outside!r}, os.path.join(work, 'to-dir'))
os.symlink('/nonexistent/dangling', os.path.join(work, 'dangling'))
os.makedirs(os.path.join(work, 'a', 'b', 'c'))
os.symlink('/', os.path.join(work, 'a', 'b', 'c', 'deep-root'))
for path in ('a/b/c', 'a/b', 'a'):
    os.chmod(os.path.join(work, path), 0)
home = os.environ['HOME']
os.symlink('/', os.path.join(home, 'to-root'))
os.chmod(home, 0)
"""


class Cleanup:
    """What a command leaves behind never makes cleanup raise, follow a link or touch the judge."""

    def setUp(self):
        super().setUp()
        self.outside = Path(tempfile.mkdtemp(prefix="judge-state-"))
        (self.outside / "keep").write_text("judge")
        self.outside.chmod(0o750)

    def tearDown(self):
        self.outside.chmod(0o700)
        isolate._remove(self.outside)

    def assertOutsideUntouched(self):
        self.assertEqual(self.outside.stat().st_mode & 0o777, 0o750)
        self.assertEqual((self.outside / "keep").read_text(), "judge")
        self.assertEqual(Path("/").stat().st_mode & 0o777, 0o755)

    def sandbox_dirs(self):
        return set(Path(tempfile.gettempdir()).glob(isolate.INSTANCE_PREFIX + "*"))


class CleanupTests(Cleanup, unittest.TestCase):
    """_remove itself: runs on every host and Python (no backend needed)."""

    def test_remove_unlinks_symlinks_and_unlocks_directories(self):
        base = Path(tempfile.mkdtemp(prefix="isolate-cleanup-"))
        nested = base / "work" / "a" / "b" / "c"
        nested.mkdir(parents=True)
        (base / "work" / "to-root").symlink_to("/")
        (base / "work" / "to-dir").symlink_to(self.outside)
        (base / "work" / "dangling").symlink_to("/nonexistent/dangling")
        (nested / "deep-root").symlink_to("/")
        (nested / "file").write_text("x")
        for path in (nested, nested.parent, nested.parent.parent, base / "work"):
            path.chmod(0)
        isolate._remove(base)  # must not raise, on any Python or glibc
        self.assertFalse(os.path.lexists(base))
        self.assertOutsideUntouched()

    def test_remove_of_missing_base_is_quiet(self):
        isolate._remove(Path(tempfile.gettempdir()) / "isolate-cleanup-missing")


class HostileTreeTests(Cleanup, Backend):
    """A run or worker that leaves symlinks and locked directories is an ordinary Result."""

    def test_run(self):
        before = self.sandbox_dirs()
        result = self.python(HOSTILE_TREE.format(outside=str(self.outside)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.sandbox_dirs(), before)
        self.assertOutsideUntouched()

    def test_worker(self):
        before = self.sandbox_dirs()
        code = (HOSTILE_TREE.format(outside=str(self.outside)) +
                "import sys\nsys.stdout.buffer.write(b'ready\\n'); sys.stdout.flush()\n"
                "sys.stdin.read()\n")
        worker = self.start_([PYTHON, "-I", "-S", "-c", code])
        self.assertEqual(worker.stdout.readline(), b"ready\n")
        worker.close()  # must not raise
        self.assertEqual(self.sandbox_dirs(), before)
        self.assertOutsideUntouched()


class NetworkTests(unittest.TestCase):
    def test_loopback_needs_userns(self):
        if not isolate.backend_available("landlock"):
            self.skipTest("no landlock backend here")
        with mock.patch.object(isolate, "BACKENDS", ("landlock",)), \
                self.assertRaises(isolate.IsolateError):
            isolate.run(["true"], timeout_seconds=5, network="loopback")


class LayerTests(Backend):
    """layers=: the extension point a task's judge uses for what only it needs (landlock only)."""

    def test_hooks_run_in_order(self):
        with tempfile.NamedTemporaryFile("r") as log:
            result = self.python("pass", layers=[Recording("a", log.name),
                                                 Recording("b", log.name)])
            self.assertEqual(log.read().split("\n"), [
                "a prepare", "b prepare", "a preexec", "b preexec", "a result", "b result", ""])
        self.assertEqual(result.measurements, {"a": True, "b": True})

    def test_child_setup_and_rules_hold_before_exec(self):
        cpu = min(os.sched_getaffinity(0))
        result = self.python("""
import os
print(sorted(os.sched_getaffinity(0)))
try:
    os.sched_setaffinity(0, range(os.cpu_count())); print('moved')
except PermissionError:
    print('pinned')
""", layers=[Pinned(cpu)])
        self.assertEqual(result.stdout.split(), [f"[{cpu}]".encode(), b"pinned"], result.stderr)

    def test_kept_fd(self):
        with tempfile.NamedTemporaryFile("w") as handle:
            handle.write("passed")
            handle.flush()
            result = self.python("import os\nprint(os.read(100, 6).decode())",
                                 layers=[KeptFile(handle.name)])
        self.assertEqual(result.stdout.strip(), b"passed", result.stderr)

    def test_peak_rss_is_the_command_peak(self):
        # A layer defined in isolate.py itself: the launcher resolves it to its own module.
        result = self.python("b = bytearray(200 << 20)\nb[::4096] = b'x' * len(b[::4096])",
                             layers=[isolate.PeakRss()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(result.measurements["max_rss_bytes"], 200 << 20)
        self.assertLess(result.measurements["max_rss_bytes"], 400 << 20)

    def test_relaxing_layer_must_declare_it(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(2)
            port = listener.getsockname()[1]
            with self.assertRaisesRegex(isolate.IsolateError, "relaxes"):
                self.python("pass", layers=[Undeclared(port)])
            with socket.socket() as other:
                other.bind(("127.0.0.1", 0))
                other.listen(1)
                result = self.python(f"""
import socket
for port in ({port}, {other.getsockname()[1]}):
    try:
        socket.create_connection(('127.0.0.1', port), timeout=2).close(); print('connected')
    except OSError:
        print('denied')
try:
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM); print('udp')
except OSError:
    print('denied')
""", layers=[Connect(port)])
        self.assertEqual(result.stdout.split(), [b"connected", b"denied", b"denied"],
                         result.stderr)

    def test_bad_layers(self):
        class Nested(isolate.Layer):
            pass
        unserializable = Pinned(0)
        unserializable.cpu = {0}
        for layers in (Pinned(0), [object()], [Nested()], [unserializable]):
            with self.subTest(layers=layers), self.assertRaises(isolate.IsolateError):
                self.python("pass", layers=layers)


class ProgramTests(unittest.TestCase):
    def test_seccomp_program_checks_arch_first(self):
        for profile in isolate.SYSCALL_PROFILES:
            program = isolate.seccomp_program(profile, False, "loopback")
            self.assertEqual(program[0], (0x20, 0, 0, 4))
            self.assertEqual(program[1][3], isolate.AUDIT_ARCH_X86_64)
            self.assertLessEqual(len(program), 4096)

    def test_nvidia_allows_exactly_its_socket_families(self):
        def families(**kwargs):
            program = isolate.seccomp_program("default", False, "none", **kwargs)
            start = program.index((0x20, 0, 0, 16))
            return {k for code, _, _, k in program[start + 1:] if code == 0x15}
        self.assertEqual(families(), set())
        self.assertEqual(families(devices=["nvidia"]), {isolate.AF_UNIX})



for _name in isolate.BACKENDS:
    for _base in (EscapeTests, AccountTests, LimitTests, WorkerTests, UsernsTests,
                  HostileTreeTests, LayerTests):
        if (_base is UsernsTests and _name != "userns") \
                or (_base is LayerTests and _name != "landlock"):
            continue
        _class = type(f"{_base.__name__.removesuffix('Tests')}Tests_{_name}",
                      (_base, unittest.TestCase), {"backend": _name})
        globals()[_class.__name__] = unittest.skipUnless(
            isolate.backend_available(_name), f"needs the {_name} backend")(_class)
del _name, _base, _class


if __name__ == "__main__":
    unittest.main()
