"""isolate: the launcher where a judge runs submitted code as a jailed subprocess (#757).

Not a container: task.toml's [test_sandboxes.*] are separate containers beside the judge.

A judge is trusted; the code it builds from a submission is not. A judge runs every command that
executes submitted code (an interpreter that imports it, a binary built from it) with `run`
instead of `subprocess`, or, for a long-lived worker it talks to over pipes, with `start`
instead of `subprocess.Popen`. A trusted program that is not Python and spawns submitted code
itself, on pipes it set up and with the child's wait4 status and resource usage as its
measurement (a C harness), spawns a small Python program that calls `exec_on_stdio` instead of
the submitted program: that process becomes the instance's launcher, so the harness's child
keeps the harness's pipes, its exit status and its resource usage. The command then runs in a
fresh instance with:

  - read access only to the system directories (SYSTEM_READ), the judge's own interpreter, its
    own writable copy of `tree`, its read-only `inputs` and the `read_paths` the judge names:
    the judge, its data, the reference, the verifier input/output and every other run are
    unreadable (symlinks into them resolve to denied paths);
  - writes only to its own private directories (work, out, home, tmp; HOME and TMPDIR point
    there), deleted after the run, so nothing persists from one run to the next (no /tmp,
    /dev/shm, SysV IPC, POSIX message queues or keyrings);
  - an environment built from nothing (PATH, HOME, TMPDIR, LANG, LC_ALL plus `env`): no
    FRONTIER_* variable, no Driver token;
  - no network sockets (socketpair() works; AF_UNIX sockets only with unix_sockets=True: then
    under Landlock the instance can connect to any pathname AF_UNIX socket it can name, so a
    judge passing it keeps no pathname socket of its own while the instance runs), no
    ptrace or /proc/<pid>/{environ,mem} of the judge, no signals to it, no mount, namespace,
    BPF, perf, io_uring or keyring calls; with syscalls="no_fork" no new processes (threads
    still work), with "strict" neither processes nor threads nor signals;
  - limits on wall time, CPU time, summed resident memory, private disk, output bytes,
    processes, open files and file size; every process it started is killed when the run ends
    (the launcher is a child subreaper, so double-forked daemons are found too).

Two backends build the same boundary; the first of them the host can build is used, and
Result.backend names it:

  landlock  Landlock (ABI >= 6) + seccomp + a child-subreaper launcher process. It needs no
            privilege, so it works in the EKS Verifier pod (uid 65532, every capability dropped,
            the runtime's default seccomp profile). The command keeps the judge's uid.
  userns    for Verifier GPU runners (Modal, gVisor: unprivileged user namespaces work, Landlock
            does not). Fresh user, mount, network, PID and IPC namespaces; every directory that
            can hold judge state (PROTECTED_ROOTS, TMPDIR) is covered by an empty read-only tmpfs,
            with only the instance's own directories and the read_paths bound back in; /proc is
            the instance's own; then a nested user namespace in which the command is uid
            USERNS_UID without capabilities, and the same seccomp filter.

A boundary that cannot be built is an IsolateError (a verification failure, never a candidate
score and never a silent fallback to running unsandboxed). The first run of each backend in a
process self-tests it. Setup failures reach the launcher only through a close-on-exec pipe
(as in CPython's subprocess), so nothing the command does is ever an IsolateError.

GPUs: `devices=["nvidia"]` gives the instance the /dev/nvidia* nodes (and CUDA_VISIBLE_DEVICES /
NVIDIA_VISIBLE_DEVICES from the judge's environment); without it they are unreachable (landlock:
not granted; userns: /dev/null bound over them). The CUDA libraries live under /usr and the
judge's interpreter, which every instance may read; under userns the instance may also create
AF_UNIX sockets, which libcuda needs on the runners (DEVICE_SOCKETS). `network="loopback"`
(userns only) gives the instance a private loopback interface and AF_INET/AF_INET6/AF_NETLINK
sockets in its own empty network namespace (vLLM's local rendezvous).

Layers (landlock backend only): what one task needs beyond this boundary is a Layer its judge
defines and passes as `layers=[...]`; this file knows no task. A layer that allows something the core denies names it
in `relaxes`, and scripts/validate_task_layout.py lists every task's relaxations for review.
Generic layers several judges use live here (PeakRss).

The launcher half of this file runs as `python3 -I -S isolate.py <control dir>`.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import glob
import importlib.util
import inspect
import json
import os
import platform
import resource
import select
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

LAUNCHER = os.path.abspath(__file__)
BACKENDS = ("landlock", "userns")
# Read-only (and executable) roots every instance may use. Judge state never lives here: the
# task bundle is /opt/frontier/task, scratch is under /frontier or TMPDIR.
SYSTEM_READ = ("/usr", "/lib", "/lib32", "/lib64", "/libx32", "/bin", "/sbin", "/etc", "/proc",
               "/sys")
DEVICES_RW = ("/dev/null", "/dev/zero", "/dev/full")
DEVICES_RO = ("/dev/random", "/dev/urandom")
# Where judge state lives: a read path may lie below these but never contain one of them.
PROTECTED_ROOTS = ("/opt/frontier/task", "/frontier", "/datasets", "/tmp",
                   "/var/tmp", "/dev/shm", "/workspace", "/root", "/home", "/run", "/platform")
DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"
SYSCALL_PROFILES = ("default", "no_fork", "strict")
NETWORKS = ("none", "loopback")
DEVICES = {"nvidia": ("/dev/nvidia*", "/dev/nvidia-caps/*")}
# Passed on from the judge's environment to an instance given that device class.
DEVICE_ENV = {"nvidia": ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")}
# Socket families a device class needs beyond the call's own (seccomp; the families' numbers
# are below). libcuda's cuInit fails with CUDA_ERROR_OPERATING_SYSTEM on the GPU runners
# (gVisor nvproxy) when socket(AF_UNIX) is refused (Verifier GPU runner probes, #854). Under
# the userns backend the instance has its own empty network namespace either way; the landlock
# backend does not add these (see _prepare).
DEVICE_SOCKETS = {"nvidia": ("AF_UNIX",)}
# userns backend: the command's uid inside its nested user namespace (no capabilities there).
USERNS_UID = 60000
# userns backend: the account the instance's /etc/passwd and /etc/group give USERNS_UID (with
# the instance's HOME), so getpwuid()/getpass.getuser() work in any image (torch._inductor and
# others look the user up).
USERNS_USER = "frontier-isolate"
# userns backend: size of the private tmpfs over /dev/shm when dev_shm is true (the instance's
# memory limit counts what it holds).
USERNS_SHM_BYTES = 8 << 30
# Defaults of the limits a call does not set. memory_bytes, cpu_seconds and file_bytes default
# to none (the Verifier pod's own cgroup still bounds memory; private disk is bounded by
# disk_bytes; an RLIMIT_FSIZE would also hit runtimes that size an anonymous memfd, like .NET).
# processes (RLIMIT_NPROC) is counted per uid, shared with the judge, and counts threads: torch,
# vLLM and Ray start hundreds of threads per runner CPU, so its default is high.
DEFAULT_LIMITS = {"disk_bytes": 8 << 30, "output_bytes": 16 << 20, "processes": 1 << 16,
                  "open_files": 4096}
MIN_LANDLOCK_ABI = 6  # scoped signals and abstract sockets (Linux 6.12)
# Landlock net access rights (LANDLOCK_ACCESS_NET_BIND_TCP, _CONNECT_TCP): every instance is
# denied both except for the ports a layer allows (Rules.allow_tcp_port).
NET_BIND_TCP, NET_CONNECT_TCP = 1 << 0, 1 << 1
POLL_SECONDS = 0.05
DISK_POLL_SECONDS = 1.0
KILL_GRACE_SECONDS = 10.0
# An instance lives in <tempdir>/ts-XXXXXXXX/r/{work,in,out,home,tmp}. The names are short so
# that an AF_UNIX socket bound below TMPDIR still fits sun_path (107 bytes): Python's
# multiprocessing managers bind <TMPDIR>/pymp-XXXXXXXX/listener-XXXXXXXX (32 more bytes), and the
# Verifier's TMPDIR is already /frontier/scratch/frontier-verify-XXXXXXXX/tmp.
INSTANCE_PREFIX = "ts-"


class IsolateError(RuntimeError):
    """The sandbox could not be built or was used wrongly: a judge/infrastructure fault."""


@dataclass(frozen=True)
class Result:
    """One instance's outcome. `limit` names the limit that ended it, if any."""
    returncode: int            # the command's exit status; -N when killed by signal N
    stdout: bytes
    stderr: bytes
    outputs: dict = field(default_factory=dict)   # declared output name -> bytes | None
    timed_out: bool = False
    limit: str | None = None   # "time" | "memory" | "disk" | "output" | None
    wall_seconds: float = 0.0     # the whole call: staging, launch, cleanup
    command_seconds: float = 0.0  # the command alone: exec to exit (for timed judges)
    backend: str = ""             # "landlock" | "userns"
    measurements: dict = field(default_factory=dict)   # what layers added (Layer.result)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.limit is None


class Layer:
    """A task's addition to the boundary, passed as run/start(layers=[...]).

    Subclass it in a module of its own in the task's judge (the launcher loads that file in its
    own process), as a top-level class whose instance attributes are JSON values: the launcher
    rebuilds the layer from them without calling __init__. Hooks, each called for every layer in
    the order the judge gave:

      1. prepare(rules)       launcher, before fork: rules for the core to add (Rules)
      2. preexec()            the command's process, after the core's rlimits, before the core
                              confines it (Landlock, seccomp) and execs the command
      3. result(result, run)  judge, after the command ended and before its directories are
                              deleted: returns the Result, e.g. with `measurements` added

    relaxes  what the layer allows that the core denies (e.g. ("tcp",)). Rules refuses an allow
             rule from a layer that names nothing; scripts/validate_task_layout.py lists every
             task's relaxations, so keep it a literal tuple of strings in the class body.
    """
    relaxes: tuple = ()

    def prepare(self, rules: "Rules") -> None:
        pass

    def preexec(self) -> None:
        pass

    def result(self, result: Result, run: "Run") -> Result:
        return result


class Rules:
    """What one layer adds to an instance's Landlock ruleset, seccomp filter and kept fds."""

    def __init__(self, backend: str, relaxes: tuple):
        self.backend, self._relaxes = backend, relaxes
        self.denied: set = set()          # syscall numbers -> EPERM
        self.sockets: dict = {}           # family -> (type, protocols) allowed
        self.tcp_ports: dict = {}         # port -> NET_BIND_TCP | NET_CONNECT_TCP
        self.fds: set = set()             # launcher fds the command keeps

    def _relaxing(self, what: str) -> None:
        if not self._relaxes:
            raise IsolateError(f"a layer that allows {what} must say so in `relaxes`")

    def deny_syscall(self, number: int) -> None:
        self.denied.add(number)

    def allow_socket(self, family: int, kind: int, protocols) -> None:
        """socket(family, kind [| SOCK_NONBLOCK | SOCK_CLOEXEC], protocol in `protocols`)."""
        self._relaxing(f"socket family {family}")
        self.sockets[family] = (kind, tuple(protocols))

    def allow_tcp_port(self, port: int, access: int) -> None:
        """Landlock: TCP bind (NET_BIND_TCP) and/or connect (NET_CONNECT_TCP) to `port`."""
        self._relaxing(f"TCP port {port}")
        self.tcp_ports[port] = self.tcp_ports.get(port, 0) | access

    def keep_fd(self, descriptor: int) -> None:
        self.fds.add(descriptor)


@dataclass(frozen=True)
class Run:
    """What Layer.result sees of the ended command besides its Result."""
    paths: dict    # the instance's directories (work, in, out, home, tmp), still present
    rusage: dict   # the command's wait4 resource usage (ru_* fields; on userns the init's,
                   # which includes the command it waited for)


class PeakRss(Layer):
    """Result.measurements["max_rss_bytes"]: the command's peak resident set, wait4's ru_maxrss
    (the launcher reaps exactly the command, so it is the kernel's high-water mark for it and
    the children it waited for), measured outside the instance. For judges that score or limit
    memory."""

    def result(self, result, run):
        return replace(result, measurements={
            **result.measurements, "max_rss_bytes": run.rusage["ru_maxrss"] * 1024})


# ------------------------------------------------------------------ judge-side API
_LOCK = threading.Lock()
_SELF_TESTED: set = set()


def run(argv, *, stdin: bytes = b"", **options) -> Result:
    """Run `argv` in a fresh instance and return its Result. `stdin` is the command's whole
    standard input; `options` are those of _check_call."""
    call = _check_call(argv, **options)
    _self_test(call["backend"])
    return _launch(call, stdin=bytes(stdin))


def start(argv, **options) -> "Instance":
    """Start `argv` as a long-lived worker in a fresh instance and return its Instance.

    The boundary and `options` are those of `run`; timeout_seconds bounds the worker's whole
    life. Its stdin and stdout are pipes to the judge (Instance.stdin, Instance.stdout:
    unbuffered binary files); stderr is collected as in `run`. Instance.close() ends the worker
    (SIGKILL to every process it started), deletes its directories and returns its Result; the
    worker also ends when the judge's end of the instance goes away (the judge exits).
    """
    call = _check_call(argv, **options)
    _self_test(call["backend"])
    return Instance(call)


def exec_on_stdio(argv, *, failure_record: str, **options):
    """Replace this process with the launcher of a fresh instance that runs `argv` on this
    process's standard input and output; never returns.

    The boundary and `options` are those of `run`, except that nothing is collected: no
    `outputs` and no `layers` (their result hook needs a judge after the run); the command's
    stderr is discarded (bounded by output_bytes). fds 0 and 1 are inherited, not copied, so the
    command reads and writes this process's own pipes. This process (its pid) stays the
    launcher until the instance has ended and its directories are deleted, then ends exactly as
    the command did: its exit code, whatever it is, or death by the same signal (a limit ends
    the command with SIGKILL, so the launcher dies by SIGKILL too; no core is written).

    A failure of the launcher itself (the call, the self-test, building the instance, the
    launcher's own bookkeeping) is never an exit status, which the command could produce too:
    it is reported out of band, in `failure_record`, an absolute path the caller owns (its
    directory exists and lies outside every instance; the file must not exist yet). The record
    (JSON {"error": ...}) is written, then this process exits with status 1; the caller tells a
    launcher failure apart from the command's own exit 1 only by the record. A path that cannot
    hold the record raises IsolateError here, before anything runs.

    wait4 on this process therefore returns the command's outcome, and its resource usage
    includes the command's (the launcher reaps it) plus this process's own CPU before and after
    the command (the Python start, the self-test of the backend's first instance in this
    process, staging, cleanup): a fixed cost per call, never a cost per byte of the command's
    I/O.
    """
    record = Path(failure_record) if isinstance(failure_record, str) else None
    if record is None or not record.is_absolute() or not record.parent.is_dir() \
            or os.path.lexists(record):
        raise IsolateError("failure_record must be a new file's absolute path in an existing "
                           "directory")
    try:
        if options.get("outputs") or options.get("layers"):
            raise IsolateError("exec_on_stdio collects nothing: no outputs or layers")
        call = _check_call(argv, **options)
        _self_test(call["backend"])
        _base, control, _paths, spec = _prepare_launch(call, b"")
        spec["pipes"] = {"stdin": 0, "stdout": 1, "stop": None}
        spec["exec"] = {"failure_record": str(record)}
        (control / "spec.json").write_text(json.dumps(spec))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, "-I", "-S", LAUNCHER, str(control)])
    except BaseException as error:  # noqa: BLE001 - reported out of band, never as a status
        _fail_out_of_band(str(record), f"{type(error).__name__}: {error}")


def _fail_out_of_band(record: str, error: str) -> None:
    """exec_on_stdio: write the launcher's failure record, then exit 1."""
    descriptor = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    os.write(descriptor, json.dumps({"error": error}).encode())
    os.close(descriptor)
    os._exit(1)


def _check_call(argv, *, timeout_seconds: float, tree=None, inputs=None, outputs=(),
                read_paths=(), env=None, cwd: str = "{work}", memory_bytes: int | None = None,
                cpu_seconds: int | None = None, processes: int | None = None,
                open_files: int | None = None, file_bytes: int | None = None,
                output_bytes: int | None = None, disk_bytes: int | None = None,
                syscalls: str = "default", unix_sockets: bool = False, dev_shm: bool = False,
                path: str = DEFAULT_PATH, devices=(), network: str = "none",
                layers=()) -> dict:
    """The options of run/start, validated into the call the launcher gets.

    tree        a directory copied (writable) to the instance's work dir, the default cwd
    inputs      {name: path | bytes}: copied read-only into the instance's in/ dir
    outputs     names the command may write under out/; returned (bounded) in Result.outputs
    read_paths  existing files or directories the command may read in place (not copied):
                absolute, symlink-free, never / or a directory holding judge state
    argv, cwd and env values may use {work} {in} {out} {home} {tmp}: the instance's own paths.
    cpu_seconds sets RLIMIT_CPU (soft; SIGKILL 2 s later); processes, open_files and file_bytes
    set RLIMIT_NPROC (per uid, shared with the judge), RLIMIT_NOFILE and RLIMIT_FSIZE.
    devices     device classes the instance may use: [] or ["nvidia"]
    network     "none", or "loopback" (a private lo; userns backend only)
    layers      the task's Layer instances, applied in this order (landlock backend only)
    """
    argv = list(argv) if isinstance(argv, (list, tuple)) else None
    if not argv or not all(isinstance(a, str) and "\0" not in a for a in argv):
        raise IsolateError("argv must be a non-empty list of strings")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) \
            or not 0 < timeout_seconds < 7 * 86400:
        raise IsolateError("timeout_seconds must be a positive number")
    limits = {"timeout_seconds": float(timeout_seconds)}
    given = {"memory_bytes": memory_bytes, "cpu_seconds": cpu_seconds, "processes": processes,
             "open_files": open_files, "file_bytes": file_bytes, "output_bytes": output_bytes,
             "disk_bytes": disk_bytes}
    for key, value in given.items():
        if value is None:
            value = DEFAULT_LIMITS.get(key)
        elif type(value) is not int or value <= 0:
            raise IsolateError(f"{key} must be a positive integer")
        limits[key] = value
    if syscalls not in SYSCALL_PROFILES:
        raise IsolateError(f"syscalls must be one of {list(SYSCALL_PROFILES)}")
    for key, value in (("unix_sockets", unix_sockets), ("dev_shm", dev_shm)):
        if type(value) is not bool:
            raise IsolateError(f"{key} must be true or false")
    if not isinstance(path, str) or not path:
        raise IsolateError("path must be a PATH string")
    if not isinstance(cwd, str) or not cwd:
        raise IsolateError("cwd must be a path or placeholder")
    if isinstance(devices, str) or not all(name in DEVICES for name in devices):
        raise IsolateError(f"devices must be a list of {sorted(DEVICES)}")
    devices = sorted(set(devices))
    backend = choose_backend()
    if network not in NETWORKS:
        raise IsolateError(f"network must be one of {list(NETWORKS)}")
    if network != "none" and backend != "userns":
        raise IsolateError(f"network={network!r} needs the userns backend")
    if layers and backend != "landlock":
        raise IsolateError("layers are supported on the landlock backend only")
    outputs = list(outputs)
    for name in outputs:
        _check_name("output", name)
    options = {**limits, "syscalls": syscalls, "unix_sockets": unix_sockets, "dev_shm": dev_shm,
               "path": path, "read_paths": _check_read_paths(read_paths), "devices": devices,
               "network": network, "layers": _layer_specs(layers)}
    call_env = {name: os.environ[name] for device in devices for name in DEVICE_ENV[device]
                if name in os.environ}
    call_env.update(_check_env("env", dict(env or {})))
    return {"backend": backend, "argv": argv, "options": options, "outputs": outputs,
            "env": call_env, "tree": tree, "inputs": dict(inputs or {}), "cwd": cwd,
            "layers": list(layers)}


def _layer_specs(layers) -> list[dict]:
    """How the launcher rebuilds each layer: its class's file and name, and its attributes."""
    if isinstance(layers, Layer) or not all(isinstance(layer, Layer) for layer in layers):
        raise IsolateError("layers must be a list of Layer instances")
    specs = []
    for layer in layers:
        cls = type(layer)
        spec = {"file": inspect.getfile(cls), "class": cls.__qualname__, "state": vars(layer)}
        try:
            json.dumps(spec)
        except (TypeError, ValueError) as error:
            raise IsolateError(f"layer {cls.__qualname__}: attributes must be JSON values") \
                from error
        if "." in cls.__qualname__:
            raise IsolateError(f"layer {cls.__qualname__} must be a top-level class")
        specs.append(spec)
    return specs


def _check_read_paths(paths) -> list[str]:
    """The judge-named read paths, validated: each exists, is absolute, normalized and free
    of symlinks (a submitted tree may hold a link into the judge), and holds no judge root."""
    if isinstance(paths, (str, os.PathLike)):
        raise IsolateError("read_paths must be a list of paths")
    checked = []
    for value in paths:
        text = os.fspath(value)
        if not isinstance(text, str) or not text.startswith("/"):
            raise IsolateError(f"read_paths: {text!r} must be an absolute path")
        normal = os.path.normpath(text)
        if normal == "/" or os.path.realpath(normal) != normal or not os.path.exists(normal):
            raise IsolateError(f"read_paths: {text!r} must be an existing path other than "
                                   "/ with no symlink in it")
        for root in (*PROTECTED_ROOTS, tempfile.gettempdir()):
            if _within(root, normal):
                raise IsolateError(f"read_paths: {text!r} contains judge state under {root}")
        checked.append(normal)
    return checked


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _check_name(kind: str, name: object) -> None:
    if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
        raise IsolateError(f"invalid {kind} name {name!r}")


def _check_env(where: str, env: object) -> dict:
    if not isinstance(env, dict):
        raise IsolateError(f"{where} must be a table of strings")
    for key, value in env.items():
        if (not isinstance(key, str) or not isinstance(value, str) or not key or "=" in key
                or "\0" in key or "\0" in value):
            raise IsolateError(f"{where}: invalid variable {key!r}")
    return env


_USERNS: dict = {}


def backend_available(name: str) -> bool:
    """Whether this host can build that backend (x86_64 Linux only: the seccomp filter)."""
    if sys.platform != "linux" or platform.machine() != "x86_64":
        return False
    if name == "landlock":
        return landlock_abi() >= MIN_LANDLOCK_ABI
    if name == "userns":
        with _LOCK:
            if "ok" not in _USERNS:
                _USERNS["ok"] = _userns_probe()
        return _USERNS["ok"]
    return False


def available() -> bool:
    return any(backend_available(name) for name in BACKENDS)


def choose_backend() -> str:
    """The first of BACKENDS this host can build."""
    for name in BACKENDS:
        if backend_available(name):
            return name
    raise IsolateError(
        f"no isolate backend here: needs Linux x86_64 with Landlock ABI >= "
        f"{MIN_LANDLOCK_ABI} (found {landlock_abi()}) or unprivileged user namespaces")


def _userns_probe() -> bool:
    """Can this process build the userns backend's namespaces (and mount a tmpfs in them)?"""
    code = ("import ctypes, os\n"
            "libc = ctypes.CDLL(None, use_errno=True)\n"
            "uid, gid = os.getuid(), os.getgid()\n"
            "assert libc.unshare(0x10000000 | 0x00020000 | 0x40000000 | 0x08000000 | 0x20000000) == 0\n"
            "for name, text in (('setgroups', 'deny'), ('uid_map', f'0 {uid} 1'), ('gid_map', f'0 {gid} 1')):\n"
            "    open(f'/proc/self/{name}', 'w').write(text)\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    assert libc.mount(None, b'/', None, 16384 | (1 << 18), None) == 0\n"
            "    assert libc.mount(b'tmpfs', b'/tmp', b'tmpfs', 2 | 4, b'size=1m') == 0\n"
            "    os._exit(0 if libc.unshare(0x10000000) == 0 else 1)\n"
            "os._exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))\n")
    try:
        probe = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True,
                               env={}, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _probe(chosen: str) -> dict:
    """Probe the boundary from inside an instance; raises IsolateError if it leaks."""
    with tempfile.TemporaryDirectory(prefix="isolate-canary-") as canary_dir:
        canary = Path(canary_dir) / "canary"
        canary.write_text("judge-secret")
        options = {"timeout_seconds": 60.0, "memory_bytes": None, "cpu_seconds": None,
                   "file_bytes": None,
                   **DEFAULT_LIMITS, "syscalls": "default", "unix_sockets": False,
                   "dev_shm": False, "path": DEFAULT_PATH, "read_paths": [], "devices": [],
                   "network": "none", "layers": []}
        argv = [sys.executable, "-I", "-S", "-c", _PROBE, str(canary), str(os.getpid()),
                str(os.getuid()), chosen]
        result = _launch({"backend": chosen, "argv": argv, "options": options, "outputs": [],
                          "env": {}, "tree": None, "inputs": {}, "cwd": "{work}", "layers": []},
                         stdin=b"")
    try:
        report = json.loads(result.stdout.decode())
    except ValueError as error:
        raise IsolateError(f"isolate self-test produced no report "
                               f"(exit {result.returncode}): {result.stderr[-2000:]!r}") from error
    leaks = [key for key, held in report.items() if held is not True]
    if leaks:
        raise IsolateError(f"isolate self-test failed: {leaks}")
    return report


def _self_test(backend: str) -> None:
    with _LOCK:
        tested = backend in _SELF_TESTED
    if not tested:
        _probe(backend)
        with _LOCK:
            _SELF_TESTED.add(backend)


# Runs inside the instance: each key True when the boundary holds.
_PROBE = r"""
import glob, json, os, socket, stat, sys
canary, judge, judge_uid, backend = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
def denied(action):
    try:
        action()
    except OSError:
        return True
    return False
def network():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.close()
def gpu_closed(path):
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return True
    try:
        return os.fstat(fd).st_rdev == os.stat("/dev/null").st_rdev
    finally:
        os.close(fd)
def named_user():
    # Libraries look the user up (torch._inductor: getpass.getuser()); the uid needs an account.
    import getpass, pwd
    try:
        return pwd.getpwuid(os.getuid()).pw_name == getpass.getuser() != ""
    except (KeyError, OSError):
        return False
report = {
    "judge_file_unreadable": denied(lambda: open(canary).read()),
    "judge_dir_unreadable": not os.path.exists("/opt/frontier/task")
                            or denied(lambda: os.listdir("/opt/frontier/task"))
                            or os.listdir("/opt/frontier/task") == [],
    "env_clean": not any(k.startswith(("FRONTIER_", "HARBOR_")) for k in os.environ),
    "no_inet_socket": denied(network),
    "tmp_not_writable": denied(lambda: open("/tmp/isolate-probe", "w").close()),
    "cannot_signal_judge": denied(lambda: os.kill(judge, 0)),
    "judge_environ_unreadable": denied(lambda: open("/proc/%d/environ" % judge, "rb").read()),
    "no_capabilities": all(line.split()[1].strip("0") == "" for line in open("/proc/self/status")
                           if line.startswith(("CapEff:", "CapPrm:"))),
    "gpu_unreachable": all(gpu_closed(path) for path in glob.glob("/dev/nvidia*")),
    "named_user": named_user(),
}
if backend == "userns":
    # its own PID namespace: the judge's pid number means nothing here; only the namespace's
    # init (pid 1, the probe's parent) and the probe itself are seen
    alone = ({int(e) for e in os.listdir("/proc") if e.isdigit()} == {1, os.getpid()}
             and os.getppid() == 1)
    report["cannot_signal_judge"] = report["judge_environ_unreadable"] = alone
    report["own_uid"] = os.getuid() not in (0, judge_uid)
print(json.dumps(report))
"""


def _interpreter_roots() -> list[str]:
    """The judge's Python, so a judge may run `sys.executable` on submitted code."""
    roots = []
    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        prefix = os.path.realpath(prefix)
        # never a directory that holds judge state, nor the task bundle or scratch itself
        if not any(_within(root, prefix) for root in (*PROTECTED_ROOTS, tempfile.gettempdir())) \
                and not _within(prefix, "/opt/frontier/task") \
                and not _within(prefix, tempfile.gettempdir()):
            roots.append(prefix)
    return sorted(roots)


def _stage(root: Path, tree, inputs: dict) -> dict:
    paths = {name: root / name for name in ("work", "in", "out", "home", "tmp")}
    for name, path in paths.items():
        if name == "work" and tree is not None:
            shutil.copytree(tree, path, symlinks=True)
        else:
            path.mkdir()
    for name, value in inputs.items():
        _check_name("input", name)
        target = paths["in"] / name
        if isinstance(value, (bytes, bytearray)):
            target.write_bytes(value)
        elif Path(value).is_dir():
            shutil.copytree(value, target, symlinks=True)
        else:
            shutil.copyfile(value, target)
    return {name: str(path) for name, path in paths.items()}


def _prepare_launch(call: dict, stdin: bytes) -> tuple:
    """Stage an instance's directories and the launcher's spec: (base, control, paths, spec)."""
    base = Path(tempfile.mkdtemp(prefix=INSTANCE_PREFIX))
    options = call["options"]
    try:
        control = base / "control"
        control.mkdir(mode=0o700)
        root = base / "r"
        root.mkdir(mode=0o700)
        paths = _stage(root, call["tree"], call["inputs"])

        def expand(text: str) -> str:
            for name, path in paths.items():
                text = text.replace("{" + name + "}", path)
            return text

        environment = {"PATH": options["path"], "HOME": paths["home"], "TMPDIR": paths["tmp"],
                       "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        environment.update({k: expand(v) for k, v in call["env"].items()})
        for name, data in (("stdin", stdin), ("stdout", b""), ("stderr", b"")):
            (control / name).write_bytes(data)
        spec = {"backend": call["backend"], "argv": [expand(a) for a in call["argv"]],
                "cwd": expand(call["cwd"]),
                "env": environment, "root": str(root), "paths": paths, "options": options,
                "interpreter": _interpreter_roots(), "control": str(control), "pipes": None,
                "hidden": _hidden_roots()}
        return base, control, paths, spec
    except BaseException:
        _remove(base)
        raise


def _collect(call: dict, control: Path, paths: dict, launcher_exit: int,
             launcher_stderr: bytes, started: float) -> Result:
    try:
        status = json.loads((control / "status.json").read_text())
    except (OSError, ValueError) as error:
        raise IsolateError(f"isolate launcher failed (exit {launcher_exit}): "
                               f"{launcher_stderr[-2000:]!r}") from error
    if status.get("error"):
        raise IsolateError(f"isolate launcher: {status['error']}")
    cap = call["options"]["output_bytes"]
    collected = {}
    budget = cap
    for name in call["outputs"]:
        data = _read_bounded(Path(paths["out"]) / name, budget)
        collected[name] = data
        budget -= len(data or b"")
    result = Result(returncode=status["returncode"],
                    stdout=_read_bounded(control / "stdout", cap) or b"",
                    stderr=_read_bounded(control / "stderr", cap) or b"",
                    outputs=collected, timed_out=status["limit"] == "time",
                    limit=status["limit"], wall_seconds=round(time.monotonic() - started, 3),
                    command_seconds=status["command_ns"] / 1e9, backend=call["backend"])
    run = Run(paths=paths, rusage=status["rusage"])
    for layer in call["layers"]:
        result = layer.result(result, run)
    return result


def _launch(call: dict, stdin: bytes) -> Result:
    started = time.monotonic()
    base, control, paths, spec = _prepare_launch(call, stdin)
    try:
        (control / "spec.json").write_text(json.dumps(spec))
        process = subprocess.run([sys.executable, "-I", "-S", LAUNCHER, str(control)],
                                 stdin=subprocess.DEVNULL, capture_output=True, env={},
                                 timeout=call["options"]["timeout_seconds"] + 120)
        return _collect(call, control, paths, process.returncode, process.stderr, started)
    except subprocess.TimeoutExpired as error:
        raise IsolateError("isolate launcher did not finish") from error
    finally:
        _remove(base)


def _hidden_roots() -> list[str]:
    """userns backend: the existing directories covered by an empty tmpfs (judge state)."""
    found = []
    for path in (*PROTECTED_ROOTS, tempfile.gettempdir(), "/dev/mqueue"):
        real = os.path.realpath(path)
        if os.path.isdir(real) and real != "/":
            found.append(real)
    # the outermost of nested roots only (a mount over /tmp already hides /tmp/x)
    return sorted({p for p in found if not any(q != p and _within(p, q) for q in found)})


class Instance:
    """A worker started by `start`: stdin/stdout pipes to the command, close() ends it."""

    def __init__(self, call: dict):
        self._started = time.monotonic()
        self._call, self._result = call, None
        self._base, self._control, self._paths, spec = _prepare_launch(call, b"")
        ends = []
        try:
            stdin_r, stdin_w = os.pipe()
            ends += [stdin_r, stdin_w]
            stdout_r, stdout_w = os.pipe()
            ends += [stdout_r, stdout_w]
            # The launcher ends the instance when this pipe reports EOF: on close(), or when
            # the judge process goes away.
            stop_r, stop_w = os.pipe()
            ends += [stop_r, stop_w]
            spec["pipes"] = {"stdin": stdin_r, "stdout": stdout_w, "stop": stop_r}
            (self._control / "spec.json").write_text(json.dumps(spec))
            with open(self._control / "launcher.log", "wb") as log:
                self._process = subprocess.Popen(
                    [sys.executable, "-I", "-S", LAUNCHER, str(self._control)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, env={},
                    pass_fds=(stdin_r, stdout_w, stop_r))
        except BaseException:
            for descriptor in ends:
                os.close(descriptor)
            _remove(self._base)
            raise
        for descriptor in (stdin_r, stdout_w, stop_r):
            os.close(descriptor)
        self._stop = stop_w
        self.stdin = os.fdopen(stdin_w, "wb", buffering=0)
        self.stdout = os.fdopen(stdout_r, "rb", buffering=0)

    def close(self) -> Result:
        """End the worker if it still runs, delete its directories and return its Result."""
        if self._result is not None:
            return self._result
        try:
            os.close(self._stop)
            for stream in (self.stdin, self.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
            try:
                self._process.wait(timeout=KILL_GRACE_SECONDS + 30)
            except subprocess.TimeoutExpired as error:
                self._process.kill()
                self._process.wait()
                raise IsolateError("isolate launcher did not finish") from error
            log = _read_bounded(self._control / "launcher.log", 1 << 20) or b""
            self._result = _collect(self._call, self._control, self._paths,
                                    self._process.returncode, log, self._started)
            return self._result
        finally:
            _remove(self._base)

    def __enter__(self) -> "Instance":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def _read_bounded(path: Path, cap: int) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        chunks, left = [], max(cap, 0)
        while left > 0:
            chunk = os.read(descriptor, min(left, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _remove(base: Path) -> None:
    """Delete a run's directories even where the command removed its own permissions.

    The command left these trees, so nothing in them may make cleanup raise: a failure here is
    never a verification error. Symlinks are unlinked, never followed or chmodded (os.walk lists
    a symlink to a directory among subdirs; chmod(follow_symlinks=False) raises
    NotImplementedError for it, and for every path on a glibc older than 2.32). Every process of
    the run is dead by now, so a path checked as no symlink stays one for the plain chmod.
    """
    for directory, subdirs, _ in os.walk(base, topdown=True):
        for name in list(subdirs):
            path = os.path.join(directory, name)
            try:
                if os.path.islink(path):
                    subdirs.remove(name)
                    os.unlink(path)
                else:
                    os.chmod(path, 0o700)
            except (OSError, NotImplementedError):
                pass
    shutil.rmtree(base, ignore_errors=True)


# ------------------------------------------------------------------ kernel interfaces
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long
SYS_LANDLOCK_CREATE_RULESET, SYS_LANDLOCK_ADD_RULE, SYS_LANDLOCK_RESTRICT_SELF = 444, 445, 446
# Called directly: some judge interpreters (conda builds) lack os.pidfd_open; the kernel has it.
SYS_PIDFD_OPEN = 434
PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, PR_SET_CHILD_SUBREAPER = 38, 22, 36
SECCOMP_MODE_FILTER = 2


def landlock_abi() -> int:
    if sys.platform != "linux":
        return 0
    value = _LIBC.syscall(SYS_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(1))
    return int(value) if value > 0 else 0


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("fs", ctypes.c_uint64), ("net", ctypes.c_uint64), ("scoped", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


FS_EXECUTE, FS_WRITE_FILE, FS_READ_FILE, FS_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
FS_TRUNCATE, FS_IOCTL_DEV = 1 << 14, 1 << 15
FS_FILE_ONLY = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV


def _device_nodes(devices) -> list[str]:
    return sorted(path for name in devices for pattern in DEVICES[name]
                  for path in glob.glob(pattern) if stat.S_ISCHR(os.stat(path).st_mode))


class _NetPort(ctypes.Structure):
    _fields_ = [("allowed", ctypes.c_uint64), ("port", ctypes.c_uint64)]


LANDLOCK_RULE_PATH_BENEATH, LANDLOCK_RULE_NET_PORT = 1, 2


def _landlock_ruleset(read: list[str], write: list[str], devices: list[str],
                      tcp_ports: dict) -> int:
    """A ruleset allowing only these trees, denying TCP bind/connect except to the ports given
    ({port: access}), and scoping signals and abstract sockets to the instance (ABI >= 6).
    Built before fork; the child restricts itself."""
    handled = (1 << 16) - 1                 # ABI 5: execute .. ioctl_dev
    handled_net = NET_BIND_TCP | NET_CONNECT_TCP
    attr = _RulesetAttr(handled, handled_net, 0b11)
    ruleset = _LIBC.syscall(SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr),
                            ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0))
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    read_access = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR
    grants = ([(p, read_access) for p in read] + [(p, handled) for p in write]
              + [(p, FS_READ_FILE | FS_WRITE_FILE | FS_TRUNCATE | FS_IOCTL_DEV)
                 for p in DEVICES_RW]
              + [(p, FS_READ_FILE) for p in DEVICES_RO]
              + [(p, FS_READ_FILE | FS_WRITE_FILE | FS_IOCTL_DEV) for p in devices])
    for path, access in grants:
        try:
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except FileNotFoundError:
            continue  # an absent optional root (e.g. /lib32)
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                access &= FS_FILE_ONLY
            rule = _PathBeneath(access & handled, descriptor)
            if _LIBC.syscall(SYS_LANDLOCK_ADD_RULE, ruleset, LANDLOCK_RULE_PATH_BENEATH,
                             ctypes.byref(rule), 0) != 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
        finally:
            os.close(descriptor)
    for port, access in sorted(tcp_ports.items()):
        rule = _NetPort(access, port)
        if _LIBC.syscall(SYS_LANDLOCK_ADD_RULE, ruleset, LANDLOCK_RULE_NET_PORT,
                         ctypes.byref(rule), 0) != 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule tcp port {port}")
    return ruleset


# x86_64 syscall numbers. Denied in every profile: debugging/inspection of other processes,
# mounts and namespaces, kernel extension points, io_uring (bypasses seccomp for network and
# file ops), keyrings, and the IPC objects that outlive a process (SysV, POSIX mqueue).
SYSCALLS_DENIED = {
    "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311, "kcmp": 312,
    "pidfd_getfd": 438, "mount": 165, "umount2": 166, "pivot_root": 155, "chroot": 161,
    "unshare": 272, "setns": 308, "open_by_handle_at": 304, "name_to_handle_at": 303,
    "fsopen": 430, "fsconfig": 431, "fsmount": 432, "fspick": 433, "move_mount": 429,
    "open_tree": 428, "mount_setattr": 442, "bpf": 321, "perf_event_open": 298,
    "userfaultfd": 323, "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    "keyctl": 250, "add_key": 248, "request_key": 249, "reboot": 169, "kexec_load": 246,
    "kexec_file_load": 320, "init_module": 175, "finit_module": 313, "delete_module": 176,
    "swapon": 167, "swapoff": 168, "acct": 163, "quotactl": 179, "syslog": 103, "shmget": 29,
    "semget": 64, "msgget": 68, "mq_open": 240,
}
# "no_fork": no new processes (fork, vfork, clone without CLONE_THREAD); threads still work.
SYSCALLS_NO_FORK = {"fork": 57, "vfork": 58}
# "strict" adds: no threads either, no signals, no socketpairs (one-process probes).
SYSCALLS_STRICT = {"clone": 56, "fork": 57, "vfork": 58, "kill": 62, "tkill": 200,
                   "tgkill": 234, "socketpair": 53, "rt_sigqueueinfo": 129,
                   "rt_tgsigqueueinfo": 297, "pidfd_send_signal": 424}
SYS_SOCKET, SYS_CLONE, SYS_CLONE3 = 41, 56, 435
SOCK_TYPE_MASK = 0xF   # without SOCK_NONBLOCK / SOCK_CLOEXEC
AF_INET, AF_INET6, AF_NETLINK = 2, 10, 16
AUDIT_ARCH_X86_64 = 0xC000003E
CLONE_NAMESPACES = 0x7E020080  # NEWNS|NEWCGROUP|NEWUTS|NEWIPC|NEWUSER|NEWPID|NEWNET|NEWTIME
CLONE_THREAD = 0x00010000
AF_UNIX = 1
RET_KILL_PROCESS, RET_ALLOW, RET_ERRNO = 0x80000000, 0x7FFF0000, 0x00050000


def seccomp_program(profile: str, unix_sockets: bool, network: str = "none", devices=(),
                    rules=None) -> list[tuple]:
    """The classic-BPF filter as (code, jt, jf, k) tuples (labels resolved); `rules` (Rules)
    adds the layers' denied syscalls and allowed socket families."""
    LD, JEQ, JGE, JSET, RET, AND = 0x20, 0x15, 0x35, 0x45, 0x06, 0x54
    program: list = []
    labels: dict = {}

    def emit(code, k, jt=0, jf=0):
        program.append([code, jt, jf, k])

    emit(LD, 4)                                              # arch
    emit(JEQ, AUDIT_ARCH_X86_64, 0, "kill")
    emit(LD, 0)                                              # nr
    emit(JGE, 0x40000000, "enosys", 0)                       # x32 ABI
    emit(JEQ, SYS_CLONE3, "enosys", 0)                       # libc falls back to clone
    denied = set(SYSCALLS_DENIED.values()) | (rules.denied if rules else set())
    if profile == "no_fork":
        denied |= set(SYSCALLS_NO_FORK.values())
    if profile == "strict":
        denied |= set(SYSCALLS_STRICT.values())
    for number in sorted(denied):
        emit(JEQ, number, "eperm", 0)
    emit(JEQ, SYS_SOCKET, "socket", 0)
    if profile != "strict":
        emit(JEQ, SYS_CLONE, "clone", 0)
    emit(RET, RET_ALLOW)
    labels["clone"] = len(program)
    emit(LD, 16)                                             # clone flags (arg 0, low word)
    if profile == "no_fork":
        emit(JSET, CLONE_NAMESPACES, "eperm", 0)
        emit(JSET, CLONE_THREAD, "allow", "eperm")
    else:
        emit(JSET, CLONE_NAMESPACES, "eperm", "allow")
    labels["socket"] = len(program)
    emit(LD, 16)                                             # socket family (arg 0)
    families = ([AF_UNIX] if unix_sockets else []) + (
        [AF_INET, AF_INET6, AF_NETLINK] if network == "loopback" else [])
    families += [globals()[name] for device in devices for name in DEVICE_SOCKETS.get(device, ())]
    for family in sorted(set(families)):   # loopback: only in userns's empty network namespace
        emit(JEQ, family, "allow", 0)
    layered = sorted((rules.sockets if rules else {}).items())
    for family, _ in layered:
        emit(JEQ, family, f"family{family}", 0)
    emit(RET, RET_ERRNO | errno.EAFNOSUPPORT)
    for family, (kind, protocols) in layered:
        labels[f"family{family}"] = len(program)
        emit(LD, 24)                                         # type (arg 1)
        emit(AND, SOCK_TYPE_MASK)
        emit(JEQ, kind, 0, "eafnosupport")
        emit(LD, 32)                                         # protocol (arg 2)
        for protocol in protocols:
            emit(JEQ, protocol, "allow", 0)
        emit(RET, RET_ERRNO | errno.EAFNOSUPPORT)
    for name, value in (("allow", RET_ALLOW), ("eperm", RET_ERRNO | errno.EPERM),
                        ("eafnosupport", RET_ERRNO | errno.EAFNOSUPPORT),
                        ("enosys", RET_ERRNO | errno.ENOSYS), ("kill", RET_KILL_PROCESS)):
        labels[name] = len(program)
        emit(RET, value)
    resolved = []
    for index, (code, jt, jf, k) in enumerate(program):
        jt = labels[jt] - index - 1 if isinstance(jt, str) else jt
        jf = labels[jf] - index - 1 if isinstance(jf, str) else jf
        if not (0 <= jt <= 255 and 0 <= jf <= 255):
            raise IsolateError("seccomp program jump out of range")
        resolved.append((code, jt, jf, k))
    return resolved


class _Fprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def _seccomp_filter(profile: str, unix_sockets: bool, network: str, devices, rules) -> tuple:
    """(buffer, sock_fprog) for PR_SET_SECCOMP; the buffer must outlive the call."""
    program = seccomp_program(profile, unix_sockets, network, devices, rules)
    blob = b"".join(struct.pack("HBBI", *instruction) for instruction in program)
    buffer = ctypes.create_string_buffer(blob, len(blob))
    return buffer, _Fprog(len(program), ctypes.addressof(buffer))


def _confine(ruleset: int | None, fprog: _Fprog) -> None:
    """In the child: no_new_privs, the Landlock domain (landlock backend), then the seccomp
    filter. For good."""
    if _LIBC.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "no_new_privs")
    if ruleset is not None:
        if _LIBC.syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
        os.close(ruleset)
    if _LIBC.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "seccomp")


# ------------------------------------------------------------------ launcher (own process)
def _descendants(root: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                fields = handle.read().rsplit(b")", 1)[1].split()
        except (OSError, IndexError):
            continue
        children.setdefault(int(fields[1]), []).append(int(entry))
    found, stack = [], [root]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def _resident_bytes(pids: list[int]) -> int:
    total = 0
    page = os.sysconf("SC_PAGE_SIZE")
    for pid in pids:
        try:
            with open(f"/proc/{pid}/statm", "rb") as handle:
                total += int(handle.read().split()[1]) * page
        except (OSError, IndexError, ValueError):
            continue
    return total


def _disk_bytes(root: str) -> int:
    total = 0
    for directory, _, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_blocks * 512
            except OSError:
                continue
    return total


def _memory_bytes(shm: bool) -> int:
    pids = _descendants(os.getpid())
    return _resident_bytes(pids) + (_shm_bytes(pids[1:]) if shm else 0)


def _kill_all() -> None:
    """SIGKILL every descendant (orphans re-parent to us, the subreaper) until none is left."""
    deadline = time.monotonic() + KILL_GRACE_SECONDS
    while True:
        pids = _descendants(os.getpid())
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
        if time.monotonic() > deadline:
            raise IsolateError("instance processes survived SIGKILL")
        time.sleep(0.01)


def _rlimits(options: dict) -> None:
    limits = {resource.RLIMIT_CORE: (0, 0),
              resource.RLIMIT_NOFILE: (options["open_files"],) * 2,
              resource.RLIMIT_NPROC: (options["processes"],) * 2}
    if options["file_bytes"] is not None:
        limits[resource.RLIMIT_FSIZE] = (options["file_bytes"],) * 2
    if options["cpu_seconds"] is not None:
        limits[resource.RLIMIT_CPU] = (options["cpu_seconds"], options["cpu_seconds"] + 2)
    for kind, (soft, hard) in limits.items():
        _, current = resource.getrlimit(kind)
        if current != resource.RLIM_INFINITY:
            soft, hard = min(soft, current), min(hard, current)
        resource.setrlimit(kind, (soft, hard))


def _prepare(launch: dict) -> dict:
    """Everything the child needs, built before fork so the timed command starts promptly."""
    options = launch["options"]
    control = Path(launch["control"])
    paths = launch["paths"]
    read = [*SYSTEM_READ, *launch["interpreter"], *options["read_paths"], paths["in"]]
    write = [paths["work"], paths["out"], paths["home"], paths["tmp"]]
    if options["dev_shm"]:
        write.append("/dev/shm")
    layers = _load_layers(options["layers"])
    rules = Rules(launch["backend"], ())
    for layer in layers:
        own = Rules(launch["backend"], layer.relaxes)
        layer.prepare(own)
        rules.denied |= own.denied
        rules.sockets.update(own.sockets)
        rules.fds |= own.fds
        for port, access in own.tcp_ports.items():
            rules.tcp_ports[port] = rules.tcp_ports.get(port, 0) | access
    # DEVICE_SOCKETS only under userns, where the instance's own network namespace and the
    # covered roots leave no judge socket to reach (Landlock ABI 6 does not mediate connect()
    # to a pathname socket, and the landlock hosts have no GPU).
    buffer, fprog = _seccomp_filter(
        options["syscalls"], options["unix_sockets"], options["network"],
        options["devices"] if launch["backend"] == "userns" else (), rules)
    pipes = launch["pipes"]
    if pipes:   # a started worker: stdin and stdout are the judge's pipes
        stdio = [pipes["stdin"], pipes["stdout"]]
    else:
        stdio = [os.open(control / "stdin", os.O_RDONLY),
                 os.open(control / "stdout", os.O_WRONLY | os.O_TRUNC)]
    stdio.append(os.open(control / "stderr", os.O_WRONLY | os.O_TRUNC))
    ruleset = None
    if launch["backend"] == "landlock":
        ruleset = _landlock_ruleset(read, write, _device_nodes(options["devices"]),
                                    rules.tcp_ports)
    elif rules.tcp_ports:
        raise IsolateError("TCP port rules need the landlock backend")
    return {"ruleset": ruleset, "buffer": buffer, "fprog": fprog, "stdio": stdio,
            "layers": layers, "fds": rules.fds,
            "executable": _which(launch["argv"][0], launch["env"]["PATH"])}  # None: exit 127


def _load_layers(specs: list[dict]) -> list[Layer]:
    """In the launcher: rebuild the judge's layers from their files and attributes."""
    sys.modules.setdefault("isolate", sys.modules[__name__])   # for `import isolate`
    layers = []
    for index, spec in enumerate(specs):
        if os.path.realpath(spec["file"]) == os.path.realpath(LAUNCHER):   # a layer of this file
            # (executing this file a second time would break its dataclasses)
            module = sys.modules[__name__]
        else:
            found = importlib.util.spec_from_file_location(f"_isolate_layer{index}",
                                                           spec["file"])
            module = importlib.util.module_from_spec(found)
            found.loader.exec_module(module)
        layer = object.__new__(getattr(module, spec["class"]))
        layer.__dict__.update(spec["state"])
        layers.append(layer)
    return layers


def _child(launch: dict, ready: dict) -> None:
    """In the forked child: enter the boundary, then exec. Never returns. A setup failure is
    written to the close-on-exec pipe ready["errpipe"]; a missing command is exit 127."""
    errpipe, executing = ready["errpipe"], False
    try:
        os.sched_setaffinity(0, ready["affinity"])
        os.setsid()
        for source, target in zip(ready["stdio"], (0, 1, 2)):
            os.dup2(source, target)
        ruleset = ready["ruleset"]
        low = 3   # closes the judge's stop pipe of a started worker too
        kept = (ruleset, ready["status"], errpipe, *ready["fds"])
        for keep in sorted(fd for fd in kept if fd is not None):
            os.closerange(low, keep)
            low = keep + 1
        os.closerange(low, 65536)
        for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):  # Python ignores these; exec keeps that
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), signal.SIG_DFL)
        _rlimits(launch["options"])
        for layer in ready["layers"]:
            layer.preexec()
        if launch["backend"] == "userns":
            _userns_enter(launch)
            _userns_init(ready["status"], errpipe)   # returns in the command's process only
        os.chdir(launch["cwd"])
        _confine(ruleset, ready["fprog"])
        executing = True
        if ready["executable"] is None:
            raise FileNotFoundError(errno.ENOENT, "command not found")
        os.execve(ready["executable"], launch["argv"], launch["env"])
    except BaseException as error:  # noqa: BLE001 - report and die; never run unsandboxed
        try:
            if executing and isinstance(error, FileNotFoundError):
                os.write(2, f"{launch['argv'][0]}: {error.strerror}\n".encode())
                os._exit(127)
            os.write(errpipe, f"{type(error).__name__}: {error}".encode())
        finally:
            os._exit(1)


CLONE_NEWNS, CLONE_NEWUSER, CLONE_NEWPID, CLONE_NEWNET, CLONE_NEWIPC = (
    0x00020000, 0x10000000, 0x20000000, 0x40000000, 0x08000000)
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_REMOUNT, MS_BIND, MS_REC, MS_PRIVATE = (
    1, 2, 4, 8, 32, 4096, 16384, 1 << 18)
PR_CAPBSET_DROP, PR_SET_DUMPABLE = 24, 4


def _mount(source, target, fstype, flags, data=None) -> None:
    encode = (lambda text: text.encode() if text is not None else None)
    if _LIBC.mount(encode(source), encode(target), encode(fstype), ctypes.c_ulong(flags),
                   encode(data)) != 0:
        raise OSError(ctypes.get_errno(), f"mount {target}")


def _write(path: str, text: str) -> None:
    with open(path, "w") as handle:
        handle.write(text)


def _userns_unshare() -> None:
    """In the launcher: new user (mapped to root), network, IPC and PID namespaces; the PID
    namespace is the child's (its first process), and so is the mount namespace (_userns_enter:
    the launcher keeps seeing the control directory)."""
    uid, gid = os.getuid(), os.getgid()
    if _LIBC.unshare(CLONE_NEWUSER | CLONE_NEWNET | CLONE_NEWIPC | CLONE_NEWPID) != 0:
        raise OSError(ctypes.get_errno(), "unshare")
    for name, text in (("setgroups", "deny"), ("uid_map", f"0 {uid} 1\n"),
                       ("gid_map", f"0 {gid} 1\n")):
        _write(f"/proc/self/{name}", text)


def _userns_accounts(control: Path, home) -> list[tuple[str, str]]:
    """The instance's /etc/passwd and /etc/group: the image's own, plus USERNS_USER for
    USERNS_UID (home: the instance's HOME). Written to the launcher's control directory and
    bound read-only over the image's files; returns (source, target) pairs."""
    pairs = []
    for name, entry in (("passwd", f"{USERNS_USER}:x:{USERNS_UID}:{USERNS_UID}::{home}:/bin/sh"),
                        ("group", f"{USERNS_USER}:x:{USERNS_UID}:")):
        target = f"/etc/{name}"
        lines = Path(target).read_text().splitlines() if os.path.exists(target) else []
        lines = [line for line in lines if line.split(":")[2:3] != [str(USERNS_UID)]]
        source = control / f"instance-{name}"
        source.write_text("\n".join([*lines, entry]) + "\n")
        os.chmod(source, 0o444)
        pairs.append((str(source), target))
    return pairs


def _userns_enter(launch: dict) -> None:
    """In the child, as root of the new user namespace: cover judge state, bind the instance's
    own directories and read paths back, mask unrequested devices, mount its own /proc, then
    become USERNS_UID in a nested user namespace without capabilities (the mounts above are
    locked to it)."""
    options, paths = launch["options"], launch["paths"]
    hidden = launch["hidden"]
    covered = (lambda path: any(_within(path, root) for root in hidden))
    accounts = _userns_accounts(Path(launch["control"]), paths["home"])
    # (source, read_only): what the instance sees again at the same path once covered
    exposed = [(launch["root"], False), (paths["in"], True)]
    exposed += [(path, True) for path in (*options["read_paths"], *launch["interpreter"])
                if covered(path)]
    if _LIBC.unshare(CLONE_NEWNS) != 0:
        raise OSError(ctypes.get_errno(), "mount namespace")
    _mount(None, "/", None, MS_REC | MS_PRIVATE)
    handles = [(os.open(path, os.O_PATH | os.O_CLOEXEC), path, read_only)
               for path, read_only in exposed]
    handles += [(os.open(source, os.O_PATH | os.O_CLOEXEC), target, True)
                for source, target in accounts]
    try:  # where the kernel allows it; the covered roots are what holds judge state
        _mount(None, "/", None, MS_BIND | MS_REMOUNT | MS_RDONLY)
    except OSError:
        pass
    for root in hidden:
        size = USERNS_SHM_BYTES if root == "/dev/shm" and options["dev_shm"] else 1 << 20
        _mount("tmpfs", root, "tmpfs", MS_NOSUID | MS_NODEV, f"mode=0755,size={size}")
    for descriptor, path, read_only in handles:
        if os.path.isdir(f"/proc/self/fd/{descriptor}"):
            os.makedirs(path, exist_ok=True)
        elif not os.path.lexists(path):   # a mount point (an existing one may be read-only)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "a").close()
        _mount(f"/proc/self/fd/{descriptor}", path, None, MS_BIND | MS_REC)
        _mount(None, path, None, MS_BIND | MS_REMOUNT | MS_NOSUID | MS_NODEV
               | (MS_RDONLY if read_only else 0))
        os.close(descriptor)
    for root in hidden:
        if not (root == "/dev/shm" and options["dev_shm"]):
            _mount(None, root, None, MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV)
    for name, patterns in DEVICES.items():
        if name not in options["devices"]:
            for path in sorted(p for pattern in patterns for p in glob.glob(pattern)):
                if not os.path.isdir(path):
                    _mount("/dev/null", path, None, MS_BIND)
    _mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    if options["network"] == "loopback":
        _loopback_up()
    if _LIBC.unshare(CLONE_NEWUSER) != 0:
        raise OSError(ctypes.get_errno(), "nested user namespace")
    for name, text in (("setgroups", "deny"), ("uid_map", f"{USERNS_UID} 0 1\n"),
                       ("gid_map", f"{USERNS_UID} 0 1\n")):
        _write(f"/proc/self/{name}", text)
    for capability in range(64):
        _LIBC.prctl(PR_CAPBSET_DROP, capability, 0, 0, 0)
    # execve as a non-root uid clears the permitted and effective capability sets


def _userns_init(status_fd: int, errpipe: int) -> None:
    """The first process of the instance's PID namespace stays a minimal init, so the command
    keeps ordinary signal semantics (a namespace init ignores SIGPIPE, SIGXFSZ, SIGXCPU, ...).
    It forks the command, waits for it, sends its raw wait status to the launcher and exits,
    which ends the namespace and every process left in it."""
    pid = os.fork()
    if pid == 0:
        os.close(status_fd)
        return
    for descriptor in (0, 1, 2, errpipe):
        os.close(descriptor)
    _LIBC.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    while True:
        try:
            child, wait_status = os.waitpid(-1, 0)
        except ChildProcessError:
            os._exit(1)
        except InterruptedError:
            continue
        if child == pid:
            os.write(status_fd, str(wait_status).encode())
            os._exit(0)


def _loopback_up() -> None:
    """Bring up the namespace's lo (gVisor creates it up and refuses SIOCSIFFLAGS)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        request = struct.pack("16sH14s", b"lo", 0, b"")
        flags = struct.unpack("16sH14s", fcntl.ioctl(sock.fileno(), 0x8913, request))[1]
        if not flags & 0x1:   # IFF_UP
            fcntl.ioctl(sock.fileno(), 0x8914, struct.pack("16sH14s", b"lo", flags | 0x1 | 0x40,
                                                           b""))


def _shm_bytes(pids: list[int]) -> int:
    """userns backend: bytes the instance holds in its private /dev/shm tmpfs, seen through
    the root of any of its processes (the init is not dumpable, so not that one)."""
    for pid in pids:
        try:
            info = os.statvfs(f"/proc/{pid}/root/dev/shm")
        except OSError:
            continue
        return (info.f_blocks - info.f_bfree) * info.f_frsize
    return 0


def _which(command: str, path: str) -> str | None:
    if "/" in command:
        return command
    for directory in path.split(":"):
        candidate = os.path.join(directory, command)
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def _launcher(control: Path) -> int:
    launch = json.loads((control / "spec.json").read_text())
    options = launch["options"]
    status = {"returncode": None, "limit": None, "error": None, "command_ns": 0, "rusage": {}}
    userns = launch["backend"] == "userns"
    try:
        if not userns and landlock_abi() < MIN_LANDLOCK_ABI:
            raise OSError(errno.ENOSYS, f"Landlock ABI {landlock_abi()} < {MIN_LANDLOCK_ABI}")
        if _LIBC.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "child subreaper")
        if userns:
            _userns_unshare()
        deadline = time.monotonic() + options["timeout_seconds"]
        memory = options["memory_bytes"]
        ready = _prepare(launch)
        # The CPUs the judge runs on are the command's (a timed judge pins itself to one). The
        # launcher leaves them, so its own work never shares the command's CPU.
        ready["affinity"] = os.sched_getaffinity(0)
        os.sched_setaffinity(0, range(os.cpu_count()))
        command_status, ready["status"] = os.pipe() if userns else (None, None)
        errpipe_read, ready["errpipe"] = os.pipe()   # close-on-exec: EOF once the command runs
        pid = os.fork()
        if pid == 0:
            _child(launch, ready)
        for descriptor in (ready["ruleset"], ready["status"], ready["errpipe"], *ready["stdio"]):
            if descriptor is not None:
                os.close(descriptor)
        with os.fdopen(errpipe_read, "rb") as errpipe:
            failure = errpipe.read(50000)
            # EOF without a report: the command's execve succeeded. command_seconds starts here,
            # so the fork and the instance setup in the child are never inside it.
            started = time.perf_counter_ns()
        if failure:
            raise IsolateError(f"instance setup failed: {failure.decode(errors='replace')}")
        exited = _LIBC.syscall(SYS_PIDFD_OPEN, pid, 0)   # readable when the command exits
        if exited < 0:
            raise OSError(ctypes.get_errno(), "pidfd_open")
        stop = launch["pipes"]["stop"] if launch["pipes"] else None
        # userns: the init reports the command's own wait status the moment it exits
        watched = [fd for fd in (exited, stop, command_status) if fd is not None]
        reported = b""
        next_disk = time.monotonic() + DISK_POLL_SECONDS

        def reap(flags: int) -> tuple[int, int]:
            """waitpid on the command, keeping its resource usage (for Layer.result)."""
            reaped, wait_status, usage = os.wait4(pid, flags)
            if reaped:
                status["rusage"] = {name: getattr(usage, name) for name in dir(usage)
                                    if name.startswith("ru_")}
            return reaped, wait_status

        while True:
            # Wait first: the exit is stamped the moment the pidfd turns readable.
            now = time.monotonic()
            wait = min(POLL_SECONDS, max(0.0, deadline - now))
            ready_fds = select.select(watched, [], [], wait)[0]
            if command_status is not None and stop in ready_fds \
                    and command_status not in ready_fds:   # (the judge may see EOF first)
                ready_fds += select.select([command_status], [], [], POLL_SECONDS)[0]
            if command_status is not None and command_status in ready_fds:
                ended = time.perf_counter_ns()
                reported = os.read(command_status, 64)
                done, wait_status = True, reap(0)[1]
            else:
                done = exited in ready_fds
                ended = time.perf_counter_ns()
                wait_status = reap(0)[1] if done else None
            if done:
                status["returncode"] = os.waitstatus_to_exitcode(wait_status)
                break
            now = time.monotonic()
            if stop in ready_fds:   # the judge closed the instance (or is gone): end it
                os.kill(pid, signal.SIGKILL)
                _, wait_status = reap(0)
                ended = time.perf_counter_ns()
                status["returncode"] = os.waitstatus_to_exitcode(wait_status)
                break
            if now >= deadline:
                status["limit"] = "time"
            elif memory is not None and _memory_bytes(userns and options["dev_shm"]) > memory:
                status["limit"] = "memory"
            elif any((control / name).stat().st_size > options["output_bytes"]
                     for name in ("stdout", "stderr")):
                status["limit"] = "output"
            elif now >= next_disk:
                next_disk = now + DISK_POLL_SECONDS
                if _disk_bytes(launch["root"]) > options["disk_bytes"]:
                    status["limit"] = "disk"
            if status["limit"]:
                os.kill(pid, signal.SIGKILL)
                _, wait_status = reap(0)
                ended = time.perf_counter_ns()
                status["returncode"] = os.waitstatus_to_exitcode(wait_status)
                break
        status["command_ns"] = ended - started
        os.close(exited)
        if stop is not None:
            os.close(stop)
        if command_status is not None:   # userns: the command's own status, from the init
            reported = reported or os.read(command_status, 64)
            os.close(command_status)
            if reported and status["limit"] is None:
                status["returncode"] = os.waitstatus_to_exitcode(int(reported))
        _kill_all()
    except Exception as error:  # noqa: BLE001 - reported to the judge as an infra fault
        status["error"] = f"{type(error).__name__}: {error}"
        try:
            _kill_all()
        except Exception:  # noqa: BLE001
            pass
    if launch.get("exec"):   # exec_on_stdio: no judge collects; end as the command did
        _remove(control.parent)
        if status["error"]:
            _fail_out_of_band(launch["exec"]["failure_record"], status["error"])
        _end_as(status["returncode"])
    (control / "status.json").write_text(json.dumps(status))
    return 0


def _end_as(returncode: int) -> None:
    """exec_on_stdio: exit with the command's code, or die by the signal that ended it."""
    if returncode >= 0:
        os._exit(returncode)
    number = -returncode
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if number not in (signal.SIGKILL, signal.SIGSTOP):
        signal.signal(number, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, [number])
    os.kill(os.getpid(), number)
    os._exit(128 + number)   # a signal whose default action does not end a process


if __name__ == "__main__":
    raise SystemExit(_launcher(Path(sys.argv[1])))
