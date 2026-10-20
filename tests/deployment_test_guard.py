"""Accidental-side-effect guard for tests, not an OS security sandbox."""

from __future__ import annotations

import atexit
import fcntl
import functools
import ipaddress
import json
import os
import shlex
import shutil
import socket
import sys
import tempfile
import threading
from pathlib import Path

# Direct unittest discovery must also preserve dynamically imported source trees.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
STATE_PATHS_ENV = "_PREVIEW_TEST_STATE_PATHS"
STATE_PATHS = tuple(
    dict.fromkeys(
        [
            ROOT / ".state",
            Path("/root/FrontierCS-2.0-Preview/.state"),
            *(Path(path) for path in json.loads(os.environ.get(STATE_PATHS_ENV, "[]"))),
        ]
    )
)
TOOLS = frozenset(
    {
        "kubectl",
        "terraform",
        "docker",
        "aws",
        "helm",
        "ssh",
        "scp",
        "curl",
        "wget",
        "crane",
        "git-push",
        "git-receive-pack",
    }
)
SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh", "ash"})
GIT_NETWORK_VERBS = frozenset({"push", "fetch", "clone"})
GIT_VALUE_OPTIONS = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}
)
STATE_ERROR = "test isolation: write into checkout state"
NETWORK_ERROR = "test isolation: external network"
_open_context = threading.local()
_original_open = os.open


def descriptor_path(descriptor):
    """Resolve the opened object itself; a guessed path could bypass the guard."""
    try:
        if sys.platform == "darwin":
            # Darwin has no procfs. F_GETPATH returns a MAXPATHLEN-sized buffer.
            value = fcntl.fcntl(
                descriptor, getattr(fcntl, "F_GETPATH", 50), bytes(1024)
            )
            path = os.fsdecode(value.split(b"\0", 1)[0])
            if not os.path.isabs(path):
                raise ValueError("descriptor did not resolve to an absolute path")
            return path
        path = f"/proc/self/fd/{descriptor}"
        os.fstat(descriptor)
        if not os.path.islink(path):
            raise ValueError("descriptor path resolution is unavailable")
        return path
    except (OSError, ValueError) as error:
        raise RuntimeError(STATE_ERROR) from error


def resolved_path(value, dir_fd=None, *, follow_final=True):
    if isinstance(value, int):
        value = descriptor_path(value)
    elif not isinstance(value, (str, bytes, os.PathLike)):
        return None
    path = os.fsdecode(value)
    if not os.path.isabs(path):
        base = os.getcwd() if dir_fd in (None, -1) else descriptor_path(dir_fd)
        path = os.path.join(base, path)
    if follow_final:
        return os.path.realpath(path)
    parent, name = os.path.split(path.rstrip(os.sep))
    return os.path.join(os.path.realpath(parent), name)


def check_state(value, dir_fd=None, *, entry=False):
    path = resolved_path(value, dir_fd)
    if path is None:
        return
    paths = {path}
    if entry:
        paths.add(resolved_path(value, dir_fd, follow_final=False))
    # Keep both the installation-time identity and the current symlink target.
    roots = {*_state_roots, *(os.path.realpath(p) for p in STATE_PATHS)}
    for root in roots:
        if any(path == root or path.startswith(root + os.sep) for path in paths):
            raise RuntimeError(STATE_ERROR)


def writes(mode, flags):
    return (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
        isinstance(flags, int)
        and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)
        )
    )


def shell_commands(value):
    text = os.fsdecode(value)
    # Recognize ordinary command separators, without evaluating shell syntax.
    try:
        # Keep quotes until after separators are recognized: echo ';' aws is
        # one command, and aws is just an argument.
        lexer = shlex.shlex(text, posix=False, punctuation_chars=";&|\n")
        lexer.whitespace = " \t\r"  # A newline also starts another command.
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        words = text.split()
    command = []
    for word in words:
        if word and all(char in ";&|\n" for char in word):
            if command:
                yield command
            command = []
        else:
            try:
                command.append("".join(shlex.split(word)))
            except ValueError:
                command.append(word)
    if command:
        yield command


def git_verb(arguments):
    arguments = iter(arguments)
    for arg in arguments:
        if arg in {"--version", "--help", "-v", "-h"}:
            return None
        if arg in GIT_VALUE_OPTIONS:
            next(arguments, None)
        elif not arg.startswith("-"):
            return arg
    return None


def check_argv(argv):
    if not argv:
        return
    tool = os.path.basename(argv[0].rstrip("/"))
    if tool in TOOLS or (tool == "git" and git_verb(argv[1:]) in GIT_NETWORK_VERBS):
        raise RuntimeError(f"test isolation: {tool} is not available under tests")
    if tool == "env":
        # env's first non-option, non-assignment argument is a command too.
        arguments = iter(argv[1:])
        for arg in arguments:
            if arg in {"-u", "--unset", "-C", "--chdir"}:
                next(arguments, None)
            elif not arg.startswith("-") and "=" not in arg:
                check_argv([arg, *arguments])
                break
    elif tool in SHELLS:
        for index, arg in enumerate(argv[1:], 1):
            if arg in {"-c", "-lc", "-ic", "--command"} and index + 1 < len(argv):
                for command in shell_commands(argv[index + 1]):
                    check_argv(command)
                break


def check_command(executable, argv):
    if isinstance(argv, (str, bytes)):
        if not executable:  # os.system passes a shell string directly.
            for command in shell_commands(argv):
                check_argv(command)
            return
        argv = [argv]
    words = [os.fsdecode(arg) for arg in (argv or ())]
    check_argv(words)
    if executable:
        # Popen/exec can select an executable different from argv[0].
        check_argv([os.fsdecode(executable), *words[1:]])


def check_host(host):
    if isinstance(host, bytes):
        host = os.fsdecode(host)
    if host in ("localhost", "127.0.0.1", "::1"):
        return
    try:
        address = ipaddress.ip_address(host)
    except (ValueError, TypeError):
        raise RuntimeError(NETWORK_ERROR) from None
    if not address.is_loopback:
        raise RuntimeError(NETWORK_ERROR)


def guard(event, args):
    if event == "open":
        if writes(args[1], args[2]):
            check_state(args[0], getattr(_open_context, "dir_fd", None))
    elif event == "test_isolation.open":
        if writes(None, args[1]):
            check_state(args[0], args[2])
    elif event in {"os.rename", "os.link"}:
        check_state(args[0], args[2], entry=True)
        check_state(args[1], args[3], entry=True)
    elif event == "os.symlink":
        # The destination is written; the symlink's contents are not modified.
        check_state(args[1], args[2], entry=True)
    elif event in {"os.remove", "os.rmdir"}:
        check_state(args[0], args[1], entry=True)
    elif event in {"os.mkdir", "os.chmod", "os.utime"}:
        check_state(args[0], args[2] if event != "os.utime" else args[3])
    elif event == "os.truncate":
        check_state(args[0])
    elif event in {"tempfile.mkstemp", "tempfile.mkdtemp"}:
        check_state(args[0])
    elif event in {"subprocess.Popen", "os.exec", "os.posix_spawn"}:
        check_command(args[0], args[1])
    elif event == "os.system":
        check_command("", args[0])
    elif event in {"socket.connect", "socket.sendto", "socket.sendmsg"}:
        if args[0].family == socket.AF_UNIX:
            return
        address = args[1]
        # sendmsg without a destination uses the already-checked connection.
        if address is not None:
            check_host(address[0] if isinstance(address, tuple) else address)
    elif event in {
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
    }:
        check_host(args[0])


@functools.wraps(_original_open)
def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
    # CPython's built-in open audit event omits dir_fd. Supply it before the
    # operation, then retain it for the built-in event (including read opens).
    sys.audit("test_isolation.open", path, flags, dir_fd)
    previous = getattr(_open_context, "dir_fd", None)
    _open_context.dir_fd = dir_fd
    try:
        return _original_open(path, flags, mode, dir_fd=dir_fd)
    finally:
        _open_context.dir_fd = previous


def cleanup_stubs(directory):
    try:
        shutil.rmtree(directory)
    except Exception:
        # Cleanup must not turn an interpreter exit into a test failure.
        pass


def install():
    if getattr(sys, "_preview_test_guard_installed", False):
        return
    sys._preview_test_guard_installed = True
    sys.addaudithook(guard)
    os.open = guarded_open
    directory = Path(tempfile.mkdtemp(prefix="preview-test-guard-"))
    atexit.register(cleanup_stubs, directory)
    real_git = shutil.which("git")
    for tool in sorted(TOOLS):
        stub = directory / tool
        stub.write_text(
            "#!/bin/sh\n"
            f'echo "test isolation: {tool} is not available under tests" >&2\n'
            "exit 99\n"
        )
        stub.chmod(0o755)
    # Use the same verb parser in the PATH layer; local Git commands still work.
    git_stub = directory / "git"
    git_stub.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'tests')!r})\n"
        "from deployment_test_guard import git_verb, GIT_NETWORK_VERBS\n"
        "if git_verb(sys.argv[1:]) in GIT_NETWORK_VERBS:\n"
        '    print("test isolation: git is not available under tests", file=sys.stderr)\n'
        "    sys.exit(99)\n"
        f"git_executable = {real_git!r}\n"
        "if git_executable is None:\n"
        "    sys.exit(127)\n"
        "sys.exit(subprocess.call([git_executable, *sys.argv[1:]]))\n"
    )
    git_stub.chmod(0o755)
    # A child gets a fresh installation and its own temporary bootstrap. Keep
    # this parent directory until interpreter exit so descendants can import it.
    (directory / "sitecustomize.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        f"guard_directory = Path({str(ROOT / 'tests')!r})\n"
        "if (Path.cwd() / 'tests/deployment_test_guard.py').is_file():\n"
        "    guard_directory = Path.cwd() / 'tests'\n"
        "sys.path.insert(0, str(guard_directory))\n"
        "import deployment_test_guard\n"
    )
    # A copied repository adds its own state root without losing its parent's.
    os.environ[STATE_PATHS_ENV] = json.dumps(
        list(dict.fromkeys([*(str(path) for path in STATE_PATHS), *_state_roots]))
    )
    os.environ["PATH"] = (
        str(directory) + os.pathsep + os.environ.get("PATH", os.defpath)
    )
    os.environ["PYTHONPATH"] = (
        str(directory) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )


_state_roots = tuple(os.path.realpath(path) for path in STATE_PATHS)
# Both unittest package names and direct test scripts use the same instance.
sys.modules.setdefault("deployment_test_guard", sys.modules[__name__])
for directory in (ROOT / "tests", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
install()
