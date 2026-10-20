"""Exercise real audit events, with a second hook preventing unsafe regressions."""

from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import ast
import contextlib
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
STATE_ERROR = "test isolation: write into checkout state"
NETWORK_ERROR = "test isolation: external network"
WRITE_EVENTS = {
    "open",
    "test_isolation.open",
    "os.rename",
    "os.remove",
    "os.mkdir",
    "os.rmdir",
    "os.chmod",
    "os.utime",
    "os.link",
    "os.symlink",
    "os.truncate",
    "tempfile.mkstemp",
    "tempfile.mkdtemp",
}
COMMAND_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}
NETWORK_EVENTS = {
    "socket.connect",
    "socket.sendto",
    "socket.sendmsg",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
}
_probe_events = set()


def secondary_guard(event, args):
    if event == "open":
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int)
            and flags
            & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if not writing:
            return
    if event in _probe_events:
        raise AssertionError("primary guard missed audit event: " + event)


sys.addaudithook(secondary_guard)


class GuardTests(unittest.TestCase):
    def test_darwin_descriptor_path_preserves_state_protection(self):
        path = os.fsencode(ROOT) + b"\0"
        response = path.ljust(1024, b"\0")
        with (
            mock.patch.object(deployment_test_guard.sys, "platform", "darwin"),
            mock.patch.object(
                deployment_test_guard.fcntl, "fcntl", return_value=response
            ) as query,
        ):
            self.assertEqual(
                deployment_test_guard.resolved_path(".state/probe", dir_fd=42),
                str(ROOT / ".state/probe"),
            )
            with self.assertRaisesRegex(RuntimeError, STATE_ERROR):
                deployment_test_guard.check_state(".state/probe", dir_fd=42)
            self.assertEqual(query.call_args.args[0], 42)
            self.assertEqual(len(query.call_args.args[2]), 1024)

    def test_darwin_descriptor_resolution_failure_is_closed(self):
        with mock.patch.object(deployment_test_guard.sys, "platform", "darwin"):
            for reply in (b"relative\0", b"\0" * 1024):
                with mock.patch.object(
                    deployment_test_guard.fcntl, "fcntl", return_value=reply
                ):
                    with self.assertRaisesRegex(RuntimeError, STATE_ERROR):
                        deployment_test_guard.check_state("probe", dir_fd=42)
            with mock.patch.object(
                deployment_test_guard.fcntl, "fcntl", side_effect=OSError("invalid fd")
            ):
                with self.assertRaisesRegex(RuntimeError, STATE_ERROR):
                    deployment_test_guard.check_state(42)

    def test_stub_directories_are_removed_after_child_and_grandchild_exit(self):
        code = """
import os, subprocess, sys
print(os.environ['PYTHONPATH'].split(os.pathsep)[0], flush=True)
subprocess.run([sys.executable, '-c',
    "import os; print(os.environ['PYTHONPATH'].split(os.pathsep)[0])"], check=True)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        paths = [Path(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(set(paths)), 2)
        for path in paths:
            self.assertTrue(path.name.startswith("preview-test-guard-"))
            self.assertFalse(path.exists(), str(path))

    def test_stub_cleanup_is_best_effort(self):
        for error in (
            OSError("cleanup failed"),
            RuntimeError("audit rejected cleanup"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                mock.patch.object(shutil, "rmtree", side_effect=error),
            ):
                deployment_test_guard.cleanup_stubs("/unused/stub-directory")

    def blocked(self, operation, message, events):
        # A broken primary hook must fail the test before a syscall can happen.
        _probe_events.update(events)
        try:
            with self.assertRaisesRegex(RuntimeError, "^" + message + "$"):
                operation()
        finally:
            _probe_events.clear()

    def test_state_write_modes_absolute_relative_and_bytes(self):
        for root in deployment_test_guard.STATE_PATHS:
            for target in (root / "probe", os.fsencode(root / "probe")):
                for mode in ("w", "a", "x", "r+", "w+b", "a+b"):
                    with self.subTest(target=target, mode=mode):
                        self.blocked(
                            lambda *, mode=mode, target=target: open(target, mode),
                            STATE_ERROR,
                            WRITE_EVENTS,
                        )
        with contextlib.chdir(ROOT):
            self.blocked(
                lambda: Path(".state/probe").write_text("probe"),
                STATE_ERROR,
                WRITE_EVENTS,
            )
        with contextlib.chdir(ROOT / "tests"):
            self.blocked(
                lambda: Path("../.state/probe").write_bytes(b"probe"),
                STATE_ERROR,
                WRITE_EVENTS,
            )

    def test_absolute_symlink_alias_is_resolved_before_write(self):
        with tempfile.TemporaryDirectory() as directory:
            alias = Path(directory) / "state-alias"
            alias.symlink_to(ROOT / ".state", target_is_directory=True)
            try:
                self.blocked(
                    lambda: (alias / "probe").write_text("probe"),
                    STATE_ERROR,
                    WRITE_EVENTS,
                )
            finally:
                # The strict realpath rule also rejects Python unlink of this
                # alias. rm unlinks this one test-owned link without following it.
                subprocess.run(["/bin/rm", "--", str(alias)], check=True)

    def test_dir_fd_write_and_relative_mutations_resolve_the_descriptor(self):
        # Opening the repository read-only also works in disposable copies that
        # intentionally have no .state directory.
        descriptor = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for flags in (
                os.O_WRONLY,
                os.O_RDWR,
                os.O_CREAT | os.O_WRONLY,
                os.O_APPEND | os.O_WRONLY,
                os.O_TRUNC | os.O_WRONLY,
            ):
                with self.subTest(flags=flags):
                    self.blocked(
                        lambda *, flags=flags: os.open(
                            ".state/probe", flags, dir_fd=descriptor
                        ),
                        STATE_ERROR,
                        WRITE_EVENTS,
                    )
            for operation in (
                lambda: os.mkdir(".state/probe", dir_fd=descriptor),
                lambda: os.unlink(".state/probe", dir_fd=descriptor),
                lambda: os.rmdir(".state/probe", dir_fd=descriptor),
                lambda: os.chmod(".state/probe", 0o600, dir_fd=descriptor),
                lambda: os.utime(".state/probe", dir_fd=descriptor),
                lambda: os.rename(
                    "outside-probe", ".state/probe", dst_dir_fd=descriptor
                ),
                lambda: os.rename(
                    ".state/probe", "outside-probe", src_dir_fd=descriptor
                ),
                lambda: os.link("outside-probe", ".state/probe", dst_dir_fd=descriptor),
                lambda: os.link(".state/probe", "outside-probe", src_dir_fd=descriptor),
                lambda: os.symlink("outside-probe", ".state/probe", dir_fd=descriptor),
            ):
                self.blocked(operation, STATE_ERROR, WRITE_EVENTS)
        finally:
            os.close(descriptor)

    def test_state_mutation_apis_are_blocked_before_syscalls(self):
        probe = ROOT / ".state/probe"
        outside = ROOT / "outside-probe"
        operations = {
            "replace destination": lambda: os.replace(outside, probe),
            "replace source": lambda: os.replace(probe, outside),
            "rename destination": lambda: os.rename(outside, probe),
            "rename source": lambda: os.rename(probe, outside),
            "remove": lambda: os.remove(probe),
            "unlink": lambda: os.unlink(probe),
            "mkdir": lambda: os.mkdir(probe),
            "makedirs": lambda: os.makedirs(probe / "nested"),
            "rmdir": lambda: os.rmdir(probe),
            "chmod": lambda: os.chmod(probe, 0o600),
            "utime": lambda: os.utime(probe),
            "link destination": lambda: os.link(outside, probe),
            "link source": lambda: os.link(probe, outside),
            "symlink": lambda: os.symlink(outside, probe),
            "truncate": lambda: os.truncate(probe, 0),
            "mkstemp": lambda: tempfile.mkstemp(dir=ROOT / ".state"),
            "mkdtemp": lambda: tempfile.mkdtemp(dir=ROOT / ".state"),
        }
        for name, operation in operations.items():
            with self.subTest(operation=name):
                self.blocked(operation, STATE_ERROR, WRITE_EVENTS)

    def test_shutil_writes_are_blocked_by_underlying_audit_events(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.write_text("fixture")
            probe = ROOT / ".state/probe"
            for operation in (
                lambda: shutil.copyfile(source, probe),
                lambda: shutil.copy2(source, probe),
                lambda: shutil.copytree(directory, probe),
                lambda: shutil.move(source, probe),
            ):
                self.blocked(operation, STATE_ERROR, WRITE_EVENTS)

    def test_outward_symlink_entries_are_protected_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / ".state"
            state.mkdir()
            outside = root / "outside"
            outside.mkdir()
            entry = state / "outward"
            entry.symlink_to(outside, target_is_directory=True)
            alias = root / "state-alias"
            alias.symlink_to(state, target_is_directory=True)
            descriptor = os.open(alias, os.O_RDONLY | os.O_DIRECTORY)
            try:
                # Build the fixture before protecting it; never alter real state.
                with mock.patch.object(
                    deployment_test_guard,
                    "STATE_PATHS",
                    (*deployment_test_guard.STATE_PATHS, state),
                ):
                    with contextlib.chdir(root):
                        targets = [
                            (entry, None),
                            (os.fsencode(entry), None),
                            (".state/outward", None),
                            (alias / "outward", None),
                            ("outward", descriptor),
                        ]
                        for target, dir_fd in targets:
                            operations = {
                                "replace source": lambda *, dir_fd=dir_fd, target=target: (
                                    os.replace(target, outside, src_dir_fd=dir_fd)
                                ),
                                "replace destination": lambda *, dir_fd=dir_fd, target=target: (
                                    os.replace(outside, target, dst_dir_fd=dir_fd)
                                ),
                                "rename source": lambda *, dir_fd=dir_fd, target=target: (
                                    os.rename(target, outside, src_dir_fd=dir_fd)
                                ),
                                "rename destination": lambda *, dir_fd=dir_fd, target=target: (
                                    os.rename(outside, target, dst_dir_fd=dir_fd)
                                ),
                                "remove": lambda *, dir_fd=dir_fd, target=target: (
                                    os.remove(target, dir_fd=dir_fd)
                                ),
                                "unlink": lambda *, dir_fd=dir_fd, target=target: (
                                    os.unlink(target, dir_fd=dir_fd)
                                ),
                                "symlink": lambda *, dir_fd=dir_fd, target=target: (
                                    os.symlink(outside, target, dir_fd=dir_fd)
                                ),
                                "link source": lambda *, dir_fd=dir_fd, target=target: (
                                    os.link(target, outside, src_dir_fd=dir_fd)
                                ),
                                "link destination": lambda *, dir_fd=dir_fd, target=target: (
                                    os.link(outside, target, dst_dir_fd=dir_fd)
                                ),
                                "rmdir": lambda *, dir_fd=dir_fd, target=target: (
                                    os.rmdir(target, dir_fd=dir_fd)
                                ),
                            }
                            for name, operation in operations.items():
                                with self.subTest(
                                    target=target, dir_fd=dir_fd, operation=name
                                ):
                                    self.blocked(operation, STATE_ERROR, WRITE_EVENTS)
            finally:
                os.close(descriptor)
            self.assertTrue(entry.is_symlink())
            self.assertTrue(outside.is_dir())

    def test_temporary_state_directories_and_read_opens_remain_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / ".state"
            state.mkdir()
            probe = state / "probe"
            descriptor = os.open(state, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fd = os.open(
                    "probe", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=descriptor
                )
                os.write(fd, b"fixture")
                os.close(fd)
            finally:
                os.close(descriptor)
            self.assertEqual(probe.read_text(), "fixture")
            shutil.copy2(probe, state / "copy")
            os.replace(probe, state / "renamed")
            shutil.rmtree(state)
        descriptor = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
        os.close(descriptor)

    def test_tools_absolute_paths_and_argv_wrappers_are_blocked(self):
        for tool in sorted(deployment_test_guard.TOOLS):
            for argv in ([tool], ["/usr/local/bin/" + tool], ["env", tool]):
                with self.subTest(argv=argv):
                    self.blocked(
                        lambda *, argv=argv: subprocess.run(argv),
                        f"test isolation: {tool} is not available under tests",
                        COMMAND_EVENTS,
                    )
        self.blocked(
            lambda: subprocess.run(["/usr/bin/git", "-C", str(ROOT), "push"]),
            "test isolation: git is not available under tests",
            COMMAND_EVENTS,
        )

    def test_shell_system_exec_and_posix_spawn_wrappers_are_blocked(self):
        message = "test isolation: kubectl is not available under tests"
        for command in (
            "kubectl version",
            "/usr/local/bin/kubectl version",
            "true; kubectl version",
            "true && kubectl version",
            "echo hi | kubectl apply",
            "false || /usr/bin/kubectl version",
            'bash -c "/usr/local/bin/kubectl version"',
        ):
            self.blocked(
                lambda *, command=command: subprocess.run(["bash", "-c", command]),
                message,
                COMMAND_EVENTS,
            )
            self.blocked(
                lambda *, command=command: subprocess.run(command, shell=True),
                message,
                COMMAND_EVENTS,
            )
            self.blocked(
                lambda *, command=command: os.system(command), message, COMMAND_EVENTS
            )
        self.blocked(
            lambda: os.execv("/usr/local/bin/kubectl", ["kubectl"]),
            message,
            COMMAND_EVENTS,
        )
        self.blocked(
            lambda: os.execve(
                "/bin/bash", ["bash", "-c", "kubectl version"], dict(os.environ)
            ),
            message,
            COMMAND_EVENTS,
        )
        for spawn in (os.posix_spawn, os.posix_spawnp):
            self.blocked(
                lambda *, spawn=spawn: spawn(
                    "/bin/bash", ["bash", "-c", "kubectl version"], dict(os.environ)
                ),
                message,
                COMMAND_EVENTS,
            )

    def test_tool_names_in_arguments_remain_usable(self):
        commands = [
            ["/bin/echo", "aws", "/usr/bin/kubectl", "git", "push"],
            ["env", "/bin/echo", "aws"],
            ["/bin/bash", "-c", "echo aws; echo kubectl && echo git push | cat"],
            ["/bin/bash", "-c", "echo -c kubectl"],
            ["/bin/bash", "-c", 'echo ";" aws && echo "|" kubectl'],
            [sys.executable, "-c", "print('aws kubectl git push')"],
        ]
        for argv in commands:
            with self.subTest(argv=argv):
                result = subprocess.run(
                    argv, capture_output=True, text=True, timeout=10
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_git_network_verbs_are_blocked_in_command_position(self):
        for verb in ("push", "fetch", "clone"):
            for argv in (
                ["git", verb],
                ["/usr/bin/git", "-C", str(ROOT), verb],
                ["git", "-c", "alias.probe=echo", verb],
                ["env", "git", verb],
                ["/bin/bash", "-c", f"echo hi | git {verb}"],
            ):
                with self.subTest(argv=argv):
                    self.blocked(
                        lambda *, argv=argv: subprocess.run(argv),
                        "test isolation: git is not available under tests",
                        COMMAND_EVENTS,
                    )

    def test_local_git_commands_remain_usable_through_audit_and_path_stub(self):
        # The import integration suite runs in a copy without .git metadata.
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                ["/usr/bin/git", "init", "--quiet", directory],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            for command in ("git", "/usr/bin/git"):
                for arguments in (
                    ["--version"],
                    [
                        "-C",
                        directory,
                        "status",
                        "--porcelain",
                        "--untracked-files=no",
                        "--",
                        "push",
                        "fetch",
                        "clone",
                    ],
                ):
                    argv = [command, *arguments]
                    with self.subTest(argv=argv):
                        result = subprocess.run(
                            argv, capture_output=True, text=True, timeout=10
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stderr, "")

    def test_path_stubs_block_commands_inside_a_non_python_script(self):
        # The tool name is in the script, beyond the parent audit argv check.
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "wrapper.sh"
            for tool in sorted(deployment_test_guard.TOOLS | {"git"}):
                with self.subTest(tool=tool):
                    # Git's PATH stub now rejects only network verbs.
                    script.write_text(tool + (" push" if tool == "git" else "") + "\n")
                    result = subprocess.run(
                        ["/bin/bash", str(script)],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 99, result.stderr)
                    self.assertEqual(
                        result.stderr,
                        f"test isolation: {tool} is not available under tests\n",
                    )

            for verb in ("fetch", "clone"):
                with self.subTest(git_verb=verb):
                    script.write_text(f"git -C {shlex.quote(directory)} {verb}\n")
                    result = subprocess.run(
                        ["/bin/bash", str(script)],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 99, result.stderr)
                    self.assertEqual(
                        result.stderr,
                        "test isolation: git is not available under tests\n",
                    )

    def test_external_tcp_udp_and_dns_are_blocked_before_syscalls(self):
        with socket.socket() as tcp, socket.socket(type=socket.SOCK_DGRAM) as udp:
            tcp.settimeout(0.05)
            self.blocked(
                lambda: tcp.connect(("192.0.2.1", 9)), NETWORK_ERROR, NETWORK_EVENTS
            )
            self.blocked(
                lambda: udp.sendto(b"probe", ("192.0.2.1", 9)),
                NETWORK_ERROR,
                NETWORK_EVENTS,
            )
            self.blocked(
                lambda: udp.sendmsg([b"probe"], [], 0, ("192.0.2.1", 9)),
                NETWORK_ERROR,
                NETWORK_EVENTS,
            )
        for lookup in (
            socket.getaddrinfo,
            socket.gethostbyname,
            socket.gethostbyname_ex,
            socket.gethostbyaddr,
        ):
            args = (
                ("example.invalid", 9)
                if lookup is socket.getaddrinfo
                else ("example.invalid",)
            )
            self.blocked(
                lambda *, args=args, lookup=lookup: lookup(*args),
                NETWORK_ERROR,
                NETWORK_EVENTS,
            )
        for host in ("0.0.0.0", "::", "192.0.2.1", "::ffff:192.0.2.1"):
            self.blocked(
                lambda *, host=host: socket.getaddrinfo(host, 9),
                NETWORK_ERROR,
                NETWORK_EVENTS,
            )

    def test_loopback_tcp_udp_and_name_lookups_remain_usable(self):
        for host in ("localhost", "127.0.0.1", "127.7.8.9", "::1"):
            self.assertTrue(socket.getaddrinfo(host, 9))
        with socket.socket() as listener, socket.socket() as client:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            client.settimeout(1)
            client.connect(listener.getsockname())
            peer, _ = listener.accept()
            peer.close()
        with (
            socket.socket(type=socket.SOCK_DGRAM) as receiver,
            socket.socket(type=socket.SOCK_DGRAM) as sender,
        ):
            receiver.bind(("127.0.0.1", 0))
            receiver.settimeout(1)
            sender.sendto(b"probe", receiver.getsockname())
            self.assertEqual(receiver.recv(16), b"probe")
            sender.sendmsg([b"probe"], [], 0, receiver.getsockname())
            self.assertEqual(receiver.recv(16), b"probe")
            sender.connect(receiver.getsockname())
            sender.sendmsg([b"connected"])
            self.assertEqual(receiver.recv(16), b"connected")

    def test_unix_stream_and_datagram_sockets_remain_usable(self):
        with tempfile.TemporaryDirectory() as directory:
            address = str(Path(directory) / "socket")
            with (
                socket.socket(socket.AF_UNIX) as listener,
                socket.socket(socket.AF_UNIX) as client,
            ):
                listener.bind(address)
                listener.listen(1)
                listener.settimeout(1)
                client.settimeout(1)
                client.connect(address)
                peer, _ = listener.accept()
                peer.close()
            os.unlink(address)
            with (
                socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver,
                socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender,
            ):
                receiver.bind(address)
                receiver.settimeout(1)
                for destination in (address, os.fsencode(address)):
                    sender.sendto(b"probe", destination)
                    self.assertEqual(receiver.recv(16), b"probe")
                    sender.sendmsg([b"probe"], [], 0, destination)
                    self.assertEqual(receiver.recv(16), b"probe")

    def test_child_python_inherits_command_state_and_network_guards(self):
        probes = [
            (
                "import subprocess; subprocess.run(['kubectl'])",
                "test isolation: kubectl",
            ),
            (
                "import subprocess; subprocess.run(['/usr/local/bin/kubectl'])",
                "test isolation: kubectl",
            ),
            (f"open({str(ROOT / '.state/probe')!r}, 'w')", STATE_ERROR),
            (
                "import socket; s=socket.socket(); s.settimeout(.05); s.connect(('192.0.2.1',9))",
                NETWORK_ERROR,
            ),
            ("import socket; socket.getaddrinfo('example.invalid',9)", NETWORK_ERROR),
        ]
        # A secondary hook in each child fails closed if sitecustomize is absent.
        safety = (
            "import sys\n"
            "def fallback(event, args):\n"
            "    if event in {'subprocess.Popen','socket.connect','socket.getaddrinfo'} "
            "or (event == 'open' and isinstance(args[1], str) and 'w' in args[1]):\n"
            "        raise AssertionError('child primary guard missing')\n"
            "sys.addaudithook(fallback)\n"
        )
        for code, message in probes:
            with self.subTest(code=code):
                result = subprocess.run(
                    [sys.executable, "-c", safety + code],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("RuntimeError: " + message, result.stderr)
                self.assertNotIn(
                    "AssertionError: child primary guard missing", result.stderr
                )

    def test_child_in_a_copied_repository_keeps_parent_state_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "tests").mkdir()
            shutil.copy2(
                ROOT / "tests/deployment_test_guard.py",
                root / "tests/deployment_test_guard.py",
            )
            code = (
                "import deployment_test_guard as guard, sys\n"
                "assert str(guard.ROOT) == sys.argv[1], (str(guard.ROOT), sys.argv[1])\n"
                "def fallback(event, args):\n"
                "    if event == 'open' and args[1] == 'w':\n"
                "        raise AssertionError('parent state protection missing')\n"
                "sys.addaudithook(fallback)\n"
                f"open({str(ROOT / '.state/probe')!r}, 'w')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code, str(root)],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("RuntimeError: " + STATE_ERROR, result.stderr)
            self.assertNotIn("AssertionError:", result.stderr)

    def test_every_test_module_imports_guard_before_other_runtime_code(self):
        missing = []
        for path in (ROOT / "tests").rglob("test_*.py"):
            nodes = ast.parse(path.read_text()).body
            nodes = [
                node
                for node in nodes
                if not (
                    isinstance(node, ast.ImportFrom) and node.module == "__future__"
                )
                and not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )
            ]
            if not (
                nodes
                and isinstance(nodes[0], ast.Import)
                and [alias.name for alias in nodes[0].names]
                == ["deployment_test_guard"]
            ):
                missing.append(str(path.relative_to(ROOT)))
        self.assertEqual(
            missing, [], "Guard must be the first runtime import: " + ", ".join(missing)
        )


if __name__ == "__main__":
    unittest.main()
