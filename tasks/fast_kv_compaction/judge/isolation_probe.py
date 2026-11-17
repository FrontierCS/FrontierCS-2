#!/usr/bin/env python3
"""Worker-side isolation checks, run by the harness in the shared isolate sandbox exactly as a
worker is (the same options, the method tree read-only at {in}/tree).

  probe TREE     prints FASTKV_ISOLATION_OK if every check holds, else FASTKV_ISOLATION_FAIL
                 <reasons>; leaves a marker in every directory it can write
  residue TREE   in a fresh instance after `probe`: FASTKV_ISOLATION_OK if no marker of the
                 earlier instance is visible anywhere
  hog            allocates host memory without bound (the instance's memory limit must end it)
  hog-shm        fills the private /dev/shm instead (the limit must count it too)

The harness adds the parent-side checks (no marker reached the runner's own /tmp, /dev/shm,
/var/tmp; the memory limit ends both hogs) and treats any failure as infrastructure.
"""
import ctypes
import os
import socket
import sys
import time

MARKER = "fastkv-isolation-marker"
JUDGE = "/opt/frontier/task"   # judge state on the runner: covered by isolate
SKIP_SCAN = ("/proc", "/sys", "/dev")
CLONE_NEWUSER, CLONE_NEWNS, MNT_DETACH, MS_REMOUNT, MS_BIND = 0x10000000, 0x00020000, 2, 32, 4096
EROFS = 30


def libc():
    return ctypes.CDLL(None, use_errno=True)


def own_dirs():
    """What the instance may write: its own directory (isolate: the parent of its HOME, TMPDIR,
    work and out directories; deleted with it) and its private /dev/shm."""
    root = os.path.dirname(os.environ["HOME"])
    if os.path.dirname(os.environ["TMPDIR"]) != root:
        raise RuntimeError("HOME and TMPDIR are not one instance's directories")
    return [root, "/dev/shm"]


def judge_hidden(failures, tree):
    try:
        if os.listdir(JUDGE):
            failures.append("judge directory not empty")
    except OSError:
        pass
    for path in (f"{JUDGE}/gate.py", f"{JUDGE}/harness.py", f"{JUDGE}/data", f"{JUDGE}/targets",
                 f"{JUDGE}/pristine", os.path.join(tree, "data")):
        if os.path.exists(path):
            failures.append(f"{path} visible")


def cannot_unmount(failures, tree):
    lib = libc()
    for target in (JUDGE, "/tmp", tree):
        if lib.umount2(target.encode(), MNT_DETACH) == 0:
            failures.append(f"umount {target} succeeded")
        if lib.mount(None, target.encode(), None, MS_REMOUNT | MS_BIND, None) == 0:
            failures.append(f"remount {target} succeeded")
    # Should a namespace of its own be creatable, the outer mounts must stay locked there.
    # Each escape that works sets one bit of the exit code.
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            if lib.unshare(CLONE_NEWUSER | CLONE_NEWNS) != 0:
                os._exit(0)  # cannot even create one: nothing to escape with
            peek = os.path.join(os.environ["TMPDIR"], "fastkv-probe-peek").encode()
            os.makedirs(peek, exist_ok=True)
            for bit, source in ((1, b"/"), (2, b"/opt/frontier")):  # non-recursive binds
                if lib.mount(source, peek, None, MS_BIND, None) == 0:
                    revealed = os.path.join(peek.decode(), "opt/frontier/task/gate.py" if source == b"/"
                                            else "task/gate.py")
                    if os.path.exists(revealed) or os.listdir(os.path.dirname(revealed)):
                        code |= bit
                    lib.umount2(peek, MNT_DETACH)
            if lib.mount(None, tree.encode(), None, MS_REMOUNT | MS_BIND, None) == 0:
                code |= 4
            for bit, target in ((8, JUDGE.encode()), (16, b"/tmp"), (32, b"/workspace")):
                if os.path.isdir(target) and lib.umount2(target, MNT_DETACH) == 0:
                    code |= bit
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    code = os.WEXITSTATUS(status)
    for bit, what in ((1, "bind of / revealed the judge"), (2, "bind of /opt/frontier revealed the judge"),
                      (4, "method tree remounted writable"), (8, "judge tmpfs unmounted"),
                      (16, "/tmp tmpfs unmounted"), (32, "/workspace tmpfs unmounted")):
        if code & bit:
            failures.append(f"nested namespace: {what}")


def tree_read_only(failures, tree):
    try:
        with open(os.path.join(tree, "fastkv-probe-write"), "w") as stream:
            stream.write("x")
        failures.append("method tree is writable")
    except OSError as error:
        if error.errno != EROFS:
            failures.append(f"method tree write failed with errno {error.errno}, not EROFS")


def processes_hidden(failures, tree):
    pids = [entry for entry in os.listdir("/proc") if entry.isdigit()]
    if len(pids) > 8:
        failures.append(f"{len(pids)} processes visible")
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as stream:
                command = stream.read()
            with open(f"/proc/{pid}/environ", "rb") as stream:
                environment = stream.read()
        except OSError:
            continue
        if b"harness.py" in command or b"evaluator.py" in command or b"FRONTIER_" in environment:
            failures.append(f"trusted process {pid} visible")


def environment_clean(failures, tree):
    leaked = sorted(key for key in os.environ if key.startswith(("FRONTIER_", "HARBOR_")))
    if leaked:
        failures.append(f"environment {leaked}")


def network_loopback_only(failures, tree):
    names = sorted(name for _index, name in socket.if_nameindex())
    if names != ["lo"]:
        failures.append(f"interfaces {names}")
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3).close()
        failures.append("external network reachable")
    except OSError:
        pass


def unprivileged(failures, tree):
    if os.getuid() == 0 or os.geteuid() == 0:
        failures.append("worker is uid 0 in its namespace")
    with open("/proc/self/status") as stream:
        status = dict(line.split(":", 1) for line in stream if ":" in line)
    for field in ("CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if int(status.get(field, "0").strip(), 16) != 0:
            failures.append(f"{field} not empty")
    if status.get("NoNewPrivs", "0").strip() != "1":
        failures.append("no_new_privs not set")


def mqueue_private(failures, tree):
    if os.path.isdir("/dev/mqueue") and os.listdir("/dev/mqueue"):
        failures.append("/dev/mqueue is not a fresh instance")


def writes_confined(failures, tree):
    """Only the instance's own directories are writable; leave a marker wherever it can."""
    own = own_dirs()
    for directory in (os.environ["HOME"], os.environ["TMPDIR"], "/dev/shm", "/tmp", "/var/tmp"):
        try:
            with open(os.path.join(directory, MARKER), "w") as stream:
                stream.write("x")
        except OSError:
            pass
    found = 0
    for root, directories, files in os.walk("/", topdown=True):
        if root in SKIP_SCAN or any(root == d or root.startswith(d + "/") for d in own):
            directories[:] = []
            continue
        if root != "/" and os.access(root, os.W_OK):
            failures.append(f"{root} writable")
            found += 1
        for name in files:  # files too: an existing writable file is a channel as well
            path = os.path.join(root, name)
            if not os.path.islink(path) and os.access(path, os.W_OK):
                failures.append(f"{path} writable")
                found += 1
        if found > 20:
            return
        directories[:] = [d for d in directories if not os.path.islink(os.path.join(root, d))]


def residue(failures, tree):
    """No marker of an earlier instance anywhere this one can see."""
    for root, directories, files in os.walk("/", topdown=True):
        if root in ("/proc", "/sys"):
            directories[:] = []
            continue
        if MARKER in files:
            failures.append(f"marker of an earlier instance in {root}")
        directories[:] = [d for d in directories if not os.path.islink(os.path.join(root, d))]


CHECKS = (judge_hidden, cannot_unmount, tree_read_only, processes_hidden, environment_clean,
          network_loopback_only, unprivileged, mqueue_private, writes_confined)


def main() -> int:
    if sys.argv[1:] == ["hog-shm"]:
        index = 0
        while True:
            try:
                with open(f"/dev/shm/fastkv-hog-{index}", "wb") as stream:
                    stream.write(b"\xa5" * (256 << 20))
            except OSError:
                pass
            index += 1
            time.sleep(0.05)
    if sys.argv[1:] == ["hog"]:
        blocks = []
        while True:
            blocks.append(bytearray(256 << 20))
            time.sleep(0.05)
    mode, tree = sys.argv[1:]
    failures = []
    for check in ((residue,) if mode == "residue" else CHECKS):
        try:
            check(failures, tree)
        except Exception as error:  # noqa: BLE001 - a check that cannot run is a failure
            failures.append(f"{check.__name__}: {error}")
    if failures:
        print("FASTKV_ISOLATION_FAIL " + "; ".join(failures)[:2000], flush=True)
        return 1
    print("FASTKV_ISOLATION_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
