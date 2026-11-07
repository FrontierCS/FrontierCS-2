#!/usr/bin/env python3
"""Worker-side isolation checks, run by the launcher exactly as a worker would be.

  probe   prints FASTKV_ISOLATION_OK if every check holds, else FASTKV_ISOLATION_FAIL <reasons>
  hog        allocates host memory without bound (the harness's watchdog must kill it)
  hog-tmpfs  fills the RAM-backed /tmp instead (the watchdog must count tmpfs too)

The harness adds the parent-side checks (writes did not persist, the
watchdog kills a hog) and treats any failure as infrastructure.
"""
import ctypes
import os
import socket
import sys
import time

MARKER = "fastkv-isolation-marker"
WRITE_PROBES = ("/tmp", "/dev/shm", "/var/tmp", "/tmp/home")
HIDDEN = ("/opt/frontier/task", "/tmp", "/dev/shm", "/var/tmp", "/workspace", "/frontier", "/home", "/root",
          "/run/lock", "/dev/mqueue")
WORKER_ROOT = "/opt/frontier/worker-root"
SKIP_SCAN = ("/proc", "/sys", "/dev")
CLONE_NEWUSER, CLONE_NEWNS, MNT_DETACH, MS_REMOUNT, MS_BIND = 0x10000000, 0x00020000, 2, 32, 4096
EROFS = 30


def libc():
    return ctypes.CDLL(None, use_errno=True)


def judge_hidden(failures):
    try:
        if os.listdir("/opt/frontier/task"):
            failures.append("judge directory not empty")
    except OSError:
        pass
    for path in ("/opt/frontier/task/gate.py", "/opt/frontier/task/data", "/opt/frontier/task/targets"):
        if os.path.exists(path):
            failures.append(f"{path} visible")


def cannot_unmount(failures):
    lib = libc()
    for target in ("/opt/frontier/task", "/tmp"):
        if lib.umount2(target.encode(), MNT_DETACH) == 0:
            failures.append(f"umount {target} succeeded")
        if lib.mount(None, target.encode(), None, MS_REMOUNT | MS_BIND, None) == 0:
            failures.append(f"remount {target} succeeded")
    # In a namespace of its own (where the worker is root), the outer mounts must
    # stay locked (MNT_LOCKED): no bind, remount or unmount may reveal the judge or
    # make the tree writable. Each escape that works sets one bit of the exit code.
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            if lib.unshare(CLONE_NEWUSER | CLONE_NEWNS) != 0:
                os._exit(0)  # cannot even create one: nothing to escape with
            peek = b"/tmp/fastkv-probe-peek"
            os.makedirs(peek, exist_ok=True)
            for bit, source in ((1, b"/"), (2, b"/opt/frontier")):  # non-recursive binds
                if lib.mount(source, peek, None, MS_BIND, None) == 0:
                    revealed = os.path.join(peek.decode(), "opt/frontier/task/gate.py" if source == b"/"
                                            else "task/gate.py")
                    if os.path.exists(revealed) or os.listdir(os.path.dirname(revealed)):
                        code |= bit
                    lib.umount2(peek, MNT_DETACH)
            if lib.mount(None, WORKER_ROOT.encode(), None, MS_REMOUNT | MS_BIND, None) == 0:
                code |= 4
            for bit, target in ((8, b"/opt/frontier/task"), (16, b"/tmp"), (32, b"/workspace")):
                if os.path.isdir(target) and lib.umount2(target, MNT_DETACH) == 0:
                    code |= bit
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    code = os.WEXITSTATUS(status)
    for bit, what in ((1, "bind of / revealed the judge"), (2, "bind of /opt/frontier revealed the judge"),
                      (4, "worker tree remounted writable"), (8, "judge tmpfs unmounted"),
                      (16, "/tmp tmpfs unmounted"), (32, "/workspace tmpfs unmounted")):
        if code & bit:
            failures.append(f"nested namespace: {what}")


def tree_read_only(failures):
    try:
        with open(os.path.join(WORKER_ROOT, "fastkv-probe-write"), "w") as stream:
            stream.write("x")
        failures.append("worker tree is writable")
    except OSError as error:
        if error.errno != EROFS:
            failures.append(f"worker tree write failed with errno {error.errno}, not EROFS")


def processes_hidden(failures):
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


def network_loopback_only(failures):
    names = sorted(name for _index, name in socket.if_nameindex())
    if names != ["lo"]:
        failures.append(f"interfaces {names}")
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3).close()
        failures.append("external network reachable")
    except OSError:
        pass


def unprivileged(failures):
    if os.getuid() == 0 or os.geteuid() == 0:
        failures.append("worker is uid 0 in its namespace")
    with open("/proc/self/status") as stream:
        status = dict(line.split(":", 1) for line in stream if ":" in line)
    for field in ("CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if int(status.get(field, "0").strip(), 16) != 0:
            failures.append(f"{field} not empty")
    if status.get("NoNewPrivs", "0").strip() != "1":
        failures.append("no_new_privs not set")


def writes_confined(failures):
    for directory in WRITE_PROBES:
        try:
            with open(os.path.join(directory, MARKER), "w") as stream:
                stream.write("x")
        except OSError:
            pass
    found = 0
    for root, directories, files in os.walk("/", topdown=True):
        if root in SKIP_SCAN or root in HIDDEN or root.startswith(WORKER_ROOT):
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
        directories[:] = [d for d in directories
                          if not os.path.islink(os.path.join(root, d))
                          and os.path.join(root, d) not in SKIP_SCAN + HIDDEN]


def mqueue_private(failures):
    if os.path.isdir("/dev/mqueue") and os.listdir("/dev/mqueue"):
        failures.append("/dev/mqueue is not a fresh instance")


def main() -> int:
    if sys.argv[1:] == ["hog-tmpfs"]:
        index = 0
        while True:
            try:
                with open(f"/tmp/fastkv-hog-{index}", "wb") as stream:
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
    failures = []
    for check in (judge_hidden, cannot_unmount, tree_read_only, processes_hidden, network_loopback_only,
                  unprivileged, mqueue_private, writes_confined):
        try:
            check(failures)
        except Exception as error:  # noqa: BLE001 - a check that cannot run is a failure
            failures.append(f"{check.__name__}: {error}")
    if failures:
        print("FASTKV_ISOLATION_FAIL " + "; ".join(failures)[:2000], flush=True)
        return 1
    print("FASTKV_ISOLATION_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
