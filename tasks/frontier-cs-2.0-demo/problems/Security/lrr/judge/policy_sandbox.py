"""Run the submitted prompt policy where it cannot see judge files (no root needed).

  python3 policy_sandbox.py JUDGE_DIR BUNDLE

BUNDLE (a fresh /tmp/lrr-policy-* directory made by the evaluator) holds exactly:
  tree/              the patched public repository
  worker.py          a copy of constructor_worker.py
  request.json       {"task": ..., "lines": [...]}: the allowed inputs only

Stage 1 enters fresh user (mapped to root), mount, network, PID and IPC
namespaces through unshare. The policy process becomes PID 1 of the new PID
namespace, so when it exits the kernel kills every process it started,
including daemonized ones; --kill-child also kills it if the launcher dies.

Stage 2, as namespace root:
  - fresh tmpfs covers JUDGE_DIR (the directory holding the evaluator, the
    workloads and the reference patch) and the bundle is bound read-only at
    JUDGE_DIR/policy inside that empty tmpfs;
  - fresh tmpfs then covers /opt/frontier/task, the verifier input/output
    (/frontier), /tmp, /dev/shm, /var/tmp, /workspace and HOME directories;
  - a nested user namespace makes the policy an ordinary uid with no
    capabilities over these mounts, with the bounding set emptied and
    no_new_privs set.

This mirrors Fast KV's verifier worker launcher (fastkv-worker).
Exit 64: bad arguments; 70: isolation unavailable (an evaluator error, never a
candidate failure). Any other status is the policy's own result.
"""
import ctypes
import os
import re
import sys

REQUIRED_HIDDEN = {"/tmp": "512m", "/dev/shm": "64m"}
# /opt/frontier/task is the judge directory in the image, hidden first as JUDGE_DIR.
OPTIONAL_HIDDEN = {"/opt/frontier/task": "16m", "/frontier": "16m", "/workspace": "64m", "/var/tmp": "16m", "/home": "16m",
                   "/root": "16m", "/run/lock": "1m"}
POLICY_UID = 1000
BUNDLE = re.compile(r"^/tmp/lrr-policy-[A-Za-z0-9_]+$")

CLONE_NEWUSER = 0x10000000
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_REMOUNT, MS_BIND, MS_REC, MS_PRIVATE = 1, 2, 4, 32, 4096, 16384, 1 << 18
PR_SET_NO_NEW_PRIVS, PR_CAPBSET_DROP = 38, 24
LIBC = ctypes.CDLL(None, use_errno=True)


def fail(code: int, message: str):
    print(f"lrr-policy-sandbox: {message}", file=sys.stderr)
    raise SystemExit(code)


def mount(source, target, fstype, flags, data=None):
    if LIBC.mount(source and source.encode(), target.encode(), fstype and fstype.encode(),
                  flags, data and data.encode()) != 0:
        raise OSError(ctypes.get_errno(), f"mount {target}")


def write(path, text):
    with open(path, "w") as stream:
        stream.write(text)


def stage2(judge: str, bundle: str) -> None:
    policy_root = os.path.join(judge, "policy")
    try:
        mount(None, "/", None, MS_REC | MS_PRIVATE)
        mount("tmpfs", judge, "tmpfs", MS_NOSUID | MS_NODEV, "mode=0755,size=1m")
        os.mkdir(policy_root, 0o755)
        mount(bundle, policy_root, None, MS_BIND | MS_REC)
        mount(None, policy_root, None, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV)
        try:  # best effort: gVisor and some kernels refuse a read-only root here
            mount(None, "/", None, MS_BIND | MS_REMOUNT | MS_RDONLY)
        except OSError:
            pass
        for path, size in REQUIRED_HIDDEN.items():
            if path == judge:
                continue
            mount("tmpfs", path, "tmpfs", MS_NOSUID | MS_NODEV, f"mode=1777,size={size}")
        for path, size in OPTIONAL_HIDDEN.items():
            try:
                if path != judge and os.path.isdir(path):
                    mount("tmpfs", path, "tmpfs", MS_NOSUID | MS_NODEV, f"mode=1777,size={size}")
            except OSError:
                pass
        os.makedirs("/tmp/home", mode=0o700, exist_ok=True)
    except OSError as error:
        fail(70, f"isolation unavailable: {error}")
    if LIBC.unshare(CLONE_NEWUSER) != 0:
        fail(70, "nested user namespace unavailable")
    try:
        write("/proc/self/setgroups", "deny")
        write("/proc/self/uid_map", f"{POLICY_UID} 0 1\n")
        write("/proc/self/gid_map", f"{POLICY_UID} 0 1\n")
    except OSError as error:
        fail(70, f"nested id map failed: {error}")
    for capability in range(64):
        LIBC.prctl(PR_CAPBSET_DROP, capability, 0, 0, 0)
    if LIBC.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        fail(70, "no_new_privs failed")
    os.chdir(os.path.join(policy_root, "tree"))
    environment = {"PATH": "/usr/bin:/bin", "HOME": "/tmp/home", "TMPDIR": "/tmp/home",
                   "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    os.execve(sys.executable, [sys.executable, "-I", os.path.join(policy_root, "worker.py"),
                               os.path.join(policy_root, "request.json")], environment)


def main(argv) -> None:
    if len(argv) == 3 and argv[0] == "--stage2":
        stage2(argv[1], argv[2])
        return
    if len(argv) != 2 or BUNDLE.fullmatch(argv[1]) is None:
        fail(64, "usage: policy_sandbox.py JUDGE_DIR /tmp/lrr-policy-*")
    judge, bundle = argv
    for path in (judge, bundle):
        if not os.path.isabs(path) or os.path.realpath(path) != path or not os.path.isdir(path):
            fail(64, "arguments must be existing canonical directories")
    if judge in ("/", "/tmp") or bundle.startswith(judge.rstrip("/") + "/"):
        fail(64, "the judge directory must not contain the bundle")
    command = ["/usr/bin/unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--ipc",
               "--fork", "--mount-proc", "--kill-child", "--", sys.executable, "-I", os.path.abspath(__file__),
               "--stage2", judge, bundle]
    try:
        os.execv(command[0], command)
    except OSError as error:
        fail(70, f"unshare unavailable: {error}")


if __name__ == "__main__":
    main(sys.argv[1:])
