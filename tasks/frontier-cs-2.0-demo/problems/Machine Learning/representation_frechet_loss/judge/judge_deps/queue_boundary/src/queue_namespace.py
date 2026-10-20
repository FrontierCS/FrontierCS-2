"""Retained campaign bubblewrap boundary, with explicit visible GPU mapping."""

import hashlib
import json
import os
from pathlib import Path
import re
import stat

LOCAL_BWRAP_SHA = "499886cb0faa11dd765bff7e4241327df290a59c29877271a2847e68cd203fd5"
BWRAP_SHA = "4348d585ce8264f4ed27a4eebab95a865fb374975eb9a56376cfa8442b0b6bb6"
ACTOR_FILES = ("queue_actor.py", "queue_wire.py", "queue_profile.py", "restricted_runtime.py")


def discover_gpu_devices():
    """Resolve the single provider-exposed GPU without assuming its physical index."""
    numeric = sorted(path for path in Path("/dev").glob("nvidia*")
                     if re.fullmatch(r"nvidia[0-9]+", path.name))
    if len(numeric) != 1:
        raise ValueError("Expected exactly one assigned numeric NVIDIA device")
    gpu = numeric[0]
    minor = int(gpu.name.removeprefix("nvidia"))
    if gpu.name != f"nvidia{minor}" or not 0 <= minor <= 247:
        raise ValueError("Invalid assigned NVIDIA device number")
    required = [(gpu, 195, minor), (Path("/dev/nvidiactl"), 195, 255),
                (Path("/dev/nvidia-uvm"), None, 0)]
    for path, major, expected_minor in required:
        try:
            info = path.lstat()
        except OSError as error:
            raise ValueError(f"Required CUDA device is unavailable: {path}") from error
        actual_major, actual_minor = os.major(info.st_rdev), os.minor(info.st_rdev)
        if (not stat.S_ISCHR(info.st_mode) or actual_minor != expected_minor
                or (major is not None and actual_major != major)
                or (major is None and actual_major in (0, 195))):
            raise ValueError(f"Invalid CUDA character device: {path}")
    # nvproxy exposes primary UVM; the separate tools endpoint is not a compute binding.
    return [str(path) for path, _, _ in required]


def command(source, tools, python, *, cuda=False):
    binary = tools.parent / "tools/bwrap" if cuda else Path("/usr/bin/bwrap")
    expected_sha = BWRAP_SHA if cuda else LOCAL_BWRAP_SHA
    if hashlib.sha256(binary.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Retained namespace executable changed")
    executable = Path(python)
    if not (executable.parent.parent / "pyvenv.cfg").is_file():
        # The task's /opt/frontier/runtime-python is a system-Python symlink.
        # Deriving a prefix before resolving it would expose private /opt data.
        executable = executable.resolve(strict=True)
    python_root = executable.parent.parent.resolve()
    python_name = executable.name
    if python_root in (Path("/"), Path("/opt"), Path("/home")):
        raise ValueError("Python prefix would expose unrelated private data")
    bindings_file = python_root / "runtime-roots.json"
    wheel_roots = json.loads(bindings_file.read_text()) if bindings_file.exists() else []
    runtime = [Path(p) for p in ("/usr", "/lib", "/lib64") if Path(p).exists()]
    runtime += [python_root, *[Path(p) for p in wheel_roots]]
    argv = [str(binary), "--unshare-all", "--die-with-parent", "--new-session",
            "--cap-drop", "ALL", "--clearenv", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--tmpfs", "/home", "--dir", "/home/queue"]
    for path in dict.fromkeys(runtime):
        argv += ["--ro-bind", str(path), str(path)]
    if Path("/etc/ld.so.cache").exists():
        argv += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Queue namespace source must be a real directory")
    argv += ["--ro-bind", str(source), "/solution", "--dir", "/actor"]
    for name in ACTOR_FILES:
        path = tools / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Expected allowlisted actor tool")
        argv += ["--ro-bind", str(path), "/actor/" + name]
    library = tools.parent / "tools/libseccomp.so.2"
    if library.is_symlink() or not library.is_file():
        raise ValueError("Expected retained libseccomp library")
    argv += ["--ro-bind", str(library), "/actor/libseccomp.so.2"]
    if cuda:
        for path in discover_gpu_devices():
            argv += ["--dev-bind", str(path), str(path)]
    env = {
        "PATH": "/usr/bin:/bin", "HOME": "/home/queue", "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "CUDA_CACHE_PATH": "/tmp",
        "CUDA_VISIBLE_DEVICES": "0" if cuda else "",
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
        "FQ_PARENT_PIDNS": os.readlink("/proc/self/ns/pid"),
    }
    for key, value in env.items():
        argv += ["--setenv", key, value]
    argv += ["--remount-ro", "/", "--chdir", "/tmp", "--",
             str(python_root / "bin" / python_name), "-I", "-B", "/actor/queue_actor.py",
             "/solution", "/tmp", "namespace_cuda" if cuda else "namespace_cpu"]
    if os.geteuid() == 0:
        argv = ["/usr/bin/setpriv", "--reuid=65534", "--regid=65534",
                "--clear-groups", "--no-new-privs", *argv]
    return argv
