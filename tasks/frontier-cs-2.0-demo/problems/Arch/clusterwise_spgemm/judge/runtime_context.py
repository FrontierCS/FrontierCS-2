"""Scientific resource readbacks; development use must select an explicit context."""
import ctypes
import json
import os
from pathlib import Path
import resource
import stat
import tempfile

from runtime_contract import LogicalCPU, RuntimeLimits, select_logical_cpus, validate_cpu_quota

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "runtime-config.json").read_text())
LIMITS = RuntimeLimits.from_config(CONFIG)


def topology():
    entries = []
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        nodes = sorted(root.glob("node[0-9]*"))
        if len(nodes) != 1:
            raise RuntimeError("unavailable CPU NUMA topology")
        entries.append(LogicalCPU(int((root / "topology/physical_package_id").read_text()),
                                 int(nodes[0].name[4:]), int((root / "topology/core_id").read_text()), cpu))
    return entries


def cgroup_directory():
    rows = Path("/proc/self/cgroup").read_text().splitlines()
    paths = [line[3:] for line in rows if line.startswith("0::")]
    if len(paths) != 1 or ".." in Path(paths[0]).parts:
        raise RuntimeError("cgroup-v2 membership unavailable")
    root = Path("/sys/fs/cgroup")
    target = root / paths[0].lstrip("/")
    if not (target / "cpu.max").is_file():
        raise RuntimeError("cgroup-v2 readback unavailable")
    return target


def verify_capacity(cgroup, count=LIMITS.threads):
    root = Path("/sys/fs/cgroup")
    ancestry = []
    node = cgroup
    while True:
        ancestry.append(node)
        if node == root: break
        if root not in node.parents:
            raise RuntimeError("cgroup outside mounted hierarchy")
        node = node.parent
    validate_cpu_quota((cgroup / "cpu.max").read_text(),
                       [(p / "cpu.max").read_text() for p in ancestry[1:] if (p / "cpu.max").exists()], count)
    wanted = CONFIG["environment"]["memory_mb"] * 1024**2
    for index, path in enumerate(ancestry):
        if not (path / "memory.max").exists(): continue
        text = (path / "memory.max").read_text().strip()
        if (index == 0 and text != str(wanted)) or (text != "max" and int(text) < wanted):
            raise RuntimeError("Verifier or ancestor memory capacity differs from config")
    return [{"path": str(path), **{name: (path / name).read_text().strip()
             for name in ("cpu.max", "cpuset.cpus.effective", "cpuset.mems.effective", "memory.max")
             if (path / name).exists()}} for path in ancestry]


class Context:
    def __init__(self, *, development=False, workspace=None):
        self.development = development
        self.workspace = Path(workspace).resolve() if workspace is not None else None
        self.threads = CONFIG["environment"]["agent"]["cpus"] if development else LIMITS.threads
        entries = topology()
        self.cpus = select_logical_cpus(entries, self.threads)
        self.cgroup = None if development else cgroup_directory()
        capacities = verify_capacity(self.cgroup) if self.cgroup else []
        models = sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")})
        self.receipt = {"logical_cpus": self.cpus, "threads": self.threads,
                        "development_only": development,
                        "topology": [vars(e) for e in entries], "omp_proc_bind": "FALSE",
                        "cpu_models": models, "effective_affinity": sorted(os.sched_getaffinity(0)),
                        "cgroup_capacity_ancestry": capacities}
        if self.workspace:
            with tempfile.TemporaryFile(dir=self.workspace) as probe:
                require_disk_capture(probe)
        if self.cgroup:
            self.receipt["cgroup"] = str(self.cgroup)
            self.memory_check()

    def environment(self):
        return {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "HOME": "/nonexistent",
                **LIMITS.environment(), "OMP_NUM_THREADS": str(self.threads), "OMP_THREAD_LIMIT": str(self.threads)}

    def memory_check(self, capture=None):
        self.workspace_check(capture)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 1024**3:
            raise RuntimeError("parent exceeds 8-GiB memory ceiling")
        if self.cgroup:
            peak = int((self.cgroup / "memory.peak").read_text())
            if peak > 56 * 1024**3:
                raise RuntimeError("total measured memory exceeds 56-GiB acceptance target")
            for name in ("memory.current", "memory.peak", "memory.stat", "memory.events", "cpu.stat", "cpu.pressure", "memory.pressure"):
                self.receipt[name] = (self.cgroup / name).read_text()[:8192]

    def workspace_check(self, capture=None):
        if self.workspace is None:
            return
        total = 0
        if capture is not None:
            metadata = os.fstat(capture.fileno())
            total += max(metadata.st_size, metadata.st_blocks * 512)
        for directory, _, names in os.walk(self.workspace, followlinks=False):
            for name in names:
                try:
                    metadata = (Path(directory) / name).lstat()
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(metadata.st_mode):
                    total += max(metadata.st_size, metadata.st_blocks * 512)
        dataset = Path(os.environ.get("FRONTIER_DATASET_PATH", "/nonexistent"))
        if dataset.is_file() and self.workspace not in dataset.resolve().parents:
            total += dataset.stat().st_size
        self.receipt["accounted_workspace_bytes"] = total
        if total > CONFIG["environment"]["storage_mb"] * 1024**2:
            raise RuntimeError("32-GiB workspace ceiling exceeded")


def require_disk_capture(file):
    # fstatfs needs a platform-sized output buffer; only f_type is inspected.
    data = ctypes.create_string_buffer(256)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.fstatfs(file.fileno(), data) != 0:
        raise OSError(ctypes.get_errno(), "capture filesystem readback failed")
    if ctypes.c_long.from_buffer(data).value in {0x01021994, 0x858458F6}:
        raise RuntimeError("tmpfs/ramfs capture is prohibited")


def verify_source_tree(tree, pristine):
    import hashlib
    allowed = CONFIG["submission"]["allow_paths"]
    editable = lambda s: any(s == p or (p.endswith("/") and s.startswith(p)) for p in allowed)
    before = {p.relative_to(pristine).as_posix(): p for p in pristine.rglob("*") if p.is_file()}
    after = {}
    for path in tree.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("nonregular submission member")
        if path.is_file(): after[path.relative_to(tree).as_posix()] = path
    for name in before.keys() | after.keys():
        if editable(name): continue
        if name not in before or name not in after:
            raise ValueError("source-lock file set changed")
        if hashlib.sha256(before[name].read_bytes()).digest() != hashlib.sha256(after[name].read_bytes()).digest():
            raise ValueError("source-locked bytes changed")
