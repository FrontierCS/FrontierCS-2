"""Explicit local runtime/device allowlists for the confined queue process."""

import json
from pathlib import Path
import sys
import sysconfig

def runtime_paths(source, scratch, tools):
    bindings = Path(sys.prefix) / "runtime-roots.json"
    roots = json.loads(bindings.read_text()) if bindings.is_file() else []
    reads = [Path(sys.base_prefix), Path(sys.prefix), tools, source]
    reads += [Path(p) for p in roots]
    # Published image Python uses installed site packages; local CPU venv uses
    # exact retained-wheel roots. Neither profile mounts task-private assets.
    reads += [Path(sysconfig.get_path(name)) for name in ("purelib", "platlib", "stdlib")]
    reads += [Path(p) for p in ("/usr", "/lib", "/lib64", "/proc/self", "/proc/cpuinfo",
                               "/proc/meminfo", "/etc/ld.so.cache", "/dev/urandom") if Path(p).exists()]
    writes = [scratch, Path("/dev/null")]
    return list(dict.fromkeys(p.resolve(strict=True) for p in reads)), writes
