#!/usr/bin/env python3
"""Public two-CPU debugging adapter; scientific measurements run in the Verifier."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "frontier-runtime.json").read_text())


def run_build(command, environment, deadline):
    timeout = deadline - time.monotonic()
    if timeout <= 0: raise TimeoutError("aggregate build deadline exceeded")
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        if code: raise subprocess.CalledProcessError(code, command)
    finally:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()


def main():
    count = POLICY["debug_logical_cpus"]
    selected, cores, extra = [], set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core = tuple(int((topology / name).read_text()) for name in ("physical_package_id", "core_id"))
        if core not in cores: selected.append(cpu); cores.add(core)
        else: extra.append(cpu)
    selected = (selected + extra)[:count]
    if len(selected) != count: raise RuntimeError("insufficient debugging logical CPUs")
    os.sched_setaffinity(0, selected)
    environment = dict(os.environ, OMP_NUM_THREADS=str(count), OMP_THREAD_LIMIT=str(count),
                       OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE", OMP_PROC_BIND="FALSE")
    os.chdir(ROOT)
    if sys.argv[1:] == ["build"]:
        deadline = time.monotonic() + POLICY["build_seconds"]
        run_build(["make", "rmat", "CC=gcc", "FLAGS=-O2"], environment, deadline)
        # Same two translation units, flags and link order as the Verifier:
        # the main program (no editable header) and the kernel are separate
        # objects, linked main first.
        flags = ["-O3", "-fopenmp", "-std=c++17", "-include", "immintrin.h", "-DTBB", "-DHW_EXE",
                 "-IGTgraph/sprng2.0-lite/include"]
        (ROOT / "bin").mkdir(exist_ok=True)
        for source, target in (("sample/ReorderedFlengthClusterSpGEMM.cpp", "bin/main.o"),
                               ("sample/ReorderedFlengthClusterSpGEMM_kernel.cpp", "bin/kernel.o")):
            run_build(["g++", *flags, "-c", source, "-o", target], environment, deadline)
        objects = [f"GTgraph/R-MAT/{name}.o" for name in ("graph", "utils", "init", "globals")]
        run_build(["g++", *flags, "bin/main.o", "bin/kernel.o", *objects,
                   "-LGTgraph/sprng2.0-lite/lib", "-lsprng", "-lm", "-ltbb", "-ltbbmalloc",
                   "-o", "bin/ReorderedFlengthClusterSpGEMM_hw"], environment, deadline)
        return
    elif len(sys.argv) == 5 and sys.argv[1] == "bench" and sys.argv[4] == str(POLICY["cluster_width"]):
        command = [str(ROOT / "bin/ReorderedFlengthClusterSpGEMM_hw"), "text", sys.argv[2], sys.argv[3],
                   sys.argv[4], str(count)]
    else:
        raise ValueError("use build, or bench <matrix.mtx> <order.txt> 8")
    os.execvpe(command[0], command, environment)


if __name__ == "__main__": main()
