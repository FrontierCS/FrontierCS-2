"""Parent-owned clocks, authenticated observation sites and bounded process lifetimes.

Submitted code (the candidate build and every timed run of a built binary, reference included)
runs under the shared isolate launcher (isolate.run): a fresh Landlock + seccomp instance that reads
only the system directories and the paths named here, writes only its own private directories
(deleted after the run), has a rebuilt environment and no network, and leaves no process behind.
The judge's own trusted steps (baseline kernel object, oracle build and preprocessing) run as
plain bounded subprocesses.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

import isolate  # the shared launcher, beside edit_scope in the Verifier base
from sandbox_layers import AddressSpace, OutputsDir, Pinned
from link_guard import check as check_link
from product_oracle import validate_product
from runtime_context import CONFIG, LIMITS, ROOT, require_disk_capture

# The retained program is two translation units: ENTRYPOINT (main, loading,
# reordering, CSR construction, output; no editable header) and KERNEL (the
# editable headers and the fixed entry point frontier_clustered_spgemm).
ENTRYPOINT = Path("sample/ReorderedFlengthClusterSpGEMM.cpp")
KERNEL = Path("sample/ReorderedFlengthClusterSpGEMM_kernel.cpp")
FLAGS = ["-O3", "-fopenmp", "-std=c++17", "-include", "immintrin.h", "-DTBB", "-DHW_EXE"]
OBJECTS = [f"GTgraph/R-MAT/{name}.o" for name in ("graph", "utils", "init", "globals")]
LIBS = ["-LGTgraph/sprng2.0-lite/lib", "-lsprng", "-lm", "-ltbb", "-ltbbmalloc"]


class NativeProcessFailure(ValueError):
    def __init__(self, returncode, diagnostic=None):
        super().__init__(f"native process failed (returncode={returncode})")
        self.process_returncode = returncode
        self.native_diagnostic = diagnostic


class TrustedAggregateDeadline(RuntimeError):
    """The trusted judge budget expired before a full scientific result."""


def _resource_snapshot(context):
    """Observe counters without invoking or weakening any resource acceptance gate."""
    group = context.cgroup
    if group is None:
        return {"observed": False, "reason": "no_cgroup"}
    try:
        events = {}
        for line in (group / "memory.events").read_text().splitlines():
            key, value = line.split()
            events[key] = int(value)
        return {"observed": True, "cgroup": str(group), "memory_events": events,
                "memory_peak_bytes": int((group / "memory.peak").read_text()),
                "memory_current_bytes": int((group / "memory.current").read_text())}
    except (OSError, ValueError) as error:
        return {"observed": False, "reason": type(error).__name__}


def _native_process_diagnostic(context, binary, side, returncode, before):
    after = _resource_snapshot(context)
    with Path(binary).open("rb") as stream:
        binary_sha256 = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    return {"kind": "native_process_failure", "side": side,
            "process_returncode": returncode, "binary_sha256": binary_sha256,
            "resource_before": before, "resource_after": after,
            "limits": {"child_address_space_bytes": LIMITS.child_address_space_bytes,
                       "process_cpu_seconds": LIMITS.process_cpu_seconds,
                       "process_wall_seconds": LIMITS.process_wall_seconds,
                       "verifier_memory_bytes": CONFIG["environment"]["memory_mb"] * 1024**2,
                       "threads": context.threads}}


def remaining(deadline, phase, reserve=0.0, *, child=False):
    value = deadline - time.monotonic() - reserve
    if value <= 0:
        if child:
            raise TimeoutError(f"native {phase} deadline exceeded")
        raise TrustedAggregateDeadline(f"aggregate {phase} deadline exceeded")
    return value


def subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot enforce descendant reaping")


def terminate_group(process):
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait()
    deadline = time.monotonic() + 3.0
    while True:
        try:
            pid, _ = os.waitpid(-process.pid, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("owned descendants did not reap")
            time.sleep(0.01)


def run_command(command, cwd, environment, deadline, log, check_resources=lambda: None):
    """A trusted judge command (never submitted code), bounded by the deadline."""
    remaining(deadline, "trusted command", reserve=3)
    subreaper()
    with Path(log).open("w+b") as output:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                check_resources()
                try:
                    process.wait(timeout=min(0.25, remaining(deadline, "trusted command", reserve=3)))
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode: raise ValueError("trusted build/preprocessing command failed")
        finally:
            terminate_group(process)
            output.seek(0, os.SEEK_END)
            size = output.tell()
            output.seek(max(0, size-16384))
            tail = output.read(16384)
            output.seek(0); output.write(tail); output.truncate()
    remaining(deadline, "trusted command")


def instrument(tree):
    source = (tree / ENTRYPOINT).read_text()
    warmup = """    /* First execution is excluded from evaluation */
    const int warmups = 1;"""
    if source.count(warmup) != 1 or source.count("#define VALIDATE 0") != 1:
        raise ValueError("authentic observation sites changed")
    source = source.replace(warmup, "    /* Fresh process: one independently checked product. */\n    const int warmups = 0;")
    source = source.replace("#define VALIDATE 0", "#define VALIDATE 0\n#undef ITERS\n#define ITERS 1")
    validation = re.search(r"#if VALIDATE\n.*?\n#endif", source, re.S)
    if validation is None: raise ValueError("authentic validation site missing")
    snapshot = r'''
  auto &observed = C_csr_flength_cluster;
  FILE *capture = fopen(getenv("FRONTIER_OUTPUT_PATH"), "wb");
  if (!capture) return run.status = 91;
  const uint64_t magic = 0x4357535047454D33ULL;
  int64_t metadata[5] = {observed.csr_rows, observed.cols, observed.rows,
                         observed.cluster_sz, observed.nnzc};
  const int64_t max_bytes = @CAP@;
  if (observed.rows < 0 || observed.rows > @GROUPS@ || observed.nnzc < 0 ||
      observed.cluster_sz != 8 || observed.csr_rows != observed.rows * 8 ||
      observed.cols != observed.csr_rows ||
      observed.nnzc > (max_bytes - 48 - 8 * (observed.rows + 1)) / 72) return run.status = 92;
  if (fwrite(&magic, 8, 1, capture) != 1 || fwrite(metadata, 8, 5, capture) != 5 ||
      fwrite(observed.rowptr, 8, observed.rows + 1, capture) != size_t(observed.rows + 1) ||
      fwrite(observed.colids, 8, observed.nnzc, capture) != size_t(observed.nnzc) ||
      fwrite(observed.values, 8, observed.nnzc * 8, capture) != size_t(observed.nnzc * 8)) return run.status = 93;
  if (fclose(capture)) return run.status = 94;
'''.replace("@CAP@", str(LIMITS.max_product_bytes)).replace("@GROUPS@", str((CONFIG["benchmark"]["max_matrix_rows"] + 7)//8))
    source = source[:validation.start()] + snapshot + source[validation.end():]
    observed = tree / "sample/frontier_observed.cpp"
    observed.write_text("#include <cstdint>\n#include <cstdlib>\n" + source)
    return observed


# What the judge takes back from the build instance: the objects, binaries and map that
# link_guard inspects and the timed runs execute (regular files only, nothing else).
BUILD_ARTIFACTS = re.compile(r"(frontier-(main|main-observed|kernel)\.o|frontier-(authentic|observed)"
                             r"|frontier-observed\.map|GTgraph/R-MAT/[a-z]+\.o"
                             r"|GTgraph/sprng2\.0-lite/lib/libsprng\.a)")


def _commands():
    def compile_(path, target):
        return ["g++", *FLAGS, "-IGTgraph/sprng2.0-lite/include", "-c", str(path), "-o", target]

    def link(main, binary, *extra):
        return ["g++", *FLAGS, main, "frontier-kernel.o", *OBJECTS, *LIBS, *extra, "-o", binary]
    return compile_, link


def _sandboxed_build(tree, context, deadline):
    """The candidate-controlled compilation, in one isolate instance working on a copy of
    `tree`; the artifacts come back as a tar the judge unpacks by the allowlist above."""
    compile_, link = _commands()
    observed = "sample/frontier_observed.cpp"
    commands = [
        ["make", "rmat", "CC=gcc", "FLAGS=-O2"],
        compile_(ENTRYPOINT, "frontier-main.o"),
        compile_(observed, "frontier-main-observed.o"),
        compile_(KERNEL, "frontier-kernel.o"),
        link("frontier-main.o", "frontier-authentic"),
        link("frontier-main-observed.o", "frontier-observed", "-Wl,-Map=frontier-observed.map"),
    ]
    script = "set -e\n" + "".join(" ".join(f"'{a}'" for a in c) + " >>build.log 2>&1\n" for c in commands)
    script += ("tar -cf \"$1/build.tar\" frontier-main.o frontier-main-observed.o frontier-kernel.o "
               "frontier-authentic frontier-observed frontier-observed.map GTgraph/R-MAT/*.o "
               "GTgraph/sprng2.0-lite/lib/libsprng.a\n")
    timeout = remaining(deadline, "trusted command", reserve=3)
    environment = {k: v for k, v in context.environment().items() if k not in ("PATH", "HOME")}
    result = isolate.run(["sh", "-c", script, "build", "{out}"], tree=tree,
                              outputs=["build.tar"], timeout_seconds=timeout, env=environment,
                              path="/usr/bin:/bin", cpu_seconds=LIMITS.process_cpu_seconds,
                              layers=[AddressSpace(8 * 1024**3), Pinned(context.cpus)],
                              output_bytes=2 * 1024**3, file_bytes=LIMITS.max_product_bytes)
    if result.timed_out:
        remaining(deadline, "trusted command")  # raises when the build budget is spent
        raise TrustedAggregateDeadline("aggregate trusted command deadline exceeded")
    if not result.ok or not result.outputs.get("build.tar"):
        raise ValueError("trusted build/preprocessing command failed")
    with tempfile.TemporaryFile(dir=tree) as stream:
        stream.write(result.outputs["build.tar"])
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name).as_posix()
                if not member.isreg() or not BUILD_ARTIFACTS.fullmatch(name):
                    raise ValueError("unexpected build artifact")
                target = tree / name
                target.unlink(missing_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
                target.chmod(0o755 if name in ("frontier-authentic", "frontier-observed") else 0o644)


def build(tree, context, deadline):
    """Compile the two translation units separately, link main first, inspect.

    The main object (ENTRYPOINT, no editable header; observed and authentic
    variants) and the kernel object (KERNEL) use the unchanged flags, in an isolate
    instance. A baseline kernel object compiled by the judge from the pristine editable headers
    backs the library-symbol exemption. link_guard rejects a kernel object that
    defines locked framework or library code.
    """
    tree = Path(tree).resolve()
    instrument(tree)
    _sandboxed_build(tree, context, deadline)
    compile_, _ = _commands()
    baseline = tree / ".frontier-baseline"
    shutil.copytree(ROOT / "pristine", baseline)
    scratch = tree / ".compiler-tmp"; scratch.mkdir()
    environment = context.environment() | {"TMPDIR": str(scratch)}
    try:
        run_command(compile_(baseline / KERNEL, "frontier-kernel-baseline.o"), tree, environment,
                    deadline, tree/"build-baseline.log", context.memory_check)
    except ValueError as error:
        raise RuntimeError("pristine kernel baseline failed to build") from error
    reasons = check_link(tree/"frontier-kernel.o", tree/"frontier-main-observed.o",
                         tree/"frontier-kernel-baseline.o", tree/"frontier-observed.map",
                         tree/"frontier-observed", tree)
    if reasons:
        print("clusterwise private link rejection: " + json.dumps(reasons[:16]), file=sys.stderr, flush=True)
        raise ValueError(reasons[0][0])
    return tree / "frontier-observed"


def build_oracle(scratch, context, deadline):
    binary = scratch / "ordered-preprocess"
    command = ["g++", "-O3", "-std=c++17", str(ROOT / "oracle/preprocess.cpp"), "-o", str(binary)]
    run_command(command, scratch, context.environment() | {"TMPDIR": str(scratch)},
                deadline, scratch/"oracle-build.log", context.memory_check)
    return binary


def preprocess(binary, matrix, order, destination, context, deadline):
    destination.mkdir(mode=0o700)
    limit = CONFIG["benchmark"]
    command = [str(binary), str(matrix), str(order), str(destination), *map(str, (
        limit["max_matrix_rows"], limit["max_expanded_input_nnz"], limit["max_oracle_product_nnz"],
        limit["max_scalar_operations"], limit["max_product_bytes"]))]
    run_command(command, destination, context.environment(),
                deadline, destination.parent / "oracle-run.log", context.memory_check)
    context.memory_check()


def sample(binary, matrix, order, expected, context, deadline, scratch, *, side):
    if side not in {"candidate", "reference"}:
        raise ValueError("Unknown trusted sample side")
    binary, matrix, order = (Path(p).resolve() for p in (binary, matrix, order))
    context.memory_check()
    budget = remaining(deadline, "evaluation", reserve=5)
    wall = min(LIMITS.process_wall_seconds, budget)
    environment = {k: v for k, v in context.environment().items() if k not in ("PATH", "HOME")}
    resources_before = _resource_snapshot(context)
    landing = Path(tempfile.mkdtemp(prefix="product-", dir=scratch))
    try:
        return _sample(binary, matrix, order, expected, context, deadline, side, wall,
                       environment, resources_before, landing)
    finally:
        shutil.rmtree(landing, ignore_errors=True)


def _sample(binary, matrix, order, expected, context, deadline, side, wall, environment,
            resources_before, landing):
    # One fresh instance per run on the selected CPUs (it cannot change its affinity): threads
    # but no new processes (syscalls="no_fork"), the old 48 GiB address space and CPU limits; the
    # product goes to the instance's own out/product and is moved (not read) into `landing`;
    # the time is the launcher's fork-to-exit stamp of the binary alone.
    result = isolate.run(
        [str(binary), "text", str(matrix), str(order), "8", str(context.threads)],
        read_paths=[str(binary), str(matrix), str(order)],
        env=environment | {"FRONTIER_OUTPUT_PATH": "{out}/product"}, path="/usr/bin:/bin",
        timeout_seconds=wall, syscalls="no_fork", cpu_seconds=LIMITS.process_cpu_seconds,
        layers=[Pinned(context.cpus), AddressSpace(LIMITS.child_address_space_bytes),
                OutputsDir(landing, ["product"], LIMITS.max_product_bytes)],
        file_bytes=LIMITS.max_product_bytes, output_bytes=LIMITS.max_product_bytes,
        disk_bytes=LIMITS.max_product_bytes + (1 << 30))
    if result.timed_out:
        if wall < LIMITS.process_wall_seconds:
            raise TrustedAggregateDeadline("aggregate evaluation deadline exceeded")
        raise TimeoutError("native process deadline exceeded")
    elapsed_ms = result.command_seconds * 1000.0
    if result.returncode or result.limit:
        returncode = result.returncode or -signal.SIGKILL
        diagnostic = _native_process_diagnostic(
            context, binary, side, returncode, resources_before)
        diagnostic["sandbox_limit"] = result.limit
        print("clusterwise private native diagnostic: " + json.dumps(diagnostic),
              file=sys.stderr, flush=True)
        raise NativeProcessFailure(returncode, diagnostic)
    product = result.outputs.get("product") or landing / "missing"
    with (product.open("rb") if product.is_file() else tempfile.TemporaryFile(dir=landing)) as capture:
        require_disk_capture(capture)
        context.memory_check(capture)
        result = validate_product(capture, expected, CONFIG["benchmark"],
                                  lambda: remaining(deadline, "validation", reserve=5))
        context.memory_check(capture)
    result["elapsed_ms"] = elapsed_ms
    return result
