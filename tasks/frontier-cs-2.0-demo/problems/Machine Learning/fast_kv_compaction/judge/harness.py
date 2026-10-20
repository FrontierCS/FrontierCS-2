#!/usr/bin/env python3
"""Trusted GPU measurement of one implementation over a list of QuALITY articles.

Submitted code never runs in this process, which imports only the pristine
public tree and owns the answering model, the timer, the byte count, the logs
and the result file. The submitted method runs in compaction_worker.py, started
through the root launcher /opt/frontier/fastkv-worker in fresh network, PID and
mount namespaces as the unprivileged fastkv-worker user (no judge access, no
FRONTIER_* environment, private HOME, tree without data/). Between repetitions
the root cleaner kills every fastkv-worker process and removes its IPC objects
and temporary files, so nothing survives from one worker to the next.

Per article:
  seconds         median over TIMED_REPEATS fresh workers of the wall time from
                  sending the article to receiving its complete cache (worker
                  prefill + compaction + handoff, identical for every method);
                  each worker compacts a fixed warmup text first and then sees
                  every article once
  correct         official QA on EVERY repetition's cache, rebuilt from raw
                  bytes into plain tensors; the gate uses the minimum, so no
                  repetition can be cheap
  (final and calibration: TIMED_REPEATS repetitions after a warmup; development:
  --repeats 1 --no-warmup, one fresh worker, QA once)
  bytes           elements x element size of the returned tensors (largest
                  over repetitions)
  anchor_seconds  median of three trusted full-context HF prefills here
Free GPU memory is overwritten before every worker starts, so nothing a worker
left on the device is visible to the next one.
and per run engine_anchor_seconds: a trusted vLLM generation benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

JUDGE = Path(__file__).resolve().parent
sys.path.insert(0, str(JUDGE))
import gate  # noqa: E402
import wire  # noqa: E402

TIMED_REPEATS = 3
ANCHOR_REPEATS = 3
INFRA_EXIT_CODE = 75
MAX_NEW_TOKENS = 2048
QA_BATCH_SIZE = 20
MAX_LOG_BYTES = 256 * 1024
SUBSET = JUDGE / "data" / "quality_subset.jsonl"
LAUNCHER = "/opt/frontier/fastkv-worker"
# Host memory of the worker and everything it starts (resident set, summed);
# the watchdog kills the worker's namespace beyond it: a scored failure.
WORKER_MEMORY_MAX = 48 << 30
WATCHDOG_INTERVAL_SECONDS = 0.5
PROBE_HOG_LIMIT = 2 << 30
WORKER_READY_SECONDS = 1800
WORKER_REQUEST_SECONDS = 1800
# Time kept back from the submission for the trusted QA of each stored cache.
QA_RESERVE_SECONDS = 150


class Budget:
    """The submission's share of this run: work after `deadline` is the submission's timeout."""

    def __init__(self, seconds: float | None, reserve: float = 0.0):
        self.deadline = None if seconds is None else time.monotonic() + seconds - reserve

    def remaining(self, cap: float) -> float:
        if self.deadline is None:
            return cap
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise SubmissionFailure("the submission exceeded its time budget")
        return min(cap, left)
WARMUP_KEY = "__warmup__"
# Fixed public-domain warmup text (not a QuALITY article): it lets the worker
# compile kernels and fill allocator pools without seeing any graded input.
WARMUP_TEXT = " ".join(
    "The river ran past the old mill and the miller counted sacks of grain while "
    "the children watched the water wheel turn in the afternoon light." for _ in range(60))


class InfrastructureFailure(RuntimeError):
    """Setup or isolation failed for a reason unrelated to the submission."""


class SubmissionFailure(RuntimeError):
    """The submitted method failed, broke the protocol or returned an invalid cache."""


def _model_path() -> str:
    root = Path(os.environ.get("FRONTIER_MODEL_ROOT", "/opt/frontier/models"))
    path = root / "Qwen3-4B"
    if not (path / "config.json").is_file():
        raise InfrastructureFailure("offline Qwen3-4B checkpoint is missing")
    return str(path)


def timing_plan(articles: list[str], repeats: int = TIMED_REPEATS, warmup: bool = True) -> list[list[str]]:
    """Requests per worker process: a fresh process per repetition, each serving
    the warmup (if any) and then every article exactly once."""
    if len(set(articles)) != len(articles) or WARMUP_KEY in articles:
        raise ValueError("articles must be distinct")
    return [([WARMUP_KEY] if warmup else []) + list(articles) for _ in range(repeats)]


def worker_command(gpu: str, tree: Path, mode: str = "run") -> list[str]:
    """The namespace launcher, run as this (unprivileged) user; see verifier-gpu/fastkv-worker."""
    return [LAUNCHER, mode, gpu, str(tree)]


def harden_parent() -> None:
    """Not dumpable (its memory and /proc files stay closed to same-uid processes) and
    the subreaper of everything it starts, so no worker descendant escapes reaping."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_DUMPABLE 0, CHILD_SUBREAPER
        raise InfrastructureFailure("could not harden the trusted parent (prctl)")


def descendants(root: int) -> set[int]:
    """root and every live process below it (by parent pid, over this /proc)."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as stream:
                fields = stream.read().rsplit(b")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(entry))
        except (OSError, IndexError, ValueError):
            continue
    found, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid in found:
            continue
        found.add(pid)
        stack.extend(children.get(pid, ()))
    return found


def tmpfs_bytes(pids: set[int]) -> int:
    """Bytes held in the worker's own RAM-backed mounts (tmpfs, mqueue), read through
    /proc/<pid>/root of one of its processes; they count against its memory limit."""
    own = os.readlink("/proc/self/ns/mnt")
    for pid in sorted(pids):
        try:
            if os.readlink(f"/proc/{pid}/ns/mnt") == own:
                continue
            with open(f"/proc/{pid}/mounts") as stream:
                mounts = [line.split() for line in stream]
        except OSError:
            continue
        total = 0
        for fields in mounts:
            if len(fields) >= 3 and fields[2] in ("tmpfs", "mqueue"):
                point = fields[1].encode().decode("unicode_escape")
                try:
                    info = os.statvfs(f"/proc/{pid}/root{point}")
                except OSError:
                    continue
                total += (info.f_blocks - info.f_bfree) * info.f_frsize
        return total
    return 0


def resident_bytes(pids: set[int]) -> int:
    page = os.sysconf("SC_PAGE_SIZE")
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/statm") as stream:
                total += int(stream.read().split()[1]) * page
        except (OSError, IndexError, ValueError):
            continue
    return total


class Watchdog:
    """Kills a launcher's whole process tree once its summed RSS exceeds the limit."""

    def __init__(self, proc: subprocess.Popen, limit: int = WORKER_MEMORY_MAX):
        self.proc, self.limit = proc, limit
        self.tripped = False
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(WATCHDOG_INTERVAL_SECONDS) and self.proc.poll() is None:
            pids = descendants(self.proc.pid)
            used = resident_bytes(pids) + tmpfs_bytes(pids)
            self.peak = max(self.peak, used)
            if used > self.limit:
                self.tripped = True
                kill_tree(self.proc)
                return

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def kill_tree(proc: subprocess.Popen) -> None:
    for pid in descendants(proc.pid):
        try:
            os.kill(pid, 9)
        except (ProcessLookupError, PermissionError):
            pass


def reap(proc: subprocess.Popen, after_submission: bool, pid_namespace: str | None = None) -> None:
    """Nothing a worker started may outlive its repetition.

    Its namespaces end with the launcher (--kill-child), taking every process,
    the private IPC, network and tmpfs mounts with them. Anything still alive
    below the launcher after 30 seconds is killed; if it survives that, the fault
    is the submission's once its code ran, else the platform's.
    """
    pid_namespace = pid_namespace or _pid_namespace(proc.pid)
    deadline = time.monotonic() + 30
    while True:
        alive = (descendants(proc.pid) if proc.poll() is None else set()) | _members(pid_namespace)
        if not alive:
            return
        if time.monotonic() > deadline:
            if after_submission:
                raise SubmissionFailure("a worker process survived cleanup")
            raise InfrastructureFailure("worker cleanup failed")
        for pid in alive:
            try:
                os.kill(pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        _collect_zombies()


def _pid_namespace(launcher: int) -> str | None:
    """The PID namespace of the launcher's namespace init (its child), if any."""
    for pid in descendants(launcher) - {launcher}:
        try:
            namespace = os.readlink(f"/proc/{pid}/ns/pid")
        except OSError:
            continue
        if namespace != os.readlink("/proc/self/ns/pid"):
            return namespace
    return None


def _members(namespace: str | None) -> set[int]:
    """Processes still in that PID namespace, wherever they were reparented."""
    if namespace is None:
        return set()
    found = set()
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                if os.readlink(f"/proc/{entry}/ns/pid") == namespace:
                    found.add(int(entry))
            except OSError:
                continue
    return found


def _collect_zombies() -> None:
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _launch_probe(mode: str, gpu: str, tree: Path, timeout: int, limit: int | None = None) -> tuple[str, bool]:
    proc = subprocess.Popen(worker_command(gpu, tree, mode), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    watchdog = Watchdog(proc, limit) if limit else None
    try:
        out, _err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, _err = proc.communicate()
    finally:
        if watchdog:
            watchdog.stop()
        reap(proc, after_submission=False)
    return out.decode(errors="replace"), bool(watchdog and watchdog.tripped)


ISOLATION_MARKER = "fastkv-isolation-marker"


def probe_isolation(gpu: str, tree: Path) -> None:
    """Before any submitted code: the worker's view is isolated (worker-side checks),
    its writes do not persist outside its namespace, and the watchdog kills a hog."""
    output, _ = _launch_probe("probe", gpu, tree, timeout=300)
    if "FASTKV_ISOLATION_OK" not in output.split("\n"):
        raise InfrastructureFailure(f"worker isolation is unavailable: {output.strip()[-500:]}")
    for directory in ("/tmp", "/dev/shm", "/var/tmp"):
        if os.path.exists(os.path.join(directory, ISOLATION_MARKER)):
            raise InfrastructureFailure(f"a worker write persisted in {directory}")
    for mode in ("hog", "hog-tmpfs"):
        _output, tripped = _launch_probe(mode, gpu, tree, timeout=180, limit=PROBE_HOG_LIMIT)
        if not tripped:
            raise InfrastructureFailure(f"the worker memory watchdog did not stop a memory hog ({mode})")


def tree_digest(tree: Path) -> str:
    """Content and layout of the worker's (read-only) method tree."""
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(tree.rglob("*")):
        relative = path.relative_to(tree).as_posix().encode()
        if path.is_symlink():
            digest.update(b"L" + relative + b"\0" + os.readlink(path).encode() + b"\0")
        elif path.is_file():
            digest.update(b"F" + relative + b"\0" + path.read_bytes() + b"\0")
        else:
            digest.update(b"D" + relative + b"\0")
    return digest.hexdigest()


SCRUB_CHUNK_BYTES = 256 << 20


def scrub_gpu() -> int:
    """Overwrite all free GPU memory with random bytes, then release it."""
    import torch
    torch.cuda.empty_cache()
    blocks = []
    try:
        while True:
            try:
                block = torch.empty(SCRUB_CHUNK_BYTES, dtype=torch.uint8, device="cuda")
            except torch.cuda.OutOfMemoryError:
                break
            block.random_(0, 256)
            blocks.append(block)
        for size in (16 << 20, 1 << 20):  # fill the tail in smaller pieces
            while True:
                try:
                    block = torch.empty(size, dtype=torch.uint8, device="cuda")
                except torch.cuda.OutOfMemoryError:
                    break
                block.random_(0, 256)
                blocks.append(block)
        torch.cuda.synchronize()
        return sum(block.numel() for block in blocks)
    finally:
        del blocks
        torch.cuda.empty_cache()


def gpu_probe(mode: str, gpu: str, tree: Path) -> str:
    output, _ = _launch_probe(f"gpu-{mode}", gpu, tree, timeout=600)
    return output


def probe_gpu_scrub(gpu: str, tree: Path) -> None:
    """A worker process leaves a marker; after the scrub a fresh one must not see it."""
    if "FASTKV_GPU_WRITTEN" not in gpu_probe("write", gpu, tree):
        raise InfrastructureFailure("GPU residue probe could not write")
    scrub_gpu()
    output = gpu_probe("read", gpu, tree)
    lines = output.split("\n")
    if not any(line.startswith("FASTKV_GPU_RESIDUE ") for line in lines):
        errors = [line for line in lines if line.startswith("FASTKV_GPU_PROBE_ERROR ")]
        raise InfrastructureFailure("GPU residue probe could not read: "
                                    + (errors[0][:300] if errors else "no residue count reported"))
    if "FASTKV_GPU_RESIDUE 0" not in lines:
        raise InfrastructureFailure("GPU memory residue survives the scrub")


def worker_tree(patched: Path) -> Path:
    """World-readable copy of the patched tree without data/, in its own directory.

    The directory holds nothing else; the harness's own scratch stays 0700.
    """
    base = Path(tempfile.mkdtemp(prefix="fastkv-tree-", dir="/tmp"))
    tree = base / "worker" / "tree"
    shutil.copytree(patched, tree, symlinks=False,
                    ignore=lambda directory, names: ["data", ".git"] if Path(directory) == patched else [])
    for path in [tree, *tree.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    for path in (base, base / "worker"):
        path.chmod(0o755)
    return tree


class WorkerSession:
    """One worker process: ready, one warmup, then each article at most once.

    Its stderr is a pipe drained here into memory: the worker can never read it
    back, so it is not a channel between repetitions.
    """

    def __init__(self, command: list[str], budget: "Budget | None" = None):
        self.budget = budget or Budget(None)
        self.seen: set[str] = set()
        self.log = bytearray()
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, start_new_session=True)
        self.watchdog = Watchdog(self.proc)
        self.pid_namespace = None
        self._drain = threading.Thread(target=self._collect, daemon=True)
        self._drain.start()
        header = self._read(self.budget.remaining(WORKER_READY_SECONDS))[0]
        self.pid_namespace = _pid_namespace(self.proc.pid)
        if header.get("kind") != "ready":
            raise SubmissionFailure("worker did not become ready")

    def _collect(self) -> None:
        for chunk in iter(lambda: self.proc.stderr.read(65536), b""):
            self.log += chunk
            if len(self.log) > MAX_LOG_BYTES:
                del self.log[: len(self.log) - MAX_LOG_BYTES]

    def _read(self, timeout: float):
        result: dict = {}

        def reader():
            try:
                result["frame"] = wire.read_frame(self.proc.stdout)
            except Exception as exc:  # noqa: BLE001
                result["error"] = exc

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            self.kill()
            raise SubmissionFailure("worker timed out")
        if self.watchdog.tripped:
            raise SubmissionFailure("worker exceeded its host memory limit")
        if "error" in result:
            raise SubmissionFailure(f"worker protocol failure: {result['error']}")
        return result["frame"]

    def compact(self, key: str, article: str, target_size: int, seed: int, synchronize=None):
        """Timed request; returns (seconds, header, payloads)."""
        if key in self.seen:
            raise RuntimeError("an article may be sent to a worker only once")
        self.seen.add(key)
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        wire.write_frame(self.proc.stdin, {"kind": "compact", "article": article,
                                           "target_size": int(target_size), "seed": int(seed)})
        header, payloads = self._read(self.budget.remaining(WORKER_REQUEST_SECONDS))
        elapsed = time.perf_counter() - started
        if header.get("kind") == "error":
            raise SubmissionFailure(f"compaction failed: {str(header.get('message'))[:200]}")
        if header.get("kind") != "cache":
            raise SubmissionFailure("worker returned an unexpected frame")
        return elapsed, header, payloads

    def close(self) -> None:
        try:
            wire.write_frame(self.proc.stdin, {"kind": "exit"})
            self.proc.wait(timeout=120)
        except Exception:  # noqa: BLE001
            self.kill()
        finally:
            self.watchdog.stop()

    def kill(self) -> None:
        kill_tree(self.proc)


def rebuild_cache(header: dict, payloads: list[bytes], config) -> tuple:
    """New plain tensors on this process's GPU from raw bytes; validates the layout."""
    import torch
    specs = wire.tensor_specs(header)
    grouped = wire.layer_layout(specs, header.get("layers"), header.get("has_bias"))
    if len(grouped) != config.num_hidden_layers:
        raise SubmissionFailure("returned cache has the wrong number of layers")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    prefix = header.get("seq_len")
    layers = []
    position = 0

    def tensor(spec):
        nonlocal position
        value = torch.frombuffer(bytearray(payloads[position]), dtype=getattr(torch, spec["dtype"]))
        position += 1
        return value.reshape(spec["shape"]).to("cuda")

    for keys_spec, values_spec, bias_spec in grouped:
        heads, kept = keys_spec["shape"][1], keys_spec["shape"][2]
        if heads != config.num_key_value_heads or keys_spec["shape"][3] != head_dim:
            raise SubmissionFailure("returned cache has the wrong head layout")
        if type(prefix) is not int or kept > prefix:
            raise SubmissionFailure("returned cache keeps more tokens than the prompt has")
        keys, values = tensor(keys_spec), tensor(values_spec)
        if not (bool(torch.isfinite(keys).all()) and bool(torch.isfinite(values).all())):
            raise SubmissionFailure("returned keys or values are not finite")
        bias = tensor(bias_spec) if bias_spec else torch.zeros(keys.shape[:3], dtype=keys.dtype, device="cuda")
        # The decoder's cache layout: (keys, attention_bias, values) per layer.
        layers.append((keys, bias, values))
    return tuple(layers), wire.cache_bytes(specs)


class ReplayMethod:
    """Trusted stand-in handed to QAEvaluator: returns an already measured cache."""

    def __init__(self):
        self.cache = None
        self.expected_target = None

    def name(self) -> str:
        return "submission"

    def requires_preextracted_cache(self) -> bool:
        return False

    def returns_cache(self) -> bool:
        return True

    def compact_kv_cache(self, *, target_size, compute_stats=False, **_ignored):
        if compute_stats:
            raise RuntimeError("diagnostics are disabled in measured runs")
        if int(target_size) != self.expected_target:
            raise InfrastructureFailure("target size disagrees with the harness's computation")
        return self.cache, {}


def target_tokens(evaluator, article: dict, fraction: float) -> int:
    """QAEvaluator's own target-size arithmetic (memory-efficient path), computed here."""
    from evaluation.utils import compute_article_indices, format_context
    context = format_context(evaluator.tokenizer, article["article"], model_name=evaluator.model_name)
    seq_len = evaluator.tokenizer(context, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    article_len = len(compute_article_indices(evaluator.tokenizer, context, article["article"]))
    if article_len <= 0:
        return seq_len
    requested = max(1, int(article_len * fraction)) if 0 < fraction < 1 else max(0, int(fraction))
    return min(article_len, requested) + (seq_len - article_len)


def _prefill_anchor(evaluator, article: dict) -> float:
    import torch
    from evaluation.utils import extract_full_kv_cache
    times = []
    for _ in range(ANCHOR_REPEATS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        extract_full_kv_cache(evaluator.model, evaluator.tokenizer, article["article"], "cuda",
                              model_name=evaluator.model_name)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return float(statistics.median(times))


def _engine_anchor(pristine: Path) -> float:
    proc = subprocess.run([sys.executable, str(JUDGE / "anchor.py"), str(pristine)],
                          capture_output=True, timeout=1800)
    try:
        value = float(json.loads(proc.stdout.strip().splitlines()[-1])["engine_anchor_seconds"])
    except (IndexError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise InfrastructureFailure("engine anchor failed") from exc
    if proc.returncode != 0 or not value > 0:
        raise InfrastructureFailure("engine anchor failed")
    return value


def run(articles: list[str], role: str, seed: int, pristine: Path, patched: Path | None,
        gpu: str, scratch: Path, repeats: int = TIMED_REPEATS, warmup: bool = True,
        budget_seconds: float | None = None) -> dict:
    started = time.monotonic()
    target = 1.0
    if role == "solver":
        try:
            target = gate.declared_target_size(patched)
            gate.mode_for(target)
        except (gate.SubmissionDeclarationError, ValueError) as exc:
            raise SubmissionFailure(str(exc)) from exc
    os.chdir(pristine)
    sys.path.insert(0, str(pristine))
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        engine_anchor = _engine_anchor(pristine)  # its own process, released before workers start
        from evaluation.datasets import load_quality_data
        from evaluation.qa_evaluator import QAEvaluator
        from evaluation.utils import initialize_vllm, load_model_and_tokenizer
        from evaluation.configs.utils import load_query_config
        from compaction.compaction_methods.registry import get_compaction_method
        by_id = {row["article_id"]: row for row in load_quality_data(str(SUBSET))}
        model = _model_path()
        evaluator = QAEvaluator(model_name=model, device="cuda")
        evaluator.model, evaluator.tokenizer = load_model_and_tokenizer(model, device="cuda")
        device = torch.cuda.get_device_name(0)
        if role == "uncompressed":
            evaluator.vllm_model = initialize_vllm(model)
        else:
            harden_parent()
            tree = worker_tree(patched)
            probe_isolation(gpu, tree)
            probe_gpu_scrub(gpu, tree)
            tree_hash = tree_digest(tree)
    except (InfrastructureFailure, SubmissionFailure):
        raise
    except Exception as exc:  # nothing submitted has run yet
        raise InfrastructureFailure(f"{type(exc).__name__}: {exc}") from exc
    articles_data = []
    for article_id in articles:
        article = by_id.get(f"quality_{article_id}")
        if article is None:
            raise InfrastructureFailure(f"article {article_id} is not in the frozen subset")
        articles_data.append((article_id, article))
    rows = {article_id: {"workload": article_id, "questions": len(article["questions"]),
                         "anchor_seconds": _prefill_anchor(evaluator, article)}
            for article_id, article in articles_data}

    def answer(article, method, query_config):
        torch.manual_seed(seed)
        result = evaluator.evaluate_compaction_on_article(
            article_data=article, compaction_method=method, target_size=target,
            query_config=query_config, compute_stats=False, compute_perplexity=False,
            max_new_tokens=MAX_NEW_TOKENS, batch_size=QA_BATCH_SIZE)
        qa = result["qa_results"]
        expected = len(article["questions"])
        if qa.get("total_questions") != expected or len(qa.get("results_per_question", [])) != expected:
            raise InfrastructureFailure("incomplete question coverage")
        return int(qa["correct_answers"])

    if role == "uncompressed":
        method = get_compaction_method("original")
        for article_id, article in articles_data:
            rows[article_id]["correct"] = answer(article, method, load_query_config("repeat"))
        return {"device": device, "engine_anchor_seconds": engine_anchor,
                "rows": [rows[a] for a in articles]}

    by_key = dict(articles_data)
    # The submission may use what is left after the trusted setup, minus the QA
    # of every cache it will produce. No time left at all is the platform's fault.
    budget = Budget(None if budget_seconds is None else budget_seconds - (time.monotonic() - started),
                    reserve=QA_RESERVE_SECONDS * len(articles) * repeats)
    if budget.deadline is not None and budget.deadline <= time.monotonic():
        raise InfrastructureFailure("trusted setup left no time for the submission")
    targets = {article_id: target_tokens(evaluator, article, target) for article_id, article in articles_data}
    store = scratch / "caches"  # inside the harness's 0700 scratch
    store.mkdir(mode=0o700)
    timings = {article_id: [] for article_id in articles}
    logs = []
    try:
        for repetition, plan in enumerate(timing_plan(list(articles), repeats, warmup)):
            scrub_gpu()
            session = WorkerSession(worker_command(gpu, tree), budget)
            try:
                if warmup:
                    session.compact(plan[0], WARMUP_TEXT, 64, seed)
                for article_id in (plan[1:] if warmup else plan):
                    seconds, header, payloads = session.compact(
                        article_id, by_key[article_id]["article"], targets[article_id], seed,
                        synchronize=torch.cuda.synchronize)
                    timings[article_id].append(float(seconds))
                    path = store / f"{article_id}-{repetition}"
                    path.with_suffix(".json").write_text(json.dumps(header))
                    path.with_suffix(".bin").write_bytes(b"".join(payloads))
            finally:
                session.close()
                logs.append(bytes(session.log))
                reap(session.proc, after_submission=True, pid_namespace=session.pid_namespace)
            if tree_digest(tree) != tree_hash:
                raise SubmissionFailure("the method tree changed during a repetition")
    finally:
        shutil.rmtree(tree.parent.parent, ignore_errors=True)

    method = ReplayMethod()
    query_config = load_query_config("repeat")  # unused by the stand-in; QAEvaluator logs it
    for article_id in articles:
        corrects, footprints = [], []
        for repetition in range(repeats):
            path = store / f"{article_id}-{repetition}"
            header = json.loads(path.with_suffix(".json").read_text())
            blob = path.with_suffix(".bin").read_bytes()
            specs = wire.tensor_specs(header)
            offsets = [0]
            for spec in specs:
                offsets.append(offsets[-1] + spec["nbytes"])
            if offsets[-1] != len(blob):
                raise SubmissionFailure("stored cache is inconsistent")
            payloads = [blob[offsets[i]:offsets[i + 1]] for i in range(len(specs))]
            cache, footprint = rebuild_cache(header, payloads, evaluator.model.config)
            method.cache, method.expected_target = cache, targets[article_id]
            corrects.append(answer(by_key[article_id], method, query_config))
            footprints.append(footprint)
            del cache, method.cache
            torch.cuda.empty_cache()
        rows[article_id].update(
            seconds=float(statistics.median(timings[article_id])), timed_seconds=timings[article_id],
            bytes=max(footprints), correct=min(corrects), correct_per_repetition=corrects)
    return {"device": device, "engine_anchor_seconds": engine_anchor,
            "rows": [rows[a] for a in articles], "_worker_log_tail": logs[-1][-4096:].decode(errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("solver", "uncompressed"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--articles", required=True, help="comma-separated article ids")
    parser.add_argument("--pristine", required=True, help="unpatched public tree (imported here)")
    parser.add_argument("--patched", help="patched tree (run only inside the worker)")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--repeats", type=int, default=TIMED_REPEATS)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--budget", type=float, help="seconds this run may take in all (trusted setup included)")
    parser.add_argument("--output", required=True)
    options = parser.parse_args()
    if not options.gpu.isdigit() or options.repeats not in (1, TIMED_REPEATS):
        return INFRA_EXIT_CODE
    os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
    with tempfile.TemporaryDirectory(prefix="fastkv-harness-", dir="/tmp") as scratch:
        Path(scratch).chmod(0o700)
        try:
            result = run(options.articles.split(","), options.role, options.seed,
                         Path(options.pristine), Path(options.patched) if options.patched else None,
                         options.gpu, Path(scratch), options.repeats, not options.no_warmup, options.budget)
        except InfrastructureFailure as exc:
            print(f"infrastructure failure: {exc}", file=sys.stderr)
            return INFRA_EXIT_CODE
        except (SubmissionFailure, wire.ProtocolError) as exc:
            print(f"submission failure: {exc}", file=sys.stderr)
            return 1
    Path(options.output).write_text(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
