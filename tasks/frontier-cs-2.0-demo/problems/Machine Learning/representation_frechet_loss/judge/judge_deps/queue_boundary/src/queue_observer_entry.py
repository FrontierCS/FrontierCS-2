"""Run either scientific phase with submitted imports confined to queue actors."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer_queue import trusted_queue_import


def bind_parent_lifetime():
    # torchrun starts observers in new sessions, outside its process group.
    # Its death must also stop this observer and its die-with-parent actors.
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot bind observer to launcher lifetime")
    if parent <= 1 or os.getppid() != parent:
        raise RuntimeError("Observer launcher exited before lifetime binding")


def validate_deadline(seconds, device, suite):
    gpu_limits = {"development": 7200, "final": 10800}
    if suite not in gpu_limits or device not in ("cpu", "cuda:0"):
        raise ValueError("Unknown observer runtime contract")
    limit = 120 if device == "cpu" else gpu_limits[suite]
    if not 1 <= seconds <= limit:
        raise ValueError("Observer deadline exceeds the declared task runtime")


def main():
    bind_parent_lifetime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--suite", choices=("development", "final"), required=True)
    parser.add_argument("--trusted-repo", type=Path, required=True)
    parser.add_argument("--spine-manifest", type=Path, required=True)
    parser.add_argument("--queue-source", type=Path, required=True)
    parser.add_argument("--queue-scratch", type=Path, required=True)
    parser.add_argument("--queue-python", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--observer-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args, upstream = parser.parse_known_args()
    if upstream[:1] == ["--"]:
        upstream = upstream[1:]
    validate_deadline(args.seconds, args.device, args.suite)
    if args.observer.is_symlink() or hashlib.sha256(args.observer.read_bytes()).hexdigest() != args.observer_sha256:
        raise ValueError("Trusted observer bytes changed")
    old_argv, old_path = sys.argv, sys.path[:]
    workers = []
    # Loading observe.py by file does not add its private sibling evaluator.
    sys.path.insert(0, str(args.observer.resolve().parent))
    try:
        with trusted_queue_import(
                args.trusted_repo, args.spine_manifest, args.queue_source, args.queue_scratch,
                python=args.queue_python, deadline=time.monotonic() + args.seconds,
                allow_cuda=args.device == "cuda:0", processes=workers):
            spec = importlib.util.spec_from_file_location("fd_trusted_observer", args.observer)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.argv = [str(args.observer), "--phase", args.phase, "--repo", str(args.trusted_repo),
                        "--report", str(args.report), "--", *upstream]
            module.main()
    except BaseException as error:
        diagnostic = getattr(error, "worker_stderr", "")
        if diagnostic:
            sys.stderr.write(diagnostic[:8192])
        raise
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        if args.report.is_file():
            if args.report.stat().st_size > 16 * 1024:
                raise ValueError("Trusted observer report exceeds evidence bound")
            report = json.loads(args.report.read_text())
            report["queue_worker_pids"] = [worker.process.pid for worker in workers]
            report["queue_worker_returncodes"] = [worker.process.returncode for worker in workers]
            report["queue_workers_closed"] = all(worker.closed and worker.process.returncode is not None
                                                for worker in workers)
            args.report.write_text(json.dumps(report, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
