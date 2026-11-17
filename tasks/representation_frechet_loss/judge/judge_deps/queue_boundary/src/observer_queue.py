"""Validate queue-only snapshots and install the private trusted import proxy."""

from contextlib import contextmanager
import importlib
from pathlib import Path
import shutil
import sys
from types import ModuleType

QUEUE = "frechet_distance/queue.py"
HELPERS = "frechet_distance/frontier_solution/"


def file_map(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Repository must be a real directory")
    result = {}
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError("Symlinks are not admitted into the queue source snapshot")
        if path.is_file():
            result[name] = path
        elif not path.is_dir():
            raise ValueError("Nonregular submission entry")
    return result


def prepare_queue_snapshot(submission, destination):
    """Copy only the editable queue import surface."""
    candidate = file_map(submission)
    selected = {name: path for name, path in candidate.items() if name == QUEUE or name.startswith(HELPERS)}
    if len(selected) > 256 or sum(p.stat().st_size for p in selected.values()) > 16 * 1024**2:
        raise ValueError("Queue source snapshot exceeds local preparation bounds")
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    for name, path in selected.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target, follow_symlinks=False)
    return destination


@contextmanager
def trusted_queue_import(trusted_root, queue_source, *, python, deadline, allow_cuda=False, processes=None):
    """Only this upstream root is added to the trusted import path."""
    from trusted_queue import bind_queue_class
    if any(name == "main_fd" or name == "frechet_distance" or name.startswith("frechet_distance.")
           for name in sys.modules):
        raise RuntimeError("Install queue wiring in a fresh observer before upstream imports")
    root = str(Path(trusted_root).resolve())
    if str(Path(queue_source).resolve()) in sys.path:
        raise ValueError("Submitted source must not be on trusted sys.path")
    workers = [] if processes is None else processes
    old_path, old_modules = sys.path[:], set(sys.modules)
    cls = bind_queue_class(queue_source, python=python, deadline=deadline,
                          processes=workers, allow_cuda=allow_cuda)
    proxy_module = ModuleType("frechet_distance.queue")
    proxy_module.FeatureQueue = cls
    proxy_module.__file__ = __file__
    sys.path.insert(0, root)
    try:
        importlib.import_module("frechet_distance")
        sys.modules["frechet_distance.queue"] = proxy_module
        yield cls
    finally:
        for worker in workers:
            worker.close()
        sys.path[:] = old_path
        for name in set(sys.modules) - old_modules:
            module = sys.modules.get(name)
            filename = getattr(module, "__file__", None)
            if (name == "frechet_distance" or name.startswith("frechet_distance.")
                    or (filename and Path(filename).is_relative_to(root))):
                sys.modules.pop(name, None)
