"""Execute a registered Go source entrypoint through Python's runpy protocol.

runpy supports path importers as well as Python files. This adapter supplies a
single-use importer for one logical /app path, bound to a source file in the
patched repository. Its module code delegates to the real Go toolchain; it does
not interpret, transform, simulate, or replace any of the upstream Go code.
"""
from contextlib import contextmanager
import importlib.abc
import importlib.machinery
from pathlib import Path, PurePosixPath
import sys


class _GoSourceLoader(importlib.abc.Loader):
    def __init__(self, logical_path):
        self.logical_path = logical_path

    def get_code(self, fullname):
        if fullname != '__main__':
            raise ImportError('the Go adapter only executes the registered main')
        # The callback is private evaluator state passed via runpy.init_globals.
        # This code object contains no estimator logic or submitted Python code.
        return compile('_execute_registered_go_source()', self.logical_path, 'exec')

    def get_source(self, fullname):
        return None

    def is_package(self, fullname):
        return False


class _GoSourceFinder(importlib.abc.PathEntryFinder):
    def __init__(self, logical_path):
        self.logical_path = logical_path

    def find_spec(self, fullname, target=None):
        if fullname != '__main__':
            return None
        return importlib.machinery.ModuleSpec(
            fullname, _GoSourceLoader(self.logical_path),
            origin=self.logical_path,
        )


@contextmanager
def registered_go_entrypoint(tree, relative_path, run, deadline):
    """Temporarily register exactly one Go file from the patched source tree.

The evaluator is single-threaded. The scoped path-importer entry is restored on
success, compiler failure, timeout, and exceptions. No sys.path hook is changed.
"""
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.go':
        raise ValueError('invalid Go entrypoint path')
    root = Path(tree).resolve(strict=True)
    source = (root / relative).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError('Go entrypoint is outside the patched tree')
    logical = '/app/' + relative.as_posix()
    calls = 0

    def execute():
        nonlocal calls
        if calls:
            raise RuntimeError('Go entrypoint must execute exactly once')
        calls += 1
        return run(['go', 'run', '-p=1', source.name], source.parent, deadline)

    missing = object()
    previous = sys.path_importer_cache.get(logical, missing)
    sys.path_importer_cache[logical] = _GoSourceFinder(logical)
    try:
        yield {'_execute_registered_go_source': execute}
        if calls != 1:
            raise RuntimeError('registered Go entrypoint was not executed')
    finally:
        if previous is missing:
            sys.path_importer_cache.pop(logical, None)
        else:
            sys.path_importer_cache[logical] = previous
