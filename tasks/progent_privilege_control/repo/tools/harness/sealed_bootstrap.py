"""Source-local prototype: execute public Python only after kernel confinement."""
import ctypes
import fcntl
import importlib
import importlib.abc
import importlib.resources.abc
import importlib.util
import io
import json
import linecache
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import stat
import sys
import sysconfig

spec = importlib.util.spec_from_file_location('kernel_boundary', Path(__file__).with_name('kernel_boundary.py'))
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)

MAX_FILE = 2 * 1024**2
MAX_TOTAL = 96 * 1024**2
MAX_FILES = 8192


class Snapshot:
    """Read immutable, public source inputs without following symlinks."""
    def __init__(self):
        self.files = {}
        self.total = 0

    def add_file(self, directory, name, virtual):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                     dir_fd=directory)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE:
                raise ValueError('unsafe or oversized public file')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                data = stream.read(MAX_FILE + 1)
            after = os.fstat(fd)
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or len(data) != after.st_size):
                raise ValueError('public source changed during snapshot')
        finally:
            os.close(fd)
        self.total += len(data)
        if self.total > MAX_TOTAL or len(self.files) >= MAX_FILES:
            raise ValueError('public source snapshot exceeds bounds')
        if virtual in self.files:
            raise ValueError('ambiguous public source mapping')
        self.files[virtual] = data

    def add_tree(self, root, virtual, *, resources=False, trusted=False):
        # Root is selected by the trusted launcher. No candidate process exists
        # yet; the patched tree is private to this evaluation and validated.
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError('public root must be an ordinary directory')
        for directory, dirs, names, fd in os.fwalk(root, follow_symlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {'__pycache__', '.git', 'site-packages', 'dist-packages'})
            for name in dirs:
                if not stat.S_ISDIR(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
                    raise ValueError('symlinked public directory')
            relative = Path(directory).relative_to(root).as_posix()
            prefix = virtual + (('/' + relative) if relative != '.' else '')
            for name in sorted(names):
                if not name.endswith('.py') and not (resources and name.endswith(('.json', '.txt', '.yaml'))):
                    continue
                if trusted and os.path.islink(Path(directory) / name):
                    # A system packaging alias is not part of the snapshot.
                    continue
                self.add_file(fd, name, prefix + '/' + name)


class MemoryResourceReader(importlib.resources.abc.ResourceReader):
    def __init__(self, loader, directory):
        self.loader, self.directory = loader, directory

    def open_resource(self, resource):
        if PurePosixPath(resource).name != resource:
            raise FileNotFoundError(resource)
        return io.BytesIO(self.loader.get_data(self.directory + '/' + resource))

    def resource_path(self, resource):
        raise FileNotFoundError('resources are memory-backed')

    def is_resource(self, name):
        return self.directory + '/' + name in self.loader.files

    def contents(self):
        prefix = self.directory + '/'
        return iter(sorted({name[len(prefix):].split('/')[0]
                            for name in self.loader.files if name.startswith(prefix)}))


class PublicModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, files, roots):
        self.files, self.roots = files, roots
        self.selected = {}

    def find_spec(self, fullname, path=None, target=None):
        # Honor the search path of an already resolved package, rather than
        # mapping nested candidate directories into unrelated top-level names.
        locations = path if path is not None else self.roots
        leaf = fullname.rsplit('.', 1)[-1]
        for root in locations:
            stem = root + '/' + leaf
            package, module = stem + '/__init__.py', stem + '.py'
            if package in self.files:
                selected, is_package = package, True
            elif module in self.files:
                selected, is_package = module, False
            elif any(name.startswith(stem + '/') for name in self.files):
                selected, is_package = None, True
            else:
                continue
            self.selected[fullname] = selected
            result = importlib.util.spec_from_loader(fullname, self, is_package=is_package)
            result.origin = selected or stem
            result.has_location = selected is not None
            if is_package:
                result.submodule_search_locations = [stem]
            return result
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        filename = self.selected[module.__name__]
        if filename is None:
            return
        data = self.files[filename]
        # compile(bytes) preserves Python's declared source-encoding semantics.
        source = importlib.util.decode_source(data)
        linecache.cache[filename] = (len(data), None, source.splitlines(True), filename)
        exec(compile(data, filename, 'exec'), module.__dict__)

    def get_source(self, fullname):
        path = self.selected.get(fullname)
        return importlib.util.decode_source(self.files[path]) if path else None

    def get_filename(self, fullname):
        return self.selected[fullname]

    def get_data(self, path):
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    def get_resource_reader(self, fullname):
        filename = self.selected.get(fullname)
        if filename and filename.endswith('/__init__.py'):
            return MemoryResourceReader(self, filename.rsplit('/', 1)[0])
        return None


def preload_dependencies():
    # Installed, immutable libraries only. -I excludes the candidate and cwd.
    names = ('math itertools operator functools collections typing inspect re ast copy enum '
             'json decimal fractions statistics heapq bisect array struct hashlib pickle pathlib '
             'dataclasses random weakref unicodedata datetime timeit cmath logging '
             'sre_parse sre_constants base64 io marshal ctypes mmap binascii zlib bz2 lzma '
             'csv itertools functools queue select socket ssl sqlite3 xml.etree.ElementTree '
             'pydantic jsonschema docstring_parser z3 yaml')
    for name in names.split():
        importlib.import_module(name)
    # Resolve the confinement library before the native-thread precondition.
    ctypes.CDLL('libseccomp.so.2', mode=os.RTLD_NOW)
    import pydantic, jsonschema, docstring_parser, z3, yaml
    pydantic.create_model('_InertDependencyWarmup', amount=(int, ...))(amount=3)
    jsonschema.validate({'amount': 3}, {'type': 'object'})
    docstring_parser.parse('Example.')
    yaml.safe_load('amount: 3')
    solver = z3.Solver()
    solver.add(z3.Int('_inert_dependency_warmup') > 2)
    solver.check()


def verify_launch_context(expected_parent):
    # Only the trusted PolicyWorker launcher supplies these descriptors.
    # A bootstrap accidentally invoked with a regular file/socket on stdio
    # must not expose it to candidate code.
    stdin, stdout, stderr = (os.fstat(number) for number in (0, 1, 2))
    null = os.stat('/dev/null')
    if (not stat.S_ISFIFO(stdin.st_mode) or not stat.S_ISFIFO(stdout.st_mode)
        or (stdin.st_dev, stdin.st_ino) == (stdout.st_dev, stdout.st_ino)
        or fcntl.fcntl(0, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY
        or fcntl.fcntl(1, fcntl.F_GETFL) & os.O_ACCMODE != os.O_WRONLY
        or not stat.S_ISCHR(stderr.st_mode) or stderr.st_rdev != null.st_rdev):
        raise RuntimeError('unexpected policy transport descriptors')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError('cannot bind policy child lifetime to trusted parent')
    if os.getppid() != expected_parent:
        raise RuntimeError('trusted launcher exited before boundary installation')


def main():
    if len(sys.argv) != 5:
        raise ValueError('expected public tree, worker, protocol and parent PID')
    verify_launch_context(int(sys.argv[4]))
    tree, worker, protocol = map(Path, sys.argv[1:4])
    preload_dependencies()
    snapshot = Snapshot()
    snapshot.add_tree(tree, '/candidate')
    for path, virtual in [(worker, '/bridge/worker.py'), (protocol, '/bridge/policy_protocol.py')]:
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            snapshot.add_file(fd, path.name, virtual)
        finally:
            os.close(fd)
    # Pure Python modules may import lazily without gaining path authority.
    stdlib = Path(sysconfig.get_path('stdlib'))
    snapshot.add_tree(stdlib, str(stdlib), trusted=True)
    roots = ['/bridge', '/candidate', '/candidate/agentdojo/src', str(stdlib)]
    packages = ('pydantic pydantic_core annotated_types typing_extensions typing_inspection '
                'docstring_parser jsonschema jsonschema_specifications referencing attrs attr rpds z3 yaml')
    for name in packages.split():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError('required installed dependency missing: ' + name)
        path = Path(spec.origin)
        root = path.parent.parent if spec.submodule_search_locations else path.parent
        if str(root) not in roots:
            roots.append(str(root))
        if spec.submodule_search_locations:
            snapshot.add_tree(path.parent, str(path.parent), resources=True, trusted=True)
        elif path.suffix == '.py':
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                snapshot.add_file(fd, path.name, str(path))
            finally:
                os.close(fd)
    files = snapshot.files
    worker_source = files['/bridge/worker.py']
    for entry in os.listdir('/proc/self/fd'):
        if int(entry) > 2:
            try:
                os.close(int(entry))
            except OSError:
                pass
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
    os.chdir('/')
    os.environ.clear()
    os.environ.update(PATH='/usr/bin:/bin', HOME='/candidate', PYTHONDONTWRITEBYTECODE='1', SECAGENT_GENERATE='False')
    sys.dont_write_bytecode = True
    sys.path[:] = roots
    sys.meta_path.insert(0, PublicModules(files, roots))
    blocked = signal.valid_signals() - {signal.SIGKILL, signal.SIGSTOP}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    if len(os.listdir('/proc/self/task')) != 1:
        raise RuntimeError('trusted dependency preload left an unconstrained thread')
    boundary.confine([])
    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    namespace = {'__name__': '__main__', '__file__': '/bridge/worker.py'}
    exec(compile(worker_source, namespace['__file__'], 'exec'), namespace)


if __name__ == '__main__':
    main()
