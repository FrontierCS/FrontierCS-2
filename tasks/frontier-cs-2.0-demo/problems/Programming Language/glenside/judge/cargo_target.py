from pathlib import Path

import pytest


class CargoLibraryItem(pytest.Item):
    def runtest(self):
        self.parent.target.build()


class CargoLibraryFile(pytest.File):
    def collect(self):
        yield CargoLibraryItem.from_parent(self, name='cargo-library-build')


class CargoLibraryTarget:
    def __init__(self, entrypoint, prepared_source, action):
        self.entrypoint = Path(entrypoint).resolve()
        self.prepared_source = Path(prepared_source).resolve()
        self.action = action
        self.completed = 0

    def pytest_collect_file(self, file_path, parent):
        if file_path.resolve() == self.entrypoint:
            collector = CargoLibraryFile.from_parent(parent, path=file_path)
            collector.target = self
            return collector
        return None

    def build(self):
        if self.completed:
            raise RuntimeError('library target was dispatched more than once')
        if self.entrypoint.name != 'lib.rs' or self.entrypoint.parent.name != 'src':
            raise ValueError('expected the retained Cargo library target')
        if (self.prepared_source / 'src/lib.rs').read_bytes() != self.entrypoint.read_bytes():
            raise ValueError('prepared library target differs from retained source')
        if (self.prepared_source / 'Cargo.toml').read_bytes() != (self.entrypoint.parents[1] / 'Cargo.toml').read_bytes():
            raise ValueError('prepared Cargo manifest differs from retained source')
        self.action()
        self.completed += 1
