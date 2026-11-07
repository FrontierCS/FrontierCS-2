"""Bounded operator evidence archive writer with no runtime-side imports."""
from pathlib import Path
import tarfile


def write_evidence(directory, destination, maximum):
    import zstandard

    class BoundedWriter:
        def __init__(self, stream):
            self.stream, self.count = stream, 0
        def write(self, data):
            self.count += len(data)
            if self.count > maximum:
                raise ValueError('compressed evidence exceeds publisher bound')
            return self.stream.write(data)
        def flush(self):
            self.stream.flush()

    directory, destination = Path(directory), Path(destination)
    pending = destination.with_suffix('.pending')
    try:
        with pending.open('xb') as stream:
            with zstandard.ZstdCompressor().stream_writer(
                    BoundedWriter(stream), closefd=False) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|') as tar:
                    for path in sorted(directory.rglob('*')):
                        if path.is_symlink():
                            raise ValueError('unexpected evidence symlink')
                        if path.is_file():
                            tar.add(path, arcname=str(path.relative_to(directory)),
                                    recursive=False)
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
