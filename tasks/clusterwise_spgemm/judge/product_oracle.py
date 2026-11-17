"""Incremental complete-capture checking against private ordered sparse expectations."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import struct

import numpy as np

MAGIC = 0x4357535047454D33
BLOCK_COLUMNS = 65536


class Expected:
    def __init__(self, root, limits):
        self.root = Path(root)
        self.meta = json.loads((self.root / "metadata.json").read_text())
        n, groups, nnz = (self.meta[k] for k in ("rows", "groups", "numeric_nnz"))
        if (self.meta["width"] != 8 or n != groups * 8 or
                not 0 < n <= ((limits["max_matrix_rows"] + 7) // 8) * 8 or
                not 0 <= nnz <= limits["max_oracle_product_nnz"]):
            raise ValueError("invalid trusted oracle dimensions")
        self.pointers = self._array("rowptr.i64le", "<i8", n + 1)
        self.columns = self._array("columns.i64le", "<i8", nnz)
        self.values = self._array("values.f64le", "<f8", nnz)
        self._validated_products = set()
        if (self.pointers[0] != 0 or self.pointers[-1] != nnz or
                np.any(np.diff(self.pointers) < 0)):
            raise ValueError("invalid trusted oracle pointers")

    def _array(self, name, dtype, count):
        path = self.root / name
        if path.stat().st_size != count * np.dtype(dtype).itemsize:
            raise ValueError("truncated trusted oracle array")
        return np.memmap(path, mode="r", dtype=dtype, shape=(count,)) if count else np.empty(0, dtype=dtype)

    def close(self):
        self._validated_products.clear()
        for value in (self.pointers, self.columns, self.values):
            if isinstance(value, np.memmap):
                value._mmap.close()


def _capture_unchanged(before, after):
    return (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns)


def _validated_capture(stream, expected, before, check_deadline):
    """Reuse a complete oracle result only for identical bytes in this workload."""
    if not expected._validated_products:
        return None
    stream.seek(0)
    digest = hashlib.sha256()
    size = 0
    while True:
        check_deadline()
        block = stream.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    check_deadline()
    if not _capture_unchanged(before, os.fstat(stream.fileno())):
        raise ValueError("capture changed during validation")
    key = (size, digest.hexdigest())
    if size != before.st_size or key not in expected._validated_products:
        return None
    return {"product_bytes": size, "product_sha256": key[1]}


def validate_product(stream, expected, limits, check_deadline=lambda: None):
    """Read and hash once; keep column IDs but only one bounded values block."""
    before = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode) or not 48 <= before.st_size <= limits["max_product_bytes"]:
        raise ValueError("missing, nonregular or oversized capture")
    cached = _validated_capture(stream, expected, before, check_deadline)
    if cached is not None:
        return cached
    stream.seek(0)
    digest = hashlib.sha256()

    def read_exact(size):
        check_deadline()
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("truncated capture")
        digest.update(data)
        return data

    magic, rows, cols, groups, width, count = struct.unpack("<Q5q", read_exact(48))
    n = expected.meta["rows"]
    if magic != MAGIC or (rows, cols, groups, width) != (n, n, n // 8, 8):
        raise ValueError("incorrect capture dimensions")
    size = 48 + 8 * (groups + 1 + count * 9)
    if count < 0 or count > n * groups or size != before.st_size:
        raise ValueError("incorrect complete-capture length")
    pointers = np.frombuffer(read_exact(8 * (groups + 1)), dtype="<i8")
    if pointers[0] != 0 or pointers[-1] != count or np.any(np.diff(pointers) < 0):
        raise ValueError("invalid capture group pointers")
    # One column array is bounded by the configured total capture length.
    columns = np.empty(count, dtype="<i8")
    for start in range(0, count, BLOCK_COLUMNS):
        stop = min(count, start + BLOCK_COLUMNS)
        columns[start:stop] = np.frombuffer(read_exact(8 * (stop - start)), dtype="<i8")
    for group in range(groups):
        check_deadline()
        lo, hi = map(int, pointers[group:group+2])
        ordered = np.sort(columns[lo:hi])
        if len(ordered) and (ordered[0] < 0 or ordered[-1] >= n or np.any(ordered[1:] == ordered[:-1])):
            raise ValueError("invalid or duplicate capture columns")
        # Zero storage is not prescribed: omitted values are numerically zero.
        for lane in range(8):
            row = group * 8 + lane
            p, q = map(int, expected.pointers[row:row+2])
            positions = np.searchsorted(ordered, expected.columns[p:q])
            found = positions < len(ordered)
            selected = np.flatnonzero(found)
            found[selected] &= ordered[positions[selected]] == expected.columns[p:q][selected]
            missing = expected.values[p:q][~found]
            if np.any(np.abs(missing) > 1e-9 + 1e-8 * np.abs(missing)):
                raise ValueError("missing significant expected value")
        del ordered
        for start in range(lo, hi, BLOCK_COLUMNS):
            stop = min(hi, start + BLOCK_COLUMNS)
            values = np.frombuffer(read_exact(8 * 8 * (stop - start)), dtype="<f8").reshape(-1, 8)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite capture values")
            ids = columns[start:stop]
            for lane in range(8):
                row = group * 8 + lane
                p, q = map(int, expected.pointers[row:row+2])
                positions = np.searchsorted(expected.columns[p:q], ids)
                found = positions < q - p
                selected = np.flatnonzero(found)
                found[selected] &= expected.columns[p + positions[selected]] == ids[selected]
                target = np.zeros(stop - start)
                target[found] = expected.values[p + positions[found]]
                if np.any(np.abs(values[:, lane] - target) > 1e-9 + 1e-8 * np.abs(target)):
                    raise ValueError("incorrect numeric capture")
    if stream.read(1):
        raise ValueError("trailing capture bytes")
    after = os.fstat(stream.fileno())
    if not _capture_unchanged(before, after):
        raise ValueError("capture changed during validation")
    check_deadline()
    if len(expected._validated_products) < 16:
        expected._validated_products.add((size, digest.hexdigest()))
    return {"product_bytes": size, "product_sha256": digest.hexdigest()}
