"""Trusted native-loader admission checks; rejection samples do not execute."""
import deployment_test_guard  # noqa: F401  # isort: skip

from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest

JUDGE = (Path(__file__).resolve().parents[3] / 'tasks/frontier-cs-2.0-demo'
         / 'problems/Bioinformatics/edlib_exact_alignment/judge')
sys.path.insert(0, str(JUDGE))
import native_cmake as native


def elf_bytes(interpreter=b'/lib64/ld-linux-x86-64.so.2\0', *, machine=62,
              kind=3, dynamic=True):
    count = 2 if dynamic else 1
    offset = 64 + count * 56
    header = struct.pack('<16sHHIQQQIHHHHHH', b'\x7fELF\x02\x01\x01' + b'\0' * 9,
                         kind, machine, 1, 0, 64, 0, 0, 64, 56, count, 0, 0, 0)
    interp = struct.pack('<IIQQQQQQ', 3, 4, offset, 0, 0,
                         len(interpreter), len(interpreter), 1)
    dyn = struct.pack('<IIQQQQQQ', 2, 4, offset + len(interpreter),
                      0, 0, 16, 16, 8) if dynamic else b''
    return header + interp + dyn + interpreter + (b'\0' * 16 if dynamic else b'')


def test_system_dynamic_executable_admitted():
    native.validate_native_elf(Path('/usr/bin/true'))


@pytest.mark.parametrize('contents', [
    b'#!/bin/sh\nexit 0\n',
    elf_bytes(machine=3),
    elf_bytes(kind=1),
    elf_bytes(dynamic=False),
    elf_bytes(interpreter=b'/tmp/loader\0'),
    elf_bytes(interpreter=b'/lib64/ld-linux-x86-64.so.2'),
    elf_bytes(interpreter=b'/lib64/ld-linux-x86-64.so.2\0extra\0'),
    elf_bytes()[:80],
])
def test_invalid_loader_headers_rejected(tmp_path, contents):
    binary = tmp_path / 'header-only'
    binary.write_bytes(contents)
    with pytest.raises(native.NativeError, match='dynamic x86-64 ELF'):
        native.validate_native_elf(binary)


def test_static_executable_rejected_before_native_dispatch(tmp_path):
    source = tmp_path / 'static.c'
    source.write_text('int main(void) { return 42; }\n')
    binary = tmp_path / 'static'
    subprocess.run(['cc', '-static', str(source), '-o', str(binary)], check=True,
                   capture_output=True)
    receipts = []
    with pytest.raises(native.NativeError, match='dynamic x86-64 ELF'):
        native.run_bounded([binary], tmp_path, time.monotonic() + 30,
                           receipts=receipts)
    assert receipts == []
