"""Launch-surface and static-storage inspection on small compiled programs.

Compiles, never executes:

    python3 -m pytest -q -p no:cacheprovider tests/maintenance/edlib_exact_alignment/test_launch_surface.py
"""
import deployment_test_guard  # noqa: F401  # isort: skip

from pathlib import Path
import subprocess
import sys

import pytest

JUDGE = (Path(__file__).resolve().parents[3] / 'tasks/frontier-cs-2.0-demo'
         / 'problems/Bioinformatics/edlib_exact_alignment/judge')
sys.path.insert(0, str(JUDGE))
import elf_surface as surface

BASE = '#include <cstdlib>\n#include <vector>\nint work(int n) { std::vector<int> v(n, 1); return v[0]; }\n'
MAIN = 'int work(int);\nint main(int argc, char**) { return work(argc) - 1; }\n'


def build(directory, name, library):
    directory.mkdir(exist_ok=True)
    (directory / 'lib.cpp').write_text(library)
    (directory / 'main.cpp').write_text(MAIN)
    out = directory / name
    subprocess.run(['c++', '-std=c++14', '-O2', str(directory / 'main.cpp'), str(directory / 'lib.cpp'), '-o', str(out)], check=True)
    subprocess.run(['c++', '-std=c++14', '-O2', '-c', str(directory / 'lib.cpp'), '-o', str(directory / 'lib.o')], check=True)
    return out


def test_two_builds_of_one_source_have_the_same_surface(tmp_path):
    surface.require_same_launch_surface(build(tmp_path / 'a', 'a', BASE), build(tmp_path / 'b', 'b', BASE))


def test_a_different_body_with_new_template_instances_is_accepted(tmp_path):
    other = BASE.replace('std::vector<int> v(n, 1); return v[0];',
                         'struct Local { short a; }; std::vector<Local> v(n); std::vector<unsigned short> w(n, 1); static int calls; ++calls; return w[0] + v[0].a;')
    surface.require_same_launch_surface(build(tmp_path / 'a', 'a', other), build(tmp_path / 'b', 'b', BASE))


@pytest.mark.parametrize('addition,reason', [
    ('static volatile int fixture_flag;\n__attribute__((constructor)) static void early() { fixture_flag = 1; }\n', 'constructor, destructor or loader sections'),
    ('static volatile int fixture_flag;\n__attribute__((destructor)) static void late() { fixture_flag = 1; }\n', 'constructor, destructor or loader sections'),
    ('struct Global { Global() { std::abort(); } }; static Global global;\n', 'constructor, destructor or loader sections'),
    ('static volatile int fixture_flag;\nstatic void hook() { fixture_flag = 1; }\nstatic void (*entry)() __attribute__((used, section(".init_array"))) = hook;\n', 'constructor, destructor or loader sections'),
    ('static volatile int fixture_flag;\nstatic void hook() { fixture_flag = 1; }\nstatic void (*entry)() __attribute__((used, section(".preinit_array"))) = hook;\n', 'constructor, destructor or loader sections'),
    ('extern "C" { static void* pick() { return 0; } void chosen() __attribute__((ifunc("pick"))); }\n', 'loader-time resolver'),
    ('int visible() { return 1; }\n', 'externally visible definitions'),
    ('#include <new>\nvoid* operator new(std::size_t n) { return std::malloc(n); }\n', 'externally visible definitions'),
    ('extern "C" __attribute__((weak)) void* malloc(std::size_t) { return 0; }\n', 'runtime symbol'),
    ('#include <new>\n__attribute__((weak)) void operator delete(void* p) noexcept { std::free(p); }\n', 'runtime symbol'),
])
def test_anything_that_can_run_outside_the_call_is_rejected(tmp_path, addition, reason):
    candidate = build(tmp_path / 'c', 'c', BASE + addition)
    reference = build(tmp_path / 'r', 'r', BASE)
    with pytest.raises(surface.ElfError, match=reason):
        surface.require_same_launch_surface(candidate, reference)


def test_a_stripped_executable_is_rejected(tmp_path):
    candidate = build(tmp_path / 'c', 'c', BASE)
    reference = build(tmp_path / 'r', 'r', BASE)
    subprocess.run(['strip', str(candidate)], check=True)
    with pytest.raises(surface.ElfError, match='symbol table'):
        surface.require_same_launch_surface(candidate, reference)


def test_static_storage_counts_writable_and_thread_local_sections(tmp_path):
    def size(library):
        build(tmp_path / 's', 's', library)
        return surface.static_storage_bytes(tmp_path / 's' / 'lib.o')
    base = size(BASE)
    inside = 'int work(int n) { %s return n; }\n'
    assert size(inside % 'static char buffer[4096]; buffer[n & 4095] += 1; n += buffer[(n * 7) & 4095];') - size(inside % '') == 4096
    assert size(inside % 'static int table[1000] = {1}; table[n % 1000] += 2; n += table[(n * 7) % 1000];') - size(inside % '') == 4000
    assert size(inside % 'static thread_local char buffer[8192]; buffer[n & 8191] += 1; n += buffer[(n * 7) & 8191];') - size(inside % '') == 8192
    assert size(inside % 'static char buffer[4096] __attribute__((section("fixture_data"))); buffer[n & 4095] += 1; n += buffer[(n * 7) & 4095];') - size(inside % '') == 4096
    assert size(inside % 'static const int table[4] = {1, 2, 3, 4}; n += table[n & 3];') == size(inside % '')
    assert base >= 0
