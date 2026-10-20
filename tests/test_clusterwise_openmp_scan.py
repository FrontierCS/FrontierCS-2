"""Large production prefix scans cover the actual capped OpenMP team."""
import deployment_test_guard  # noqa: F401 # isort: skip

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

TASK = Path(__file__).resolve().parents[1] / (
    "tasks/frontier-cs-2.0-demo/problems/Arch/clusterwise_spgemm")


class ClusterwiseOpenMPScanTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'), 'native regression requires g++')
    def test_all_scan_overloads_cover_capped_team_without_reading_past_input(self):
        source = (TASK / 'judge/harbor/app/utility.h').read_text()
        self.assertEqual(source, (TASK / 'agent-base/frontier/task/repo/utility.h').read_text())
        functions = source[source.index('// Prefix sum (Sequential)'):source.index('// Sort by key')]
        # Test the exact functions; the allocator shim removes the external TBB
        # dependency and is never used for performance measurements.
        fixture = r'''
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <omp.h>
#define scalable_malloc std::malloc
#define scalable_free std::free
''' + functions + r'''
int main() {
    const int64_t n = 140001;
    auto input = new int64_t[n - 1];
    auto weights = new int64_t[n - 1];
    auto output = new int64_t[n];
    auto weighted = new int64_t[n];
    for (int64_t i = 0; i < n - 1; ++i) { input[i] = 1; weights[i] = 2; }
    auto check = [&](int factor) {
        for (int64_t i = 0; i < n; ++i) {
            if (output[i] != factor * i) { std::cerr << "bad prefix at " << i; return false; }
        }
        return true;
    };
    scan(input, output, n);
    if (!check(1)) return 1;
    scan(int64_t(1), output, n);
    if (!check(1)) return 2;
    scan(static_cast<const int64_t*>(input), output, weighted,
         static_cast<const int64_t*>(weights), n);
    if (!check(1)) return 3;
    for (int64_t i = 0; i < n; ++i) if (weighted[i] != 2 * i) return 4;
    scan(static_cast<const int64_t*>(input), output,
         static_cast<const int64_t*>(weights), n);
    if (!check(2)) return 5;
    scan(static_cast<const int64_t*>(input), output, int64_t(2), n);
    if (!check(2)) return 6;
    delete[] input; delete[] weights; delete[] output; delete[] weighted;
    std::cout << "five exact prefixes passed\n";
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scan.cpp').write_text(fixture)
            compile_result = subprocess.run(['g++', '-O1', '-g', '-fopenmp', '-std=c++17',
                '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                str(root / 'scan.cpp'), '-o', str(root / 'scan')], capture_output=True, text=True)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            for limit in (60, 7):
                with self.subTest(thread_limit=limit):
                    result = subprocess.run([str(root / 'scan')], capture_output=True, text=True,
                        timeout=45, env={**os.environ, 'OMP_NUM_THREADS': '64',
                        'OMP_THREAD_LIMIT': str(limit), 'OMP_DYNAMIC': 'FALSE',
                        'ASAN_OPTIONS': 'detect_leaks=0', 'UBSAN_OPTIONS': 'halt_on_error=1'})
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, 'five exact prefixes passed\n')


if __name__ == '__main__':
    unittest.main()
