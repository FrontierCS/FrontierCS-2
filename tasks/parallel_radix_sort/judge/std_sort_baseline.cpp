// Independently implemented sorting baseline. Reuse the unmodified source
// harness so generation, restoration, allocation accounting, checks and timing
// rounds are identical. Preload the guarded sorter headers before substituting
// only the harness's two timed call sites; no reference patch is applied.
#include <algorithm>
#include "sequence.h"
#include "radixSort.h"

template<class T, class F>
void frontierStdSort(T *array, sizeT n, F key) {
  std::sort(array, array + n,
            [&](const T &left, const T &right) { return key(left) < key(right); });
}

#define parallelIntegerSort frontierStdSort
#include "benchmark.cpp"
#undef parallelIntegerSort
