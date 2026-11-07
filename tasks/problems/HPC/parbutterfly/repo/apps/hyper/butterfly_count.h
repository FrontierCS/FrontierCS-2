#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include "parallel.h"
#include "gettime.h"
#include "utils.h"
#include "graph.h"
#include "vertex.h"
#include "sequence.h"
#include "sparseSet.h"
#include "sampleSort.h"
#ifndef OPENMP
#include "../../lib/histogram.h"
#include "../../lib/sample_sort.h"
#endif
#include "../../radixsort/RadixSort/radixSort.h"

#include "butterfly_utils.h"

using namespace std;

#include <stdexcept>

long* Count(bipartiteCSR& GA, bool use_v, long num_wedges, long max_wedges, long max_array_size, CountType type, RankType tw) {
  throw std::runtime_error("counting implementation is not supplied");
}
