#ifndef _BUTTERFLY_COUNT_TOTAL_
#define _BUTTERFLY_COUNT_TOTAL_

#include "butterfly_utils.h"

// Public integration contract for exact TOTAL counting.  The implementation
// has intentionally been removed for this task.
long CountTotal(bipartiteCSR& GA, bool use_v, long num_wedges,
                long max_wedges, long max_array_size, CountType type,
                RankType tw) {
  (void)GA;
  (void)use_v;
  (void)num_wedges;
  (void)max_wedges;
  (void)max_array_size;
  (void)type;
  (void)tw;
  return -1;
}

#endif
