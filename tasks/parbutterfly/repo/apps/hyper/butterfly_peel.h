#ifndef FRONTIER_BUTTERFLY_PEEL_H
#define FRONTIER_BUTTERFLY_PEEL_H
#include "butterfly_utils.h"
#include "index_map.h"
#include <stdexcept>

array_imap<long> Peel(bipartiteCSR& GA, bool use_v, long* butterflies,
                     long max_wedges, PeelType type, long max_array_size,
                     size_t num_buckets=128) {
  throw std::runtime_error("peeling implementation is not supplied");
}
#endif
