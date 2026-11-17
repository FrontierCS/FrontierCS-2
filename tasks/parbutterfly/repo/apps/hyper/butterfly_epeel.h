#ifndef FRONTIER_BUTTERFLY_EPEEL_H
#define FRONTIER_BUTTERFLY_EPEEL_H
#include "butterfly_utils.h"
#include "index_map.h"
#include <stdexcept>

array_imap<long> PeelE(uintE* eti, uintE* ite, bipartiteCSR& GA, bool use_v,
                      long* butterflies, long max_wedges, PeelType type,
                      size_t num_buckets=128) {
  throw std::runtime_error("peeling implementation is not supplied");
}
#endif
