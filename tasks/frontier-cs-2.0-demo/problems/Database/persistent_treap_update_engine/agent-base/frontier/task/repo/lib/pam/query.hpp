#pragma once

#include <cstddef>
#include <cstdint>

#include "lib/parlay/parallel.h"
#include "node.hpp"

namespace pam {

static inline uint64_t find_value(constnode_ptr t, size_t v) {
  if (v < t->vals[0].ver) { return 0; }
  for (size_t i = 0; i < t->nvals-1; i++) {
    if (v < t->vals[i+1].ver) { return t->vals[i].val; } }
  return t->vals[t->nvals-1].val;
}

static inline uint64_t find(constnode_ptr t, size_t v, uint64_t k) {
  while (t != nullptr) {
    if (k == t->key) { return find_value(t, v); }
    t = k < t->key ? t->lch : t->rch; }
  return 0;
}

enum RANGE_COVER { OPEN_OPEN, OPEN_CLOSE, CLOSE_OPEN, CLOSE_CLOSE };

uint64_t range_estimate(node_ptr t, size_t v, uint64_t l, uint64_t r, RANGE_COVER d) {
  if (!t) return 0;

  uint64_t ret_l, ret_r;
  switch (d) {
    case RANGE_COVER::OPEN_OPEN:
      if (r < t->key) { return range_estimate(t->lch, v, l, r, d); }
      if (l > t->key) { return range_estimate(t->rch, v, l, r, d); }
      ret_l = range_estimate(t->lch, v, l, r, RANGE_COVER::OPEN_CLOSE);
      ret_r = range_estimate(t->rch, v, l, r, RANGE_COVER::CLOSE_OPEN);
      break;
    case RANGE_COVER::OPEN_CLOSE:
      if (l > t->key) { return range_estimate(t->rch, v, l, r, d); }
      ret_l = range_estimate(t->lch, v, l, r, d);
      ret_r = range_estimate(t->rch, v, l, r, RANGE_COVER::CLOSE_CLOSE);
      break;
    case RANGE_COVER::CLOSE_OPEN:
      if (r < t->key) { return range_estimate(t->lch, v, l, r, d); }
      ret_l = range_estimate(t->lch, v, l, r, RANGE_COVER::CLOSE_CLOSE);
      ret_r = range_estimate(t->rch, v, l, r, d);
      break;
    case RANGE_COVER::CLOSE_CLOSE:
      if (t->vstat <= v) { return t->aug; }
      ret_l = range_estimate(t->lch, v, l, r, d);
      ret_r = range_estimate(t->rch, v, l, r, d);
      break;
    default: assert(false); }

  return ret_l + find_value(t, v) + ret_r;
}

static inline uint64_t range_estimate(node_ptr t, size_t v, uint64_t l, uint64_t r) {
  return range_estimate(t, v, l, r, RANGE_COVER::OPEN_OPEN);
}

}
