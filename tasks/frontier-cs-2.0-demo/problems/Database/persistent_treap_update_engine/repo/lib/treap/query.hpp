#pragma once

#include <cstddef>
#include <cstdint>

#include "lib/parlay/parallel.h"
#include "node.hpp"

namespace treap {

enum RANGE_COVER { OPEN_OPEN, OPEN_CLOSE, CLOSE_OPEN, CLOSE_CLOSE };

static inline constnode_ptr find(constnode_ptr t, uint64_t key) {
  while (t != nullptr) {
    if (key == t->key) { return t; }
    t = key < t->key ? t->lch : t->rch; }
  return nullptr;
}

uint64_t range_estimate(constnode_ptr t, uint64_t l, uint64_t r, RANGE_COVER d) {
  if (t == nullptr) { return 0; }

  uint64_t ret_l, ret_r;
  switch (d) {
    case RANGE_COVER::OPEN_OPEN:
      if (r < t->key) { return range_estimate(t->lch, l, r, d); }
      if (l > t->key) { return range_estimate(t->rch, l, r, d); }
      ret_l = range_estimate(t->lch, l, r, RANGE_COVER::OPEN_CLOSE);
      ret_r = range_estimate(t->rch, l, r, RANGE_COVER::CLOSE_OPEN);
      break;
    case RANGE_COVER::OPEN_CLOSE:
      if (l > t->key) { return range_estimate(t->rch, l, r, d); }
      ret_l = range_estimate(t->lch, l, r, d);
      ret_r = range_estimate(t->rch, l, r, RANGE_COVER::CLOSE_CLOSE);
      break;
    case RANGE_COVER::CLOSE_OPEN:
      if (r < t->key) { return range_estimate(t->lch, l, r, d); }
      ret_l = range_estimate(t->lch, l, r, RANGE_COVER::CLOSE_CLOSE);
      ret_r = range_estimate(t->rch, l, r, d);
      break;
    case RANGE_COVER::CLOSE_CLOSE: return t->aug;
    default: assert(false); }

  return ret_l + t->val + ret_r;
}

static inline uint64_t range_estimate(node_ptr t, uint64_t l, uint64_t r) {
  return range_estimate(t, l, r, RANGE_COVER::OPEN_OPEN);
}

}
