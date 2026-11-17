#pragma once

#include <assert.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "lib/parlay/parallel.h"
#include "lib/treap/node.hpp"

namespace treap {

uint64_t augment(node_ptr t) {
  if (t == nullptr) { return 0; }
  if (t->aug != aug_uninitialized) { return t->aug; }
  t->aug = augment(t->lch) + t->val + augment(t->rch);
  return t->aug;
}

uint64_t augment_parallel(node_ptr t) {
  if (t == nullptr) { return 0; }
  if (t->aug != aug_uninitialized) { return t->aug; }
  uint64_t aug_l, aug_r;
  parlay::par_do(
    [&aug_l, t]() { aug_l = augment_parallel(t->lch); },
    [&aug_r, t]() { aug_r = augment_parallel(t->rch); });
  t->aug = aug_l + t->val + aug_r;
  return t->aug;
}

}
