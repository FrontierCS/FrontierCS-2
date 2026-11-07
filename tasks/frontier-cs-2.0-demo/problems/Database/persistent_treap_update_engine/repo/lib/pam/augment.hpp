#pragma once

#include <assert.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "lib/parlay/parallel.h"
#include "node.hpp"

namespace pam {

uint64_t augment_parallel(node_ptr t) {
  if (t == nullptr) { return 0; }
  if (t->vstat != vstat_uninitialized) { return t->aug; }
  uint64_t aug_l, aug_r;
  parlay::par_do(
    [&aug_l, t]() { aug_l = augment_parallel(t->lch); },
    [&aug_r, t]() { aug_r = augment_parallel(t->rch); });
  t->aug = aug_l + t->vals[t->nvals-1].val + aug_r;
  t->vstat = std::max(t->vals[t->nvals-1].ver, std::max(subtree_vstat(t->lch), subtree_vstat(t->rch)));
  return t->aug;
}

}
