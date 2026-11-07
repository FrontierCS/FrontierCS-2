#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "lib/parlay/parallel.h"
#include "build.hpp"
#include "node.hpp"

namespace pam {

/* values in t_0 are always earlier than that in t_1 */
void merge_values(node_ptr t_0, node_ptr t_1) {
  valver* new_vals = new valver[t_0->nvals + t_1->nvals];
  for (size_t i = 0; i < t_0->nvals; i++) { new_vals[i] = t_0->vals[i]; }
  uint64_t val_base = t_0->vals[t_0->nvals-1].val;
  for (size_t i = 0; i < t_1->nvals; i++) {
    new_vals[t_0->nvals+i].val = val_base + t_1->vals[i].val;
    new_vals[t_0->nvals+i].ver = t_1->vals[i].ver; }
  t_0->nvals += t_1->nvals;
  delete t_0->vals;
  t_0->vals = new_vals;
}

node_ptr merge_versioned(node_ptr t_0, node_ptr t_1) {
  if (t_0 == nullptr) { return t_1; }
  if (t_1 == nullptr) { return t_0; }

  if (t_0->key == t_1->key) {
    merge_values(t_0, t_1);
    parlay::par_do(
      [t_0, t_1] { t_0->lch = merge_versioned(t_0->lch, t_1->lch); },
      [t_0, t_1] { t_0->rch = merge_versioned(t_0->rch, t_1->rch); });
    release_node(t_1);
    return t_0; }

  node_ptr t_l, t_r;
  // choose the node with higher priority as the root node
  if (t_0->pry > t_1->pry) {
    std::tie(t_l, t_r) = split(t_1, t_0->key);
    parlay::par_do(
      [t_0, t_l] { t_0->lch = merge_versioned(t_0->lch, t_l); },
      [t_0, t_r] { t_0->rch = merge_versioned(t_0->rch, t_r); });
    return t_0; }
  else /* t_0->pry < t_1->pry */ {
    std::tie(t_l, t_r) = split(t_0, t_1->key);
    parlay::par_do(
      [t_1, t_l] { t_1->lch = merge_versioned(t_l, t_1->lch); },
      [t_1, t_r] { t_1->rch = merge_versioned(t_r, t_1->rch); });
    return t_1; }
}

node_ptr build_versioned(size_t ver, size_t n, const uint64_t* elems) {
  if (n == 0) { return nullptr; }
  if (n == 1) { return make_node(ver, elems[0]); }
  node_ptr t_0, t_1;
  parlay::par_do(
    [ver, n, elems, &t_0] { t_0 = build_versioned(ver, n / 2, elems); },
    [ver, n, elems, &t_1] { t_1 = build_versioned(ver + n / 2, n - n / 2, elems + n / 2); });
  return merge_versioned(t_0, t_1);
}

} // namespace treap
