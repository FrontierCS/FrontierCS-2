#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "lib/parlay/parallel.h"
#include "augment.hpp"
#include "node.hpp"

namespace treap {

std::tuple<node_ptr, node_ptr> split(node_ptr t, uint64_t key) {
  node_ptr t_l, t_r;
  if (t == nullptr) { return std::make_tuple(nullptr, nullptr); }
  else if (key < t->key) {
    t_r = t;
    std::tie(t_l, t_r->lch) = split(t->lch, key); }
  else /* key > t->key */ {
    t_l = t;
    std::tie(t_l->rch, t_r) = split(t->rch, key); }
  return std::make_tuple(t_l, t_r);
}

node_ptr merge(node_ptr t_0, node_ptr t_1) {
  if (t_0 == nullptr) { return t_1; }
  if (t_1 == nullptr) { return t_0; }

  if (t_0->key == t_1->key) {
    t_1->val += t_0->val; // update value
    parlay::par_do(
      [t_0, t_1] { t_1->lch = merge(t_0->lch, t_1->lch); },
      [t_0, t_1] { t_1->rch = merge(t_0->rch, t_1->rch); });
    release_node(t_0);
    return t_1; }

  node_ptr t_l, t_r;
  // choose the node with higher priority as the root node
  if (t_0->pry > t_1->pry) {
    std::tie(t_l, t_r) = split(t_1, t_0->key);
    parlay::par_do(
      [t_0, t_l] { t_0->lch = merge(t_0->lch, t_l); },
      [t_0, t_r] { t_0->rch = merge(t_0->rch, t_r); });
    return t_0; }
  else /* t_0->pry < t_1->pry */ {
    std::tie(t_l, t_r) = split(t_0, t_1->key);
    parlay::par_do(
      [t_1, t_l] { t_1->lch = merge(t_l, t_1->lch); },
      [t_1, t_r] { t_1->rch = merge(t_r, t_1->rch); });
    return t_1; }
}

node_ptr build(size_t n, const uint64_t* elems) {
  if (n == 0) { return nullptr; }
  if (n == 1) { return make_node(0, elems[0]); }
  node_ptr t_0, t_1;
  parlay::par_do(
    [n, elems, &t_0] { t_0 = build(n / 2, elems); },
    [n, elems, &t_1] { t_1 = build(n - n / 2, elems + n / 2); });
  return merge(t_0, t_1);
}

static inline node_ptr build_parallel(size_t n, const uint64_t* elems) {
  node_ptr t;
  parlay::execute_with_scheduler(
    [&t, n, elems](){ treap::augment_parallel(t = treap::build(n, elems)); },
    std::thread::hardware_concurrency());
  return t;
}

} // namespace treap
