#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "lib/parlay/parallel.h"
#include "node.hpp"
#include "build.hpp"

namespace pam {

void update_values(node_ptr t, uint64_t val_base) {
  for (size_t i = 0; i < t->nvals; i++) { t->vals[i].val += val_base; }
}

static inline std::tuple<node_ptr, node_ptr, node_ptr> expose(node_ptr t) {
  node_ptr t_l = t->lch;
  node_ptr t_r = t->rch;
  t->lch = t->rch = nullptr;
  return std::make_tuple(t_l, t, t_r);
}

std::tuple<node_ptr, node_ptr> copy_split(constnode_ptr t, uint64_t key) {
  node_ptr t_l, t_r;
  if (t == nullptr) { return std::make_tuple(nullptr, nullptr); }
  else if (key < t->key) {
    t_r = make_weak_copy(t);
    t_r->rch = t->rch;
    std::tie(t_l, t_r->lch) = copy_split(t->lch, key); }
  else /* key > t->key */ {
    t_l = make_weak_copy(t);
    t_l->lch = t->lch;
    std::tie(t_l->rch, t_r) = copy_split(t->rch, key); }
  return std::make_tuple(t_l, t_r);
}

node_ptr copy_merge(constnode_ptr t_old, node_ptr t_ins) {
  if (t_old == nullptr) { return t_ins; }
  if (t_ins == nullptr) { return const_cast<node_ptr>(t_old); }

  constnode_ptr t_old_l, t_old_r;
  node_ptr t_new, t_ins_l, t_ins_r;

  // place a copy of t_old if it has a higher priority
  if (t_old->pry > t_ins->pry) {
    t_new = make_weak_copy(t_old);
    t_old_l = t_old->lch;
    t_old_r = t_old->rch;
    // split the temporary tree t_ins
    std::tie(t_ins_l, t_ins_r) = split(t_ins, t_old->key); }
  // otherwise, make t_ins persistent and place it
  else /* t_old->pry <= t_ins->pry */ {
    std::tie(t_ins_l, t_new, t_ins_r) = expose(t_ins);
    // if the key exist, update value and merge the subtrees
    if (t_old->key == t_ins->key) {
      t_old_l = t_old->lch;
      t_old_r = t_old->rch;
      update_values(t_new, t_old->vals[t_old->nvals-1].val); }
    // otherwise, split the persistent tree t_old with copying
    else /* t_old->key != t_ins->key */ { std::tie(t_old_l, t_old_r) = copy_split(t_old, t_ins->key); } }

  parlay::par_do(
    [t_new, t_old_l, t_ins_l]() { t_new->lch = copy_merge(t_old_l, t_ins_l); },
    [t_new, t_old_r, t_ins_r]() { t_new->rch = copy_merge(t_old_r, t_ins_r); });
  return t_new;
}

}
