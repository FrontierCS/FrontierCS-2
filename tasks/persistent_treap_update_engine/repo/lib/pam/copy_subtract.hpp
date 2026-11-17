#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "lib/parlay/parallel.h"
#include "build.hpp"
#include "node.hpp"

namespace pam {

static inline node_ptr make_copy_if_need(node_ptr t) {
  // use vstat to determine whether the node is persistent
  if (t->vstat == vstat_uninitialized) { return t; }
  return make_weak_copy(t);
}

node_ptr copy_concat(node_ptr t_l, node_ptr t_r) {
  if (t_l == nullptr) { return t_r; }
  if (t_r == nullptr) { return t_l; }
  node_ptr t_new;
  // if t_l has a higher priority, concat t_r with its right subtree
  if (t_l->pry > t_r->pry) {
    t_new = make_copy_if_need(t_l);
    t_new->lch = t_l->lch;
    t_new->rch = copy_concat(t_l->rch, t_r); }
  // if t_r has a higher priority, concat t_l with its left subtree
  else /* t_l->pry < t_r->pry */ {
    t_new = make_copy_if_need(t_r);
    t_new->lch = copy_concat(t_l, t_r->lch);
    t_new->rch = t_r->rch; }
  return t_new;
}

node_ptr copy_subtract(constnode_ptr t_old, node_ptr t_det) {
  if (t_old == nullptr) {
    release_tree(t_det);
    return nullptr; }
  if (t_det == nullptr) { return const_cast<node_ptr>(t_old); }

  node_ptr t_new;
  if (t_old->key == t_det->key) {
    node_ptr t_new_l, t_new_r;
    parlay::par_do(
      [t_old, t_det, &t_new_l]() { t_new_l = copy_subtract(t_old->lch, t_det->lch); },
      [t_old, t_det, &t_new_r]() { t_new_r = copy_subtract(t_old->rch, t_det->rch); });
      release_node(t_det);
      t_new = copy_concat(t_new_l, t_new_r); }
  // make a copy of t_old since it will not appear in t_det
  else if (t_old->pry > t_det->pry) {
    t_new = make_weak_copy(t_old);
    auto [t_det_l, t_det_r] = split(t_det, t_old->key);
    parlay::par_do(
      [t_old, t_det_l, t_new]() { t_new->lch = copy_subtract(t_old->lch, t_det_l); },
      [t_old, t_det_r, t_new]() { t_new->rch = copy_subtract(t_old->rch, t_det_r); }); }
  // discard t_det since it does not exist in t_old
  else /* t_old->pry < t_det->pry */ {
    // it will copy nothing by using make_copy_if_need
    node_ptr t_det_new = copy_concat(t_det->lch, t_det->rch);
    release_node(t_det);
    t_new = copy_subtract(t_old, t_det_new); }

  return t_new;
}

}
