#pragma once

#include <cstddef>
#include <cstdint>

#include "node.hpp"

namespace treap {

static node_ptr join_serial(size_t ver, node_ptr left, node_ptr right) {
  if (left == nullptr) { return right; }
  if (right == nullptr) { return left; }
  if (left->pry > right->pry) {
    node_ptr out = make_weak_copy(left, ver);
    out->lch = left->lch;
    out->rch = join_serial(ver, left->rch, right);
    return out;
  }
  node_ptr out = make_weak_copy(right, ver);
  out->lch = join_serial(ver, left, right->lch);
  out->rch = right->rch;
  return out;
}

// Conventional serial path-copying deletion retained as the scaffold.
node_ptr search_delete(size_t ver, node_ptr t, uint64_t, uint64_t key) {
  if (t == nullptr) { return nullptr; }
  if (key < t->key) {
    node_ptr out = make_weak_copy(t, ver);
    out->lch = search_delete(ver, t->lch, 0, key);
    out->rch = t->rch;
    return out;
  }
  if (key > t->key) {
    node_ptr out = make_weak_copy(t, ver);
    out->lch = t->lch;
    out->rch = search_delete(ver, t->rch, 0, key);
    return out;
  }
  if (t->val > 1) {
    node_ptr out = make_weak_copy(t, ver);
    out->val--;
    out->lch = t->lch;
    out->rch = t->rch;
    return out;
  }
  return join_serial(ver, t->lch, t->rch);
}

inline node_ptr search_delete(size_t ver, node_ptr t, uint64_t key) {
  return search_delete(ver, t, aes_hash::hash(key), key);
}

}
