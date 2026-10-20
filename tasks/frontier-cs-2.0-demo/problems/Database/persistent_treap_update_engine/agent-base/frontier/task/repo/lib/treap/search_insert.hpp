#pragma once

#include <cstddef>
#include <cstdint>

#include "node.hpp"

namespace treap {

// Conventional serial path-copying treap insertion retained as the functional
// scaffold. It publishes a complete root before the next request begins.
node_ptr search_insert(size_t ver, node_ptr t, uint64_t pry, uint64_t key) {
  if (t == nullptr) { return make_node(ver, pry, key); }
  node_ptr out = make_weak_copy(t, ver);
  if (key == t->key) {
    out->val++;
    out->lch = t->lch;
    out->rch = t->rch;
  } else if (key < t->key) {
    out->lch = search_insert(ver, t->lch, pry, key);
    out->rch = t->rch;
    if (out->lch->pry > out->pry) {
      node_ptr top = out->lch;
      out->lch = top->rch;
      top->rch = out;
      return top;
    }
  } else {
    out->lch = t->lch;
    out->rch = search_insert(ver, t->rch, pry, key);
    if (out->rch->pry > out->pry) {
      node_ptr top = out->rch;
      out->rch = top->lch;
      top->lch = out;
      return top;
    }
  }
  return out;
}

inline node_ptr search_insert(size_t ver, node_ptr t, uint64_t key) {
  return search_insert(ver, t, aes_hash::hash(key), key);
}

}
