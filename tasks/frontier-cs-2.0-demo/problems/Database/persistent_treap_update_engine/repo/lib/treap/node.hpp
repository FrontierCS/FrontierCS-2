#pragma once

#include <assert.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <mimalloc.h>
#include <mimalloc-override.h>
#include <mimalloc-new-delete.h>

#include "lib/aes_hash.hpp"

namespace treap {

static_assert(sizeof(size_t) == sizeof(std::atomic_size_t));
static_assert(alignof(size_t) == alignof(std::atomic_size_t));

struct alignas(64) node_t {
  size_t ver;
  uint64_t pry;
  uint64_t key;
  uint64_t val; // in this demo, it's the counter of the key
  uint64_t aug; // in this demo, it's the sum of counters, 0 as uninitialized
  node_t* lch;
  node_t* rch;
};

static_assert(sizeof(node_t) == 64);

using node_ptr = node_t*;
using constnode_ptr = const node_t*;

constexpr uint64_t aug_uninitialized = 0;

static inline node_ptr make_node(size_t ver, uint64_t pry, uint64_t key) {
  return new node_t{ver, pry, key, 1, aug_uninitialized, nullptr, nullptr};
}

static inline node_ptr make_node(size_t ver, uint64_t key) {
  return make_node(ver, aes_hash::hash(key), key);
}

static inline node_ptr make_weak_copy(constnode_ptr t) {
  assert(t != nullptr);
  return new node_t{t->ver, t->pry, t->key, t->val, aug_uninitialized, nullptr, nullptr};
}

static inline node_ptr make_weak_copy(constnode_ptr t, size_t ver) {
  assert(t != nullptr);
  return new node_t{ver, t->pry, t->key, t->val, aug_uninitialized, nullptr, nullptr};
}

static inline void release_node(node_ptr t) {
  assert(t != nullptr);
  delete t;
}

void release_tree(node_ptr t) {
  if (t == nullptr) { return; }
  release_tree(t->lch);
  release_tree(t->rch);
  release_node(t);
}

} // namespace treap
