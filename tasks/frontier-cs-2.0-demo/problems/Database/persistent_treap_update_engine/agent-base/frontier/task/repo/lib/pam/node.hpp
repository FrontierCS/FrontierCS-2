#pragma once

#include <assert.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mimalloc.h>
#include <mimalloc-override.h>
#include <mimalloc-new-delete.h>

#include "lib/aes_hash.hpp"

namespace pam {

static_assert(sizeof(size_t) == sizeof(std::atomic_size_t));
static_assert(alignof(size_t) == alignof(std::atomic_size_t));

struct valver { uint64_t val; size_t ver; };

struct alignas(64) node_t {
  size_t vstat; // version of augment
  uint64_t pry;
  uint64_t key;
  size_t nvals;
  valver* vals;
  uint64_t aug; // in this demo, it's the sum of counters, 0 as uninitialized
  node_t* lch;
  node_t* rch;
};

static_assert(sizeof(node_t) == 64);

using node_ptr = node_t*;
using constnode_ptr = const node_t*;

constexpr uint64_t vstat_uninitialized = ~uint64_t{0};
constexpr uint64_t aug_uninitialized = 0;

static inline node_ptr make_node(size_t ver, uint64_t pry, uint64_t key, uint64_t val = 1) {
  node_ptr t = new node_t {
    vstat_uninitialized,
    pry, key,
    1, new valver[1],
    aug_uninitialized,
    nullptr, nullptr };
  t->vals[0].val = val;
  t->vals[0].ver = ver;
  return t;
}

static inline node_ptr make_node(size_t ver, uint64_t key) {
  return make_node(ver, aes_hash::hash(key), key);
}

// only copy the last value
static inline node_ptr make_weak_copy(constnode_ptr t) {
  assert(t != nullptr);
  return make_node(t->vals[t->nvals-1].ver, t->pry, t->key, t->vals[t->nvals-1].val);
}

size_t subtree_vstat(node_ptr t) {
  if (t == nullptr) { return 0; }
  assert(t->vstat != vstat_uninitialized);
  return t->vstat;
}

static inline void release_node(node_ptr t) {
  assert(t != nullptr);
  if (t->vals != nullptr) { delete t->vals; }
  delete t;
}

void release_tree(node_ptr t) {
  if (t == nullptr) { return; }
  release_tree(t->lch);
  release_tree(t->rch);
  release_node(t);
}

} // namespace treap
