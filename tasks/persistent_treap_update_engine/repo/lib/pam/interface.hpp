#pragma once

#include <cstddef>
#include <cstdint>

#include "augment.hpp"
#include "build.hpp"
#include "build_versioned.hpp"
#include "copy_merge.hpp"
#include "copy_subtract.hpp"
#include "node.hpp"
#include "query.hpp"

namespace pam { struct interface {

using node_ptr = pam::node_ptr;

static inline node_ptr build(size_t n, const uint64_t* elems) {
  node_ptr t;
  parlay::execute_with_scheduler(
    [&t, n, elems](){ pam::augment_parallel(t = pam::build(n, elems)); },
    std::thread::hardware_concurrency());
  return t;
}

static inline node_ptr inserted(constnode_ptr t_old, size_t ver, uint64_t elem) {
  node_ptr t = copy_merge(t_old, make_node(ver, elem));
  augment_parallel(t);
  return t;
}

static inline node_ptr inserted(constnode_ptr t_old, size_t ver, size_t n, const uint64_t* elems) {
  node_ptr t_det = build_versioned(ver, n, elems);
  node_ptr t = copy_merge(t_old, t_det);
  augment_parallel(t);
  return t;
}

static inline node_ptr deleted(constnode_ptr t_old, size_t ver, uint64_t elem) {
  node_ptr t = copy_subtract(t_old, make_node(ver, elem));
  augment_parallel(t);
  return t;
}

static inline node_ptr deleted(constnode_ptr t_old, size_t ver, size_t n, const uint64_t* elems) {
  node_ptr t_det = build_versioned(ver, n, elems);
  node_ptr t = copy_subtract(t_old, t_det);
  augment_parallel(t);
  return t;
}

static inline uint64_t find(constnode_ptr t, size_t v, uint64_t k) {
  return pam::find(t, v, k);
}

static inline uint64_t range_estimate(node_ptr t, size_t v, uint64_t l, uint64_t r) {
  return pam::range_estimate(t, v, l, r);
}

}; }
