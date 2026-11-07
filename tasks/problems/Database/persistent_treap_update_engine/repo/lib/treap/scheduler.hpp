#pragma once

#include <cstddef>
#include <cstdint>

#include "augment.hpp"
#include "node.hpp"
#include "search_delete.hpp"
#include "search_insert.hpp"

namespace treap {

// Correctness-preserving serial scaffold for the ordered persistent-update
// contract used by DB::Contreap. Replace this implementation with an update
// engine that preserves the same observable versions.
struct scheduler {
private:
  const size_t num_tasks_;
  size_t num_issued_;
  node_ptr* roots_;

public:
  scheduler(size_t, size_t num_tasks, size_t, node_ptr t0) :
    num_tasks_(num_tasks), num_issued_(0), roots_(new node_ptr[num_tasks+1]) {
    roots_[0] = t0;
  }

  inline void insert_elem(uint64_t elem) {
    const size_t ver = ++num_issued_;
    roots_[ver] = search_insert(ver, roots_[ver-1], elem);
    augment(roots_[ver]);
  }

  inline void delete_elem(uint64_t elem) {
    const size_t ver = ++num_issued_;
    roots_[ver] = search_delete(ver, roots_[ver-1], elem);
    augment(roots_[ver]);
  }

  inline size_t nop() {
    const size_t ver = ++num_issued_;
    roots_[ver] = roots_[ver-1];
    return ver;
  }

  void process() { }

  inline node_ptr get_snapshot(size_t ver) {
    return ver <= num_issued_ ? roots_[ver] : nullptr;
  }

  inline size_t last_version() { return num_issued_; }
};

}
