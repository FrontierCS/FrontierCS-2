#pragma once

#include <assert.h>
#include "utils/log.h"

#include <cstddef>
#include <cstdint>

#include "include/interface.hpp"
#include "include/query_processor.hpp"

#include "lib/aes_hash.hpp"
#include "lib/treap/augment.hpp"
#include "lib/treap/build.hpp"
#include "lib/treap/node.hpp"
#include "lib/treap/query.hpp"
#include "lib/treap/search_delete.hpp"
#include "lib/treap/search_insert.hpp"

namespace DB {

class alignas(128) Sequential : public Interface {
private:
  const size_t num_threads_;
  std::atomic_size_t num_versions_;
  treap::node_ptr* roots_;

  alignas(128) QueryProcessor* query_processor_;

  alignas(128) uint8_t pad_[]; // padding to avoid false sharing

  auto processor_function_() {
    return [this](size_t ver, uint64_t l, uint64_t r)->uint64_t {
      while (num_versions_.load(std::memory_order_acquire) < ver) { }
      if (l == r) { return treap::find(roots_[ver], l)->val; }
      return treap::range_estimate(roots_[ver], l, r); };
  }

public:
  explicit Sequential(size_t num_threads, size_t num_clients) :
    num_threads_(num_threads),
    num_versions_(0),
    roots_(nullptr),
    query_processor_(new QueryProcessorImpl(num_clients, processor_function_()))
  { }

  void Init(size_t n, size_t m, uint64_t* elems) override {
    roots_ = new treap::node_ptr[m+1];
    roots_[0] = treap::build_parallel(n, elems);
    query_processor_->Start();
  }

  void Close() override {
    query_processor_->Stop();
    log_debug("final result: %lu", roots_[num_versions_]->aug);
  }

  int Query(uint64_t l, uint64_t r) override {
    size_t ver = 1+num_versions_.fetch_add(1, std::memory_order_release);
    roots_[ver] = roots_[ver-1];
    return query_processor_->Push(ver, l, r);
  }

  int Insert(uint64_t k) override {
    size_t ver = 1+num_versions_.fetch_add(1, std::memory_order_release);
    roots_[ver] = treap::search_insert(ver, roots_[ver-1], aes_hash::hash(k), k);
    treap::augment(roots_[ver]);
    return 0;
  }

  int Delete(uint64_t k) override {
    size_t ver = 1+num_versions_.fetch_add(1, std::memory_order_release);
    roots_[ver] = treap::search_delete(ver, roots_[ver-1], aes_hash::hash(k), k);
    treap::augment(roots_[ver]);
    return 0;
  }
};

}
