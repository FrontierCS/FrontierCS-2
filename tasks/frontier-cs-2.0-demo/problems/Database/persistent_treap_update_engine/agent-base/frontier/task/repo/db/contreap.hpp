#pragma once

#include <assert.h>
#include "utils/log.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "db/include/interface.hpp"
#include "db/include/query_processor.hpp"

#include "lib/treap/augment.hpp"
#include "lib/treap/build.hpp"
#include "lib/treap/node.hpp"
#include "lib/treap/query.hpp"
#include "lib/treap/scheduler.hpp"

namespace DB {

class alignas(128) Contreap : public Interface {
private:
  // following data are ownned by the modifier thread

  const size_t num_threads_;
  const size_t block_size_;

  alignas(128) treap::scheduler* contreap_;

  alignas(128) QueryProcessor* query_processor_;

  alignas(128) uint8_t pad_[]; // padding to avoid false sharing

  auto processor_function_() {
    return [this](size_t ver, uint64_t l, uint64_t r)->uint64_t {
      treap::node_ptr t = contreap_->get_snapshot(ver);
      if (l == r) { return treap::find(t, l)->val; }
      return treap::range_estimate(t, l, r); };
  }

public:
  explicit Contreap(size_t num_threads, size_t num_clients, size_t block_size) :
    num_threads_(num_threads),
    block_size_(block_size),
    contreap_(nullptr),
    query_processor_(new QueryProcessorImpl(num_clients, processor_function_()))
  { }

  void Init(size_t n, size_t m, uint64_t* elems) override {
    treap::node_ptr t = treap::build_parallel(n, elems);
    contreap_ = new treap::scheduler(num_threads_, m, block_size_, t);
    query_processor_->Start();
  }

  void Close() override {
    contreap_->process();
    query_processor_->Stop();
    log_debug("final result: %lu", contreap_->get_snapshot(contreap_->last_version())->aug);
  }

  int Query(uint64_t l, uint64_t r) override {
    size_t ver = contreap_->nop();
    return query_processor_->Push(ver, l, r);
  }

  int Insert(uint64_t k) override {
    contreap_->insert_elem(k);
    return 0;
  }

  int Delete(uint64_t k) override {
    contreap_->delete_elem(k);
    return 0;
  }
};

}
