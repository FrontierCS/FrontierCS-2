#pragma once

#include <assert.h>
#include "utils/log.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "db/include/interface.hpp"
#include "db/include/query_processor.hpp"

#include "lib/parlay/parallel.h"

namespace DB {

template <typename T>
class alignas(128) Batch : public Interface {
private:
  parlay::scheduler<parlay::WorkStealingJob>* schd_;
  const size_t num_threads_;

  // following data are ownned by the modifier thread

  enum operation : uint8_t { INSERT, DELETE };

  const size_t batch_size_;
  operation buffer_type_;
  size_t buffer_count_;
  uint64_t* buffer_;

  size_t version_submitted_;
  size_t version_committed_;

  // following data are shared with query processor

  alignas(128) QueryProcessor* query_processor_;

  struct root_list {
    size_t ver;
    typename T::node_ptr t;
    root_list* prev;
    root_list* next;
  };

  root_list* root_;
  root_list* root0_;
  std::atomic_size_t num_versions_;

  alignas(128) uint8_t pad_[]; // padding to avoid false sharing

  void do_update_() {
    typename T::node_ptr t_new;
    switch (buffer_type_) {
      case operation::INSERT:
        t_new = T::inserted(root_->t, version_committed_+1, buffer_count_, buffer_);
        break;
      case operation::DELETE:
        t_new = T::deleted(root_->t, version_committed_+1, buffer_count_, buffer_);
        break;
      default:
        log_fatal("unknown type of operation");
        abort(); }
    version_committed_ += buffer_count_;
    buffer_count_ = 0;
    log_debug("commit version: %zu", version_committed_);
    log_debug("current size: %lu", t_new->aug);
    root_->next = new root_list { version_committed_, t_new, root_, nullptr };
    root_ = root_->next;
    num_versions_.fetch_add(1, std::memory_order_release);
  }

  void update_enbuffer_(operation type, uint64_t key) {
    // clear buffer if another type of operations exist
    if (buffer_type_ != type) {
      do_update_();
      buffer_type_ = type; }

    buffer_[buffer_count_++] = key;
    version_submitted_++;
    if (buffer_count_ == batch_size_) { do_update_(); }
  }

  uint64_t process_range_query_(size_t ver, uint64_t l, uint64_t r) {
    static thread_local size_t cur_vid = 0;
    static thread_local size_t last_vid = 0;
    static thread_local root_list* cur_root = root0_;

    while (cur_root != root0_ && ver <= cur_root->prev->ver) {
      cur_vid--;
      cur_root = cur_root->prev; }

    while (ver > cur_root->ver) {
      while (ver > cur_root->ver && cur_vid < last_vid) {
        cur_vid++;
        cur_root = cur_root->next; }
      if (ver > cur_root->ver) { last_vid = num_versions_.load(std::memory_order_acquire); } }

    if (l == r) { return T::find(cur_root->t, ver, l); }
    return T::range_estimate(cur_root->t, ver, l, r);
  }

  auto processor_function_() {
    return [this](size_t ver, uint64_t l, uint64_t r)->uint64_t { return process_range_query_(ver, l, r); };
  }

public:
  explicit Batch(size_t num_threads, size_t num_clients, size_t batch_size) :
    schd_(nullptr),
    num_threads_(num_threads),
    batch_size_(batch_size),
    buffer_type_(operation::INSERT),
    buffer_count_(0),
    buffer_(new uint64_t[batch_size_]),
    version_submitted_(0),
    version_committed_(0),
    query_processor_(new QueryProcessorImpl(num_clients, processor_function_())),
    root_(nullptr),
    root0_(nullptr),
    num_versions_(0)
  { }

  void Init(size_t n, size_t m, uint64_t* elems) override {
    root_ = root0_ = new root_list{ 0, T::build(n, elems), nullptr, nullptr };
    log_debug("size of root0 %lu", root0_->t->aug);
    schd_ = new parlay::scheduler<parlay::WorkStealingJob>(num_threads_);
    query_processor_->Start();
  }

  void Close() override {
    if (buffer_count_ > 0) { do_update_(); }
    delete schd_;
    query_processor_->Stop();
  }

  int Query(uint64_t l, uint64_t r) override {
    return query_processor_->Push(version_submitted_, l, r);
  }

  int Insert(uint64_t k) override {
    update_enbuffer_(operation::INSERT, k);
    return 0;
  }

  int Delete(uint64_t k) override {
    update_enbuffer_(operation::DELETE, k);
    return 0;
  }
};

}
