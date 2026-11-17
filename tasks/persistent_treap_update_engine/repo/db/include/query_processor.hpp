#pragma once

#include <atomic>

#include "utils/log.h"
#include "lib/moodycamel/concurrentqueue.h"

namespace DB {

class QueryProcessor {
private:
  const size_t num_threads_;
  std::thread* const threads_;

  struct query_context {
    size_t ver;
    size_t idx;
    uint64_t l;
    uint64_t r;
  };

  size_t num_queries_;
  moodycamel::ConcurrentQueue<query_context> queries_;
  const moodycamel::ProducerToken producer_token_;

  struct result_context {
    size_t ver;
    uint64_t ret;
  };

  std::vector<result_context>* const results_;

  std::atomic_bool working_;

  void process_(size_t id) {
    log_trace("start query processing");
    query_context q;
    while (working_.load(std::memory_order_acquire)) {
      while (queries_.try_dequeue_from_producer(producer_token_, q)) {
        log_trace("client %zu processing query: <%zu, %lu, %lu>", id, q.ver, q.l, q.r);
        results_[id-1].push_back(result_context{ q.idx, do_process_(q.ver, q.l, q.r) });
        log_trace("client %zu return %lu", id, results_[id-1].back().ret); } }
    log_trace("stop query processing");
  }

protected:
  virtual uint64_t do_process_(size_t ver, uint64_t l, uint64_t r) = 0;

public:
  QueryProcessor(size_t num_threads) :
    num_threads_(num_threads),
    threads_(new std::thread[num_threads]),
    num_queries_(0),
    queries_(),
    producer_token_(queries_),
    results_(new std::vector<result_context>[num_threads]),
    working_(false) { }

  void Start() {
    working_.store(true, std::memory_order_seq_cst);
    for (size_t thread_id = 1; thread_id <= num_threads_; ++thread_id) {
      new (&threads_[thread_id-1]) std::thread(&QueryProcessor::process_, this, thread_id); }
  }

  void Stop() {
    working_.store(false, std::memory_order_seq_cst);
    for (size_t thread_id = 1; thread_id <= num_threads_; ++thread_id) { threads_[thread_id-1].join(); }
  }

  int Push(size_t ver, uint64_t l, uint64_t r) {
    num_queries_++;
    if (queries_.enqueue(producer_token_, query_context{ ver, num_queries_, l, r })) { return 0; }
    return ~0;
  }
};

template <typename F>
class alignas(128) QueryProcessorImpl : public QueryProcessor {
private:
  F f_;
protected:
  virtual uint64_t do_process_(size_t ver, uint64_t l, uint64_t r) override {
    return f_(ver, l, r);
  }
public:
  QueryProcessorImpl(size_t num_threads, F&& f) : QueryProcessor(num_threads), f_(f) { }
};

}
