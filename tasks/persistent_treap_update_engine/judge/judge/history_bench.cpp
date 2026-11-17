// Judge-owned historical range-statistics benchmark (Contreap paper Section 6.2,
// Fig. 11: average historical range-query time with every version retained).
//
// The trace is the same binary workload file that the throughput benchmark
// reads.  Every transaction is submitted through the fixed scheduler contract
// and process() completes all versions before the timed phase.  The timed phase
// then answers a deterministic sequence of inclusive range-sum queries, each on
// a pseudorandom historical version in [0, m], from one thread.  The judge
// compares the printed checksum with the trusted reference implementation's
// checksum for the same trace and query seed; the elapsed query time divided by
// the query count is the measured average historical query time.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "lib/treap/build.hpp"
#include "lib/treap/query.hpp"
#include "lib/treap/scheduler.hpp"

namespace {

struct tx_context {
  uint8_t type;
  uint64_t arg0;
  uint64_t arg1;
};

struct history_query {
  size_t version;
  uint64_t left;
  uint64_t right;
};

uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: history_bench WORKLOAD THREADS BATCH QUERIES SEED\n";
    return 2;
  }
  const std::string filename = argv[1];
  const size_t num_threads = std::stoull(argv[2]);
  const size_t batch_size = std::stoull(argv[3]);
  const size_t num_queries = std::stoull(argv[4]);
  uint64_t state = std::stoull(argv[5]);
  if (num_threads == 0 || batch_size == 0 || num_queries == 0) { return 2; }

  std::ifstream fs(filename, std::ios::binary);
  size_t n = 0, m = 0;
  fs.read(reinterpret_cast<char*>(&n), sizeof(n));
  fs.read(reinterpret_cast<char*>(&m), sizeof(m));
  if (!fs || n == 0 || m == 0) { return 2; }
  std::vector<uint64_t> elems(n);
  fs.read(reinterpret_cast<char*>(elems.data()), n * sizeof(uint64_t));
  std::vector<tx_context> txs(m);
  fs.read(reinterpret_cast<char*>(txs.data()), m * sizeof(tx_context));
  if (!fs) { return 2; }

  treap::node_ptr root = treap::build_parallel(n, elems.data());
  treap::scheduler engine(num_threads, m, batch_size, root);
  for (size_t i = 0; i < m; ++i) {
    switch (txs[i].type) {
      case 0: engine.nop(); break;
      case 1: engine.insert_elem(txs[i].arg0); break;
      case 2: engine.delete_elem(txs[i].arg0); break;
      default: return 2;
    }
  }
  engine.process();
  if (engine.last_version() != m) {
    std::cerr << "unexpected last version\n";
    return 3;
  }

  // Paper Section 6.2: each historical range statistic covers 0-10% of the
  // data.  Keys lie in [1, 4n]; the inclusive range width is uniform in
  // [0, floor(4n / 10)] and the version is uniform over every retained version.
  const uint64_t key_domain = 4 * static_cast<uint64_t>(n);
  const uint64_t max_width = key_domain / 10;
  std::vector<history_query> queries(num_queries);
  for (auto& query : queries) {
    query.version = static_cast<size_t>(splitmix64(state) % (m + 1));
    const uint64_t width = splitmix64(state) % (max_width + 1);
    query.left = 1 + splitmix64(state) % (key_domain - width);
    query.right = query.left + width;
  }

  uint64_t checksum = 0xCBF29CE484222325ULL;
  const auto started = std::chrono::steady_clock::now();
  for (const auto& query : queries) {
    treap::node_ptr snapshot = engine.get_snapshot(query.version);
    const uint64_t answer = treap::range_estimate(snapshot, query.left, query.right);
    checksum = (checksum ^ answer) * 0x100000001B3ULL;
  }
  const auto finished = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(finished - started).count();

  std::printf("history\t%zu\t%.9f\t%llu\n", num_queries, seconds,
              static_cast<unsigned long long>(checksum));
  return 0;
}
