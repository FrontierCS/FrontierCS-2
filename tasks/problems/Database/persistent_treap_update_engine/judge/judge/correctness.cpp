#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "lib/treap/build.hpp"
#include "lib/treap/query.hpp"
#include "lib/treap/scheduler.hpp"

struct Op { int type; uint64_t key; };

static bool check_tree(treap::node_ptr t, uint64_t lo, uint64_t hi,
                       size_t version, const std::map<uint64_t, uint64_t>& want,
                       uint64_t& sum, std::string& why) {
  if (t == nullptr) { sum = 0; return true; }
  if (t->key <= lo || t->key >= hi) { why = "BST ordering"; return false; }
  if (t->ver > version) { why = "future node visible in snapshot"; return false; }
  auto it = want.find(t->key);
  if (it == want.end() || it->second != t->val) {
    why = "key multiplicity"; return false;
  }
  uint64_t left = 0, right = 0;
  if (!check_tree(t->lch, lo, t->key, version, want, left, why) ||
      !check_tree(t->rch, t->key, hi, version, want, right, why)) return false;
  sum = left + t->val + right;
  return true;
}

int main(int argc, char** argv) {
  const uint64_t seed = argc > 1 ? std::stoull(argv[1]) : 17;
  const size_t initial_count = argc > 2 ? std::stoull(argv[2]) : 768;
  const size_t operation_count = argc > 3 ? std::stoull(argv[3]) : 900;
  const std::string profile = argc > 4 ? argv[4] : "balanced";
  const size_t thread_count = argc > 5 ? std::stoull(argv[5]) : 4;
  const size_t block_size = argc > 6 ? std::stoull(argv[6]) : 1000;
  const size_t client_count = argc > 7 ? std::stoull(argv[7]) : 1;
  std::mt19937_64 rng(seed);
  std::vector<uint64_t> initial;
  for (uint64_t i = 1; i <= initial_count; ++i) initial.push_back(i * 2);
  std::shuffle(initial.begin(), initial.end(), rng);

  std::vector<Op> ops;
  std::map<uint64_t, uint64_t> live;
  for (uint64_t k : initial) live[k]++;
  std::vector<uint64_t> live_keys = initial;
  std::map<uint64_t, size_t> live_pos;
  for (size_t i = 0; i < live_keys.size(); ++i) live_pos[live_keys[i]] = i;
  int insert_cut = profile == "read_heavy" ? 18 :
                   (profile == "update_heavy" ? 60 : (profile == "hotspot" ? 55 : 42));
  int delete_cut = profile == "read_heavy" ? 28 :
                   (profile == "update_heavy" ? 88 : (profile == "hotspot" ? 90 : 67));
  const uint64_t hot_limit = std::max<uint64_t>(128, initial_count / 128);
  for (size_t i = 0; i < operation_count; ++i) {
    uint64_t key = profile == "hotspot" && rng() % 100 < 98
                     ? 1 + rng() % hot_limit
                     : 1 + rng() % (4 * initial_count);
    int type = static_cast<int>(rng() % 100);
    if (type < insert_cut) {
      if (live[key] == 0) {
        live_pos[key] = live_keys.size();
        live_keys.push_back(key);
      }
      ops.push_back({1, key}); live[key]++;
    } else if (type < delete_cut && !live_keys.empty()) {
      if (profile == "hotspot" && rng() % 100 < 98) {
        do { key = 1 + rng() % hot_limit; } while (live.find(key) == live.end());
      } else {
        key = live_keys[rng() % live_keys.size()];
      }
      ops.push_back({2, key});
      if (--live[key] == 0) {
        live.erase(key);
        size_t pos = live_pos[key];
        uint64_t tail = live_keys.back();
        live_keys.pop_back();
        live_pos.erase(key);
        if (pos < live_keys.size()) {
          live_keys[pos] = tail;
          live_pos[tail] = pos;
        }
      }
    } else {
      ops.push_back({0, key});
    }
  }

  treap::node_ptr root = treap::build_parallel(initial.size(), initial.data());
  treap::scheduler engine(thread_count, ops.size(), block_size, root);
  for (const Op& op : ops) {
    if (op.type == 1) engine.insert_elem(op.key);
    else if (op.type == 2) engine.delete_elem(op.key);
    else engine.nop();
  }
  std::map<uint64_t, uint64_t> expected;
  for (uint64_t k : initial) expected[k]++;
  struct Checkpoint { size_t version; std::map<uint64_t, uint64_t> expected; };
  std::vector<Checkpoint> checkpoints;
  for (size_t version = 0; version <= ops.size(); ++version) {
    if (version > 0) {
      const Op& op = ops[version - 1];
      if (op.type == 1) expected[op.key]++;
      if (op.type == 2) {
        auto it = expected.find(op.key);
        if (it == expected.end()) { std::cerr << "invalid test delete\n"; return 2; }
        if (--it->second == 0) expected.erase(it);
      }
    }
    const size_t stride = std::max<size_t>(1, ops.size() / 16);
    if (version != 0 && version != ops.size() && version % stride != 0) continue;
    checkpoints.push_back({version, expected});
  }
  std::atomic_bool failed{false};
  std::vector<std::thread> clients;
  for (size_t client = 0; client < client_count; ++client) {
    clients.emplace_back([&, client] {
      for (size_t index = client; index < checkpoints.size(); index += client_count) {
        const auto& checkpoint = checkpoints[index];
        treap::node_ptr snapshot = engine.get_snapshot(checkpoint.version);
        uint64_t sum = 0;
        std::string why;
        if (!check_tree(snapshot, 0, UINT64_MAX, checkpoint.version,
                        checkpoint.expected, sum, why) ||
            sum != std::accumulate(checkpoint.expected.begin(), checkpoint.expected.end(),
              uint64_t{0}, [](uint64_t n, const auto& p) { return n + p.second; })) {
          failed.store(true, std::memory_order_relaxed);
          continue;
        }
        uint64_t left = 1 + ((seed ^ (checkpoint.version * 0x9E3779B97F4A7C15ULL)) %
                             (4 * initial_count - 128));
        uint64_t right = left + 127;
        uint64_t expected_range = 0;
        for (auto it = checkpoint.expected.lower_bound(left);
             it != checkpoint.expected.end() && it->first <= right; ++it)
          expected_range += it->second;
        if (treap::range_estimate(snapshot, left, right) != expected_range)
          failed.store(true, std::memory_order_relaxed);
      }
    });
  }
  for (auto& client : clients) client.join();
  if (failed.load(std::memory_order_relaxed)) {
    std::cerr << "concurrent snapshot or range query failed\n";
    return 1;
  }
  engine.process();
  std::cout << "correct\n";
  return 0;
}
