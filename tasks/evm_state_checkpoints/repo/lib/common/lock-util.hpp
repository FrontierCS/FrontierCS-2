#pragma once
#include <memory>
#include <queue>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <evmc/evmc.hpp>
#include <glog/logging.h>
#include <vector>

namespace spectrum {

#define TP std::unique_ptr<T>

class SpinLock {
    
    private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

    public:
    void Lock() {
        while(flag.test_and_set(std::memory_order_acquire)) {}
    }
    void Unlock() {
        flag.clear(std::memory_order_release);
    }

};

template<typename Lock>
class Guard {

    private:
    Lock& lock;

    public:
    Guard(Lock& lock): lock{lock} { lock.Lock(); }
    ~Guard() { lock.Unlock(); }

};

template<typename T>
class LockQueue {

    private:
    SpinLock mu;
    std::queue<TP>  queue;

    public:
    TP Pop() {
        auto guard = Guard{mu};
        if (!queue.size()) return {nullptr};
        auto tx = const_cast<TP&>(queue.top()).release();
        queue.pop();
        return std::unique_ptr<T>(tx);
    }
    void Push(TP&& tx) {
        auto guard = Guard{mu};
        queue.push(std::move(tx));
    }
    size_t Size() {
        auto guard = Guard{mu};
        return queue.size();
    }

};

template<typename T>
class LockPriorityQueue {

    struct CMP {
        bool operator()(const TP& a, const TP& b) { return a->id > b->id; }
    };

    private:
    SpinLock  mu;
    std::priority_queue<TP, std::vector<TP>, CMP> queue;

    public:
    TP Pop() {
        auto guard = Guard{mu};
        if (!queue.size()) return {nullptr};
        auto tx = const_cast<TP&>(queue.top()).release();
        queue.pop();
        return std::unique_ptr<T>(tx);
    }
    void Push(TP&& tx) {
        auto guard = Guard{mu};
        queue.push(std::move(tx));
    }
    size_t Size() {
        auto guard = Guard{mu};
        return queue.size();
    }

};

#undef TP

template<typename K, typename V, typename Hasher>
class Table {

    private:
    size_t                  num_partitions;
    std::vector<SpinLock>   locks;
    std::vector<std::unordered_map<K, V, Hasher>> partitions;

    public:
    Table(size_t partitions);
    void Get(const K& k, std::function<void(const V& v)>&& vmap);
    void Put(const K& k, std::function<void(V& v)>&& vmap);

};

template<typename K, typename V, typename Hasher>
Table<K, V, Hasher>::Table(size_t partitions)
    : locks(partitions),
      partitions(partitions),
      num_partitions{partitions}
{}

template<typename K, typename V, typename Hasher>
void Table<K, V, Hasher>::Get(const K& k, std::function<void(const V& v)>&& vmap) {
    auto partition_id = ((size_t)Hasher()(k)) % num_partitions;
    DLOG(INFO) << "at partition " << partition_id;
    auto guard = Guard{locks[partition_id]};
    auto& partition = this->partitions[partition_id];
    if (partition.contains(k)) vmap(partition[k]);
}

template<typename K, typename V, typename Hasher>
void Table<K, V, Hasher>::Put(const K& k, std::function<void(V& v)>&& vmap) {
    auto partition_id = ((size_t)Hasher()(k)) % num_partitions;
    DLOG(INFO) << "at partition " << partition_id;
    auto guard = Guard{locks[partition_id]};
    auto& partition = this->partitions[partition_id];
    vmap(partition[k]);
}

} // namespace spectrum
