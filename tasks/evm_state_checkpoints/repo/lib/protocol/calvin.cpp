#include <spectrum/protocol/calvin.hpp>
#include <spectrum/common/thread-util.hpp>
#include <fmt/core.h>

/*
    This is an implementation of "Calvin: fast distributed transactions for partitioned database systems" (Alexander Thomson, Thaddeus Diamond, Shu-Chun Weng, Kun Ren, Philip Shao, Daniel J. Abadi). (single-partition)
 */

namespace spectrum 
{

#define K std::tuple<evmc::address, evmc::bytes32>
#define T CalvinTransaction

using namespace std::chrono;

/// @brief initialize calvin protocol
/// @param workload an evm transaction workload
/// @param batch_size batch size
/// @param num_threads the number of threads in thread pool
/// @param table_partitions the number of partitions in table
Calvin::Calvin(
    Workload& workload, Statistics& statistics,
    size_t num_threads, size_t table_partitions, size_t repeat
):
    workload{workload},
    statistics{statistics},
    barrier(num_threads, []{ DLOG(INFO) << "batch complete" << std::endl; }),
    table{table_partitions},
    lock_table{table_partitions},
    num_threads{num_threads},
    repeat{repeat}
{
    LOG(INFO) << fmt::format("Calvin(num_threads={}, table_partitions={}, repeat={})", num_threads, table_partitions, repeat) << std::endl;
}

/// @brief start calvin protocol
void Calvin::Start() {
    DLOG(INFO) << "calvin start";
    for (size_t i = 0; i < num_threads; ++i) {
        workers.push_back(std::thread([this, i]() {
            CalvinExecutor(*this, i).Run();
        }));
        PinRoundRobin(workers[i], i);
    }
}

/// @brief stop calvin protocol and return statistics
/// @return statistics of current execution
void Calvin::Stop() {
    stop_flag.store(true);
    for (size_t i = 0; i < num_threads; ++i) {
        workers[i].join();
    }
    DLOG(INFO) << "calvin stop";
}

/// @brief construct an empty calvin transaction
CalvinTransaction::CalvinTransaction(
    Transaction&& inner, size_t id
):
    Transaction{std::move(inner)},
    id{id},
    start_time{std::chrono::steady_clock::now()}
{}

CalvinTransaction::CalvinTransaction(CalvinTransaction&& tx):
    Transaction{std::move(tx)},
    id{tx.id},
    start_time{tx.start_time}
{}

/// @brief initialize an calvin lock table
/// @param partitions the number of partitions used in parallel hash table
CalvinLockTable::CalvinLockTable(size_t partitions): 
    Table::Table(partitions)
{}

/// @brief initialize an calvin executor
/// @param calvin the calvin configuration object
CalvinExecutor::CalvinExecutor(Calvin& calvin, size_t worker_id):
    statistics{calvin.statistics},
    workload{calvin.workload},
    table{calvin.table},
    lock_table{calvin.lock_table},
    stop_flag{calvin.stop_flag},
    barrier{calvin.barrier},
    counter{calvin.counter},
    has_conflict{calvin.has_conflict},
    num_threads{calvin.num_threads},
    repeat{calvin.repeat},
    confirm_exit{calvin.confirm_exit},
    worker_id{worker_id}
{}

/// @brief run transactions
void CalvinExecutor::Run() {
    auto batch = std::vector<T>();
    batch.reserve(repeat);
    while(true) {
        #define LATENCY duration_cast<microseconds>(steady_clock::now() - tx.start_time).count()
        // -- stage 1: generate and predict
        auto _stop = confirm_exit.load() == num_threads;
        barrier.arrive_and_wait();
        if (_stop) { return; }
        if (stop_flag.load()) {
            confirm_exit.compare_exchange_weak(worker_id, worker_id + 1);
        }
        has_conflict.store(false);
        for (size_t i = 0; i < repeat; ++i) {
            batch.emplace_back(workload.Next(), counter.fetch_add(1));
            batch[i].Analyze(batch[i].prediction);
            this->PrepareLockTable(&batch[i]);
        }
        // -- stage 2: analyze transaction using table information
        barrier.arrive_and_wait();
        for (auto& tx: batch) { this->Analyze(&tx); }
        // -- stage 3: wait & execute each transaction
        barrier.arrive_and_wait();
        auto count_committed = 0;
        while (count_committed != repeat) {for (auto& tx: batch) {
            if (tx.committed.load()) { continue; }
            if (tx.should_wait && !tx.should_wait->committed.load()) { continue; }
            tx.Execute();
            tx.committed.store(true);
            statistics.JournalExecute();
            statistics.JournalOperations(tx.CountOperations());
            statistics.JournalCommit(LATENCY);
            statistics.JournalMemory(tx.mm_count);
            ++count_committed;
        }}
        // -- stage 4: release lock
        barrier.arrive_and_wait();
        for (auto& tx: batch) {
            this->CleanLockTable(&tx);
        }
        batch.clear();
    }
}

/// @brief put transaction id (local id) into table
/// @param tx the transaction
void CalvinExecutor::PrepareLockTable(T* tx) {
    for (auto& key: tx->prediction.get) {
        lock_table.Put(key, [&](auto& entry) {
            entry.deps_get.push_back(tx);
        });
    }
    for (auto& key: tx->prediction.put) {
        lock_table.Put(key, [&](auto& entry) {
            entry.deps_put.push_back(tx);
        });
    }
}

/// @brief fallback execution without constant
/// @param tx the transaction
void CalvinExecutor::Analyze(T* tx) {
    // read from the public table
    tx->InstallGetStorageHandler([&](
        const evmc::address &addr,
        const evmc::bytes32 &key
    ) {
        auto tup = std::make_tuple(addr, key);
        auto value = evmc::bytes32{0};
        table.Get(tup, [&](auto& entry){
            value = entry.value;
        });
        return value;
    });
    // write directly into the public table
    tx->InstallSetStorageHandler([&](
        const evmc::address &addr, 
        const evmc::bytes32 &key,
        const evmc::bytes32 &value
    ) {
        auto tup = std::make_tuple(addr, key);
        table.Put(tup, [&](auto& entry){
            entry.value = value;
        });
        return evmc_storage_status::EVMC_STORAGE_MODIFIED;
    });
    // get the latest dependency and wait on it
    #define COND (_tx->id < tx->id && (tx->should_wait == nullptr || _tx->id > tx->should_wait->id))
    for (auto& key: tx->prediction.put) {
        lock_table.Get(key, [&](auto& entry) {
            for (auto _tx: entry.deps_get) { if (COND) { tx->should_wait = _tx; } }
            for (auto _tx: entry.deps_put) { if (COND) { tx->should_wait = _tx; } }
        });
    }
    for (auto& key: tx->prediction.get) {
        lock_table.Get(key, [&](auto& entry) {
            for (auto _tx: entry.deps_put) { if (COND) { tx->should_wait = _tx; } }
        });
    }
    #undef COND
}

/// @brief clean up the lock table
/// @param tx the transaction to clean up
void CalvinExecutor::CleanLockTable(T* tx) {
    for (auto& key: tx->prediction.put) {
        lock_table.Put(key, [&](auto& entry) {
            entry.deps_put.clear();
        });
    }
    for (auto& key: tx->prediction.get) {
        lock_table.Put(key, [&](auto& entry) {
            entry.deps_get.clear();
        });
    }
}

#undef K
#undef T

} // namespace spectrum
