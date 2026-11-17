#include <spectrum/protocol/aria-fb.hpp>
#include <spectrum/common/thread-util.hpp>
#include <fmt/core.h>

/*
    This is an implementation of "Aria: A Fast and Practical Deterministic OLTP Database" (Yi Lu, Xiangyao Yu, Lei Cao, Samuel Madden). 
    In this implementation we adopt the fallback strategy and reordering strategy (enabled by the option) discussed in this paper. 
 */

namespace spectrum 
{

#define K std::tuple<evmc::address, evmc::bytes32>
#define T AriaTransaction

using namespace std::chrono;

/// @brief initialize aria protocol
/// @param workload an evm transaction workload
/// @param batch_size batch size
/// @param num_threads the number of threads in thread pool
/// @param table_partitions the number of partitions in table
Aria::Aria(
    Workload& workload, Statistics& statistics,
    size_t num_threads, size_t table_partitions, size_t repeat,
    bool enable_reordering
):
    workload{workload},
    statistics{statistics},
    barrier(num_threads, []{ DLOG(INFO) << "batch complete" << std::endl; }),
    table{table_partitions},
    lock_table{table_partitions},
    enable_reordering{enable_reordering},
    num_threads{num_threads},
    repeat{repeat}
{
    LOG(INFO) << fmt::format("Aria(num_threads={}, table_partitions={}, repeat={}, enable_reordering={})", num_threads, table_partitions, repeat, enable_reordering) << std::endl;
}

/// @brief start aria protocol
void Aria::Start() {
    DLOG(INFO) << "aria start";
    for (size_t i = 0; i < num_threads; ++i) {
        workers.push_back(std::thread([this, i]() {
            AriaExecutor(*this, i).Run();
        }));
        PinRoundRobin(workers[i], i);
    }
}

/// @brief stop aria protocol and return statistics
/// @return statistics of current execution
void Aria::Stop() {
    stop_flag.store(true);
    for (size_t i = 0; i < num_threads; ++i) {
        workers[i].join();
    }
    DLOG(INFO) << "aria stop";
}

/// @brief construct an empty aria transaction
AriaTransaction::AriaTransaction(
    Transaction&& inner, 
    size_t id, size_t batch_id
):
    Transaction{std::move(inner)},
    id{id},
    batch_id{batch_id},
    start_time{std::chrono::steady_clock::now()}
{}

AriaTransaction::AriaTransaction(AriaTransaction&& tx):
    Transaction{std::move(tx)},
    id{tx.id},
    batch_id{tx.batch_id},
    start_time{tx.start_time}
{}

/// @brief reserved a get entry
/// @param tx the transaction
/// @param k the reserved key
void AriaTable::ReserveGet(T* tx, const K& k) {
    Table::Put(k, [&](AriaEntry& entry) {
        DLOG(INFO) << tx->batch_id << ":" <<  tx->id << " reserve get, current tx(" << entry.batch_id_get << ":" << entry.reserved_get_tx << ")" << std::endl;
        if (entry.batch_id_get != tx->batch_id || entry.reserved_get_tx == nullptr || entry.reserved_get_tx->id > tx->id) {
            entry.reserved_get_tx = tx;
            entry.batch_id_get = tx->batch_id;
            DLOG(INFO) << tx->batch_id << ":" << tx->id << " reserve get ok" << std::endl;
        }
    });
}

/// @brief reserve a put entry
/// @param tx the transaction
/// @param k the reserved key
void AriaTable::ReservePut(T* tx, const K& k) {
    Table::Put(k, [&](AriaEntry& entry) {
        DLOG(INFO) << tx->batch_id << ":" <<  tx->id << " reserve put, current tx(" << entry.batch_id_put << ":" << entry.reserved_put_tx << ")" << std::endl;
        if (entry.batch_id_put != tx->batch_id || entry.reserved_put_tx == nullptr || entry.reserved_put_tx->id > tx->id) {
            entry.reserved_put_tx = tx;
            entry.batch_id_put = tx->batch_id;
            DLOG(INFO) << tx->batch_id << ":" << tx->id << " reserve put ok" << std::endl;
        }
    });
}

/// @brief compare reserved get transaction
/// @param tx the transaction
/// @param k the compared key
/// @return if current transaction reserved this entry successfully
bool AriaTable::CompareReservedGet(T* tx, const K& k) {
    bool eq = true;
    Table::Get(k, [&](auto entry) {
        eq = entry.batch_id_get != tx->batch_id || (
            entry.reserved_get_tx == nullptr || 
            entry.reserved_get_tx->id == tx->id
        );
    });
    return eq;
}

/// @brief compare reserved put transaction
/// @param tx the transaction
/// @param k the compared key
/// @return if current transaction reserved this entry successfully
bool AriaTable::CompareReservedPut(T* tx, const K& k) {
    bool eq = true;
    Table::Get(k, [&](auto entry) {
        eq = entry.batch_id_put != tx->batch_id || (
            entry.reserved_put_tx == nullptr || 
            entry.reserved_put_tx->id == tx->id
        );
    });
    return eq;
}

/// @brief initialize an aria lock table
/// @param partitions the number of partitions used in parallel hash table
AriaLockTable::AriaLockTable(size_t partitions): 
    Table::Table(partitions)
{}

/// @brief initialize an aria executor
/// @param aria the aria configuration object
AriaExecutor::AriaExecutor(Aria& aria, size_t worker_id):
    statistics{aria.statistics},
    workload{aria.workload},
    table{aria.table},
    lock_table{aria.lock_table},
    enable_reordering{aria.enable_reordering},
    stop_flag{aria.stop_flag},
    barrier{aria.barrier},
    counter{aria.counter},
    has_conflict{aria.has_conflict},
    num_threads{aria.num_threads},
    repeat{aria.repeat},
    confirm_exit{aria.confirm_exit},
    worker_id{worker_id}
{}

/// @brief run transactions
void AriaExecutor::Run() {
    auto batch_id = size_t{0};
    auto batch = std::vector<T>();
    batch.reserve(repeat);
    while(true) {
        #define LATENCY duration_cast<microseconds>(steady_clock::now() - tx.start_time).count()
        // -- stage 1: generate and execute
        auto _stop = confirm_exit.load() == num_threads;
        barrier.arrive_and_wait();
        if (_stop) { return; }
        if (stop_flag.load()) { confirm_exit.compare_exchange_weak(worker_id, worker_id + 1); }
        batch_id += 1;
        has_conflict.store(false);
        for (size_t i = 0; i < repeat; ++i) {
            batch.emplace_back(workload.Next(), counter.fetch_add(1), batch_id);
        }
        for (auto& tx: batch) {
            this->Execute(&tx);
            this->Reserve(&tx);
            statistics.JournalExecute();
            statistics.JournalOperations(tx.CountOperations());
        }
        // -- stage 2: verify + commit (or prepare fallback)
        barrier.arrive_and_wait();
        for (auto& tx: batch) {
            this->Verify(&tx);
            if (tx.flag_conflict) {
                has_conflict.store(true);
                this->PrepareLockTable(&tx);
            }
            else {
                this->Commit(&tx);
                statistics.JournalCommit(LATENCY);
                statistics.JournalMemory(tx.mm_count);
            }
        }
        // -- stage 3: fallback (skipped if no conflicts occur)
        barrier.arrive_and_wait();
        if (!has_conflict.load()) {
            batch.clear();
            continue;
        }
        for (auto& tx: batch) {
            if (tx.flag_conflict) {
                this->Fallback(&tx);
                statistics.JournalExecute();
                statistics.JournalOperations(tx.CountOperations());
                statistics.JournalCommit(LATENCY);
                statistics.JournalMemory(tx.mm_count);
            }
        }
        // -- stage 4: clean up lock table
        barrier.arrive_and_wait();
        for (auto& tx: batch) {
            this->CleanLockTable(&tx);
        }
        batch.clear();
    }
}

/// @brief execute a transaction and journal write operations locally
/// @param tx the transaction
/// @param table the table
void AriaExecutor::Execute(T* tx) {
    // read from the public table
    tx->InstallGetStorageHandler([&](
        const evmc::address &addr,
        const evmc::bytes32 &key
    ) {
        DLOG(INFO) << "tx " << tx->id << " get" << std::endl;
        // if some write on this entry is issued previously, 
        //  the read dependency will be barricated from journal. 
        auto tup = std::make_tuple(addr, key);
        if (tx->local_put.contains(tup)) {
            return tx->local_put[tup];
        }
        if (tx->local_get.contains(tup)) {
            return tx->local_get[tup];
        }
        auto value = evmc::bytes32{0};
        table.Get(tup, [&](auto& entry){
            value = entry.value;
        });
        tx->local_get[tup] = value;
        return value;
    });
    // write locally to local storage
    tx->InstallSetStorageHandler([&](
        const evmc::address &addr, 
        const evmc::bytes32 &key,
        const evmc::bytes32 &value
    ) {
        DLOG(INFO) << "tx " << tx->id << " set" << std::endl;
        auto tup = std::make_tuple(addr, key);
        tx->local_put[tup] = value;
        return evmc_storage_status::EVMC_STORAGE_MODIFIED;
    });
    // execute the transaction
    tx->Execute();
}

/// @brief journal the smallest reader/writer transaction to table
/// @param tx the transaction
void AriaExecutor::Reserve(T* tx) {
    // journal all entries to the reservation table
    DLOG(INFO) << "tx " << tx->id << " reserve" << std::endl;
    for (auto& tup: tx->local_get) {
        table.ReserveGet(tx, std::get<0>(tup));
    }
    for (auto& tup: tx->local_put) {
        table.ReservePut(tx, std::get<0>(tup));
    }
}

/// @brief verify transaction by checking dependencies
/// @param tx the transaction, flag_conflict will be altered
void AriaExecutor::Verify(T* tx) {
    // conceptually, we take a snapshot on the database before we execute a batch
    //  , and all transactions are executed viewing the snapshot. 
    // however, we want the global state transitioned 
    //  as if we executed some of these transactions sequentially. 
    // therefore, we have to pick some transactions and arange them into a sequence. 
    // this algorithm implicitly does it for us. 
    bool war = false, raw = false, waw = false;
    for (auto& tup: tx->local_get) {
        // the value is updated, snapshot contains out-dated value
        raw |= !table.CompareReservedPut(tx, std::get<0>(tup));
    }
    for (auto& tup: tx->local_put) {
        // the value is read before, therefore we should not update it
        war |= !table.CompareReservedGet(tx, std::get<0>(tup));
    }
    for (auto& tup: tx->local_put) {
        // if some write happened after write
        waw |= !table.CompareReservedPut(tx, std::get<0>(tup));
    }
    if (enable_reordering) {
        tx->flag_conflict = waw || (raw && war);
        DLOG(INFO) << "abort " << tx->batch_id << ":" << tx->id << std::endl;
    }
    else {
        tx->flag_conflict = waw || raw;
        DLOG(INFO) << "abort " << tx->batch_id << ":" << tx->id << std::endl;
    }
}

/// @brief commit written values into table
/// @param tx the transaction
void AriaExecutor::Commit(T* tx) {
    for (auto& tup: tx->local_put) {
        table.Put(std::get<0>(tup), [&](auto& entry) {
            entry.value = std::get<1>(tup);
        });
    }
}

/// @brief put transaction id (local id) into table
/// @param tx the transaction
void AriaExecutor::PrepareLockTable(T* tx) {
    for (auto& tup: tx->local_get) {
        lock_table.Put(std::get<0>(tup), [&](auto& entry) {
            entry.deps_get.push_back(tx);
        });
    }
    for (auto& tup: tx->local_put) {
        lock_table.Put(std::get<0>(tup), [&](auto& entry) {
            entry.deps_put.push_back(tx);
        });
    }
}

/// @brief fallback execution without constant
/// @param tx the transaction
void AriaExecutor::Fallback(T* tx) {
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
    T* should_wait = nullptr;
    #define COND (_tx->id < tx->id && (should_wait == nullptr || _tx->id > should_wait->id))
    for (auto& tup: tx->local_put) {
        lock_table.Get(std::get<0>(tup), [&](auto& entry) {
            for (auto _tx: entry.deps_get) { if (COND) { should_wait = _tx; } }
            for (auto _tx: entry.deps_put) { if (COND) { should_wait = _tx; } }
        });
    }
    for (auto& tup: tx->local_get) {
        lock_table.Get(std::get<0>(tup), [&](auto& entry) {
            for (auto _tx: entry.deps_put) { if (COND) { should_wait = _tx; } }
        });
    }
    #undef COND
    while(should_wait && !should_wait->committed.load()) {}
    tx->Execute();
    tx->committed.store(true);
}

/// @brief clean up the lock table
/// @param tx the transaction to clean up
void AriaExecutor::CleanLockTable(T* tx) {
    for (auto& tup: tx->local_put) {
        lock_table.Put(std::get<0>(tup), [&](auto& entry) {
            entry.deps_put.clear();
        });
    }
    for (auto& tup: tx->local_get) {
        lock_table.Put(std::get<0>(tup), [&](auto& entry) {
            entry.deps_get.clear();
        });
    }
}

#undef K
#undef T

} // namespace spectrum
