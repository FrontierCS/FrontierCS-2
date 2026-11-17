#include "evmc/evmc.hpp"
#include <spectrum/protocol/spectrum.hpp>
#include <spectrum/common/lock-util.hpp>
#include <spectrum/common/hex.hpp>
#include <spectrum/common/thread-util.hpp>
#include <functional>
#include <thread>
#include <chrono>
#include <glog/logging.h>
#include <ranges>
#include <fmt/core.h>

/*
    This is an implementation of "Spectrum: Speedy and Strictly-Deterministic Smart Contract Transactions for Blockchain Ledgers" (Zhihao Chen, Tianji Yang, Yixiao Zheng, Zhao Zhang, Cheqing Jin and Aoying Zhou). 
    (with partial rollback)
 */

namespace spectrum {

using namespace std::chrono;

#define K std::tuple<evmc::address, evmc::bytes32>
#define V SpectrumVersionList
#define T SpectrumTransaction

/// @brief wrap a base transaction into a spectrum transaction
/// @param inner the base transaction
/// @param id transaction id
SpectrumTransaction::SpectrumTransaction(Transaction&& inner, size_t id):
    Transaction{std::move(inner)},
    id{id},
    start_time{std::chrono::steady_clock::now()}
{}

/// @brief determine transaction has to rerun
/// @return if transaction has to rerun
bool SpectrumTransaction::HasWAR() {
    auto guard = Guard{rerun_keys_mu};
    return rerun_keys.size() != 0;
}

/// @brief call the transaction to rerun providing the key that caused it
/// @param key the key that caused rerun
void SpectrumTransaction::SetWAR(const K& key, size_t cause_id) {
    auto guard = Guard{rerun_keys_mu};
    rerun_keys.push_back(key);
    should_wait = std::max(should_wait, cause_id);
}

/// @brief the multi-version table for spectrum
/// @param partitions the number of partitions
SpectrumTable::SpectrumTable(size_t partitions):
    Table<K, V, KeyHasher>{partitions}
{}

/// @brief get a value
/// @param tx the transaction that reads the value
/// @param k the key of the read entry
/// @param v (mutated to be) the value of read entry
/// @param version (mutated to be) the version of read entry
void SpectrumTable::Get(T* tx, const K& k, evmc::bytes32& v, size_t& version) {
    Table::Put(k, [&](V& _v) {
        auto rit = _v.entries.rbegin();
        auto end = _v.entries.rend();
        while (rit != end) {
            if (rit->version > tx->id) {
                ++rit; continue;
            }
            v = rit->value;
            version = rit->version;
            rit->readers.insert(tx);
            DLOG(INFO) << tx->id << "(" << tx << ")" << " read " << KeyHasher()(k) % 1000 << " version " << rit->version << std::endl;
            return;
        }
        version = 0;
        DLOG(INFO) << tx->id << "(" << tx << ")" << " read " << KeyHasher()(k) % 1000 << " version 0" << std::endl;
        _v.readers_default.insert(tx);
        v = evmc::bytes32{0};
    });
}

/// @brief put a value
/// @param tx the transaction that writes the value
/// @param k the key of the written entry
/// @param v the value to write
void SpectrumTable::Put(T* tx, const K& k, const evmc::bytes32& v) {
    CHECK(tx->id > 0) << "we reserve version(0) for default value";
    DLOG(INFO) << tx->id << "(" << tx << ")" << " write " << KeyHasher()(k) % 1000 << std::endl;
    Table::Put(k, [&](V& _v) {
        auto rit = _v.entries.rbegin();
        auto end = _v.entries.rend();
        // search from insertion position
        while (rit != end) {
            if (rit->version > tx->id) {
                ++rit; continue;
            }
            // abort transactions that read outdated keys
            for (auto _tx: rit->readers) {
                DLOG(INFO) << KeyHasher()(k) % 1000 << " has read dependency " << "(" << _tx << ")" << std::endl;
                if (_tx->id > tx->id) {
                    DLOG(INFO) << tx->id << " abort " << _tx->id << std::endl;
                    _tx->SetWAR(k, tx->id);
                }
            }
            break;
        }
        for (auto _tx: _v.readers_default) {
            DLOG(INFO) << KeyHasher()(k) % 1000 << " has read dependency " << "(" << _tx << ")" << std::endl;
            if (_tx->id > tx->id) {
                DLOG(INFO) << tx->id << " abort " << _tx->id << std::endl;
                _tx->SetWAR(k, tx->id);
            }
        }
        // handle duplicated write on the same key
        if (rit != end && rit->version == tx->id) {
            rit->value = v;
            return;
        }
        // insert an entry
        _v.entries.insert(rit.base(), SpectrumEntry {
            .value   = v,
            .version = tx->id,
            .readers = std::unordered_set<T*>()
        });
    });
}

/// @brief remove a read dependency from this entry
/// @param tx the transaction that previously read this entry
/// @param k the key of read entry
void SpectrumTable::RegretGet(T* tx, const K& k, size_t version) {
    DLOG(INFO) << "remove read record " << tx->id << "(" << tx << ")" << " from " << KeyHasher()(k) % 1000 << std::endl;
    Table::Put(k, [&](V& _v) {
        auto vit = _v.entries.begin();
        auto end = _v.entries.end();
        while (vit != end) {
            if (vit->version != version) {
                ++vit; continue;
            }
            vit->readers.erase(tx);
            break;
        }
        if (version == 0) {
            _v.readers_default.erase(tx);
        }
        // run an extra check in debug mode
        #if !defined(NDEBUG)
        {
            auto end = _v.entries.end();
            for (auto vit = _v.entries.begin(); vit != end; ++vit) {
                DLOG(INFO) << "spot version " << vit->version << std::endl;
                if (vit->readers.contains(tx)) {
                    DLOG(FATAL) << "didn't remove " << tx->id << "(" << tx << ")" << " still on version " << vit->version  << std::endl;
                }
            }
            if (_v.readers_default.contains(tx)) {
                DLOG(FATAL) << "didn't remove " << tx->id << "(" << tx << ")" << " still on version " << vit->version  << std::endl;
            }
        }
        #endif
    });
}

/// @brief undo a put operation and abort all dependent transactions
/// @param tx the transaction that previously put into this entry
/// @param k the key of this put entry
void SpectrumTable::RegretPut(T* tx, const K& k) {
    DLOG(INFO) << "remove write record " << tx->id << "(" << tx << ")" << " from " << KeyHasher()(k) % 1000 << std::endl;
    Table::Put(k, [&](V& _v) {
        auto vit = _v.entries.begin();
        auto end = _v.entries.end();
        while (vit != end) {
            if (vit->version != tx->id) {
                ++vit; continue;
            }
            // abort transactions that read from current transaction
            for (auto _tx: vit->readers) {
                DLOG(INFO) << KeyHasher()(k) % 1000 << " has read dependency " << "(" << _tx << ")" << std::endl;
                DLOG(INFO) << tx->id << " abort " << _tx->id << std::endl;
                _tx->SetWAR(k, tx->id);
            }
            break;
        }
        if (vit != end) { _v.entries.erase(vit); }
    });
}

/// @brief remove a read dependency from this entry
/// @param tx the transaction that previously read this entry
/// @param k the key of read entry
/// @param version the version of read entry, which indicates the transaction that writes this value
void SpectrumTable::ClearGet(T* tx, const K& k, size_t version) {
    DLOG(INFO) << "remove read record " << tx->id << "(" << tx << ")" << " from " << KeyHasher()(k) % 1000 << std::endl;
    Table::Put(k, [&](V& _v) {
        auto vit = _v.entries.begin();
        auto end = _v.entries.end();
        while (vit != end) {
            if (vit->version != version) {
                ++vit; continue;
            }
            DLOG(INFO) << "remove " << tx->id << "(" << tx << ")" << " from version " << vit->version << std::endl; 
            vit->readers.erase(tx);
            break;
        }
        if (version == 0) {
            _v.readers_default.erase(tx);
        }
        // run an extra check in debug mode
        #if !defined(NDEBUG)
        {
            auto end = _v.entries.end();
            for (auto vit = _v.entries.begin(); vit != end; ++vit) {
                DLOG(INFO) << "spot version " << vit->version << std::endl;
                if (vit->readers.contains(tx)) {
                    DLOG(FATAL) << "didn't remove " << tx->id << "(" << tx << ")" << " still on version " << vit->version  << std::endl;
                }
            }
            if (_v.readers_default.contains(tx)) {
                DLOG(FATAL) << "didn't remove " << tx->id << "(" << tx << ")" << " still on version " << vit->version  << std::endl;
            }
        }
        #endif
    });
}

/// @brief remove versions preceeding current transaction
/// @param tx the transaction the previously wrote this entry
/// @param k the key of written entry
void SpectrumTable::ClearPut(T* tx, const K& k) {
    DLOG(INFO) << "remove write record before " << tx->id << "(" << tx << ")" << " from " << KeyHasher()(k) % 1000 << std::endl;
    Table::Put(k, [&](V& _v) {
        while (_v.entries.size() && _v.entries.front().version < tx->id) {
            _v.entries.pop_front();
        }
    });
}

/// @brief spectrum initialization parameters
/// @param workload the transaction generator
/// @param table_partitions the number of parallel partitions to use in the hash table
Spectrum::Spectrum(Workload& workload, Statistics& statistics, size_t num_executors, size_t table_partitions, EVMType evm_type):
    workload{workload},
    statistics{statistics},
    num_executors{num_executors},
    table{table_partitions},
    stop_latch{static_cast<ptrdiff_t>(num_executors), []{}}
{
    LOG(INFO) << fmt::format("Spectrum(num_executors={}, table_partitions={}, evm_type={})", num_executors, table_partitions, evm_type);
    workload.SetEVMType(evm_type);
}

/// @brief start spectrum protocol
/// @param num_executors the number of threads to start
void Spectrum::Start() {
    stop_flag.store(false);
    for (size_t i = 0; i != num_executors; ++i) {
        executors.push_back(std::thread([this]{
            std::make_unique<SpectrumExecutor>(*this)->Run();
        }));
        PinRoundRobin(executors[i], i);
    }
}

/// @brief stop spectrum protocol
void Spectrum::Stop() {
    stop_flag.store(true);
    for (auto& x: executors) 	{ x.join(); }
}

/// @brief spectrum executor
/// @param spectrum spectrum initialization paremeters
SpectrumExecutor::SpectrumExecutor(Spectrum& spectrum):
    table{spectrum.table},
    last_finalized{spectrum.last_finalized},
    stop_flag{spectrum.stop_flag},
    statistics{spectrum.statistics},
    workload{spectrum.workload},
    last_executed{spectrum.last_executed},
    stop_latch{spectrum.stop_latch}
{}

/// @brief generate a transaction and execute it
void SpectrumExecutor::Generate() {
    if(tx != nullptr) return;
    tx = std::make_unique<T>(workload.Next(), last_executed.fetch_add(1));
    tx->start_time = steady_clock::now();
    tx->berun_flag.store(true);
    tx->InstallSetStorageHandler([this](
        const evmc::address &addr, 
        const evmc::bytes32 &key, 
        const evmc::bytes32 &value
    ) {
        auto _key = std::make_tuple(addr, key);
        tx->tuples_put.push_back({
            .key = _key, 
            .value = value, 
            .is_committed=false
        });
        if (tx->HasWAR()) {
            DLOG(INFO) << "spectrum tx " << tx->id << " break" << std::endl;
            tx->Break();
        }
        DLOG(INFO) << "tx " << tx->id <<
            " tuples put: " << tx->tuples_put.size() <<
            " tuples get: " << tx->tuples_get.size();
        return evmc_storage_status::EVMC_STORAGE_MODIFIED;
    });
    tx->InstallGetStorageHandler([this](
        const evmc::address &addr, 
        const evmc::bytes32 &key
    ) {
        auto _key  = std::make_tuple(addr, key);
        auto value = evmc::bytes32{0};
        auto version = size_t{0};
        for (auto& tup: tx->tuples_put | std::views::reverse) {
            if (tup.key != _key) { continue; }
            DLOG(INFO) << "spectrum tx " << tx->id << " has key " << KeyHasher()(_key) % 1000 << " in tuples_put. ";
            return tup.value;
        }
        for (auto& tup: tx->tuples_get) {
            if (tup.key != _key) { continue; }
            DLOG(INFO) << "spectrum tx " << tx->id << " has key " << KeyHasher()(_key) % 1000 << " in tuples_get. ";
            return tup.value;
        }
        DLOG(INFO) << "tx " << tx->id << " " << 
            " read(" << tx->tuples_get.size() << ")" << 
            " key(" << KeyHasher()(_key) % 1000 << ")" << std::endl;
        table.Get(tx.get(), _key, value, version);
        tx->tuples_get.push_back({
            .key            = _key, 
            .value          = value, 
            .version        = version,
            .tuples_put_len = tx->tuples_put.size(),
            .checkpoint_id  = tx->MakeCheckpoint()
        });
        // we have to break after make checkpoint
        //   , or we will snapshot the break signal into the checkpoint!
        if (tx->HasWAR()) {
            DLOG(INFO) << "spectrum tx " << tx->id << " break" << std::endl;
            tx->Break();
        }
        return value;
    });
    DLOG(INFO) << "spectrum execute " << tx->id;
    tx->Execute();
    statistics.JournalExecute();
    statistics.JournalOperations(tx->CountOperations());
    // commit all results if possible & necessary
    for (auto entry: tx->tuples_put) {
        if (tx->HasWAR()) { break; }
        if (entry.is_committed) { continue; }
        table.Put(tx.get(), entry.key, entry.value);
        entry.is_committed = true;
    }
}

/// @brief rollback transaction with given rollback signal
/// @param tx the transaction to rollback
void SpectrumExecutor::ReExecute() {
    DLOG(INFO) << "spectrum re-execute " << tx->id;
    // get current rerun keys
    std::vector<K> rerun_keys{};
    {
        auto guard = Guard{tx->rerun_keys_mu}; 
        std::swap(tx->rerun_keys, rerun_keys);
    }
    auto back_to = ~size_t{0};
    // find checkpoint
    for (auto& key: rerun_keys) {
        for (size_t i = 0; i < tx->tuples_get.size(); ++i) {
            if (tx->tuples_get[i].key != key) { continue; }
            back_to = std::min(i, back_to); break;
        }
    }
    // good news: we don't have to rollback, so just resume execution
    if (back_to == ~size_t{0}) {
        DLOG(INFO) << "tx " << tx->id << " do not have to rollback" << std::endl;
        tx->Execute(); return;
    }
    // bad news: we have to rollback
    auto& tup = tx->tuples_get[back_to];
    tx->ApplyCheckpoint(tup.checkpoint_id);
    for (size_t i = tup.tuples_put_len; i < tx->tuples_put.size(); ++i) {
        if (tx->tuples_put[i].is_committed) {
            table.RegretPut(tx.get(), tx->tuples_put[i].key);
        }
    }
    for (size_t i = back_to; i < tx->tuples_get.size(); ++i) {
        table.RegretGet(tx.get(), tx->tuples_get[i].key, tx->tuples_get[i].version);
    }
    tx->tuples_put.resize(tup.tuples_put_len);
    tx->tuples_get.resize(back_to);
    DLOG(INFO) << "tx " << tx->id <<
        " tuples put: " << tx->tuples_put.size() <<
        " tuples get: " << tx->tuples_get.size();
    tx->Execute();
    statistics.JournalExecute();
    statistics.JournalOperations(tx->CountOperations());
    // commit all results if possible & necessary
    for (auto entry: tx->tuples_put) {
        if (tx->HasWAR()) { break; }
        if (entry.is_committed) { continue; }
        table.Put(tx.get(), entry.key, entry.value);
        entry.is_committed = true;
    }
}

/// @brief finalize a spectrum transaction
void SpectrumExecutor::Finalize() {
    DLOG(INFO) << "spectrum finalize " << tx->id;
    last_finalized.fetch_add(1, std::memory_order_seq_cst);
    for (auto entry: tx->tuples_get) {
        table.ClearGet(tx.get(), entry.key, entry.version);
    }
    for (auto entry: tx->tuples_put) {
        table.ClearPut(tx.get(), entry.key);
    }
    auto latency = duration_cast<microseconds>(steady_clock::now() - tx->start_time).count();
    statistics.JournalCommit(latency);
    statistics.JournalMemory(tx->mm_count);
    tx = nullptr;
}

/// @brief start an executor
void SpectrumExecutor::Run() {
    while (!stop_flag.load()) {
        // first generate a transaction
        Generate();
        if (tx->HasWAR()) {
            // if there are some re-run keys, re-execute to obtain the correct result
            ReExecute();
        }
        else if (last_finalized.load() + 1 == tx->id && !tx->HasWAR()) {
            // if last transaction has finalized, and currently i don't have to re-execute, 
            // then i can final commit and do another transaction. 
            Finalize();
        }
    }
    stop_latch.arrive_and_wait();
}

#undef T
#undef V
#undef K

} // namespace spectrum
