#include <spectrum/protocol/dummy.hpp>
#include <fmt/core.h>

namespace spectrum {

using namespace std::chrono;

Dummy::Dummy(Workload& workload, Statistics& statistics, size_t num_threads, size_t table_partitions, EVMType evm_type):
    workload{workload},
    statistics{statistics},
    num_threads{num_threads},
    table(table_partitions)
{
    LOG(INFO) << fmt::format("Dummy(num_threads={}, table_partitions={})", num_threads, table_partitions) << std::endl;
    workload.SetEVMType(evm_type);
}

void Dummy::Start() {
    for (size_t i = 0; i < num_threads; ++i) {executors.push_back(std::thread([this]() {while(!stop_flag.load()) {
        auto tx         = workload.Next();
        auto start_time = steady_clock::now();
        tx.InstallGetStorageHandler([&](auto& address, auto& key) {
            evmc::bytes32 v; table.Get({address, key}, [&](auto& _v) { v = _v; });
            return v;
        });
        tx.InstallSetStorageHandler([&](auto& address, auto& key, auto& v) {
            table.Put({address, key}, [&](auto& _v) { _v = v; });
            return evmc_storage_status::EVMC_STORAGE_MODIFIED;
        });
        tx.Execute();
        statistics.JournalExecute();
        statistics.JournalOperations(tx.CountOperations());
        statistics.JournalCommit(duration_cast<microseconds>(steady_clock::now() - start_time).count());
                statistics.JournalMemory(tx.mm_count);
    }}));}
}

void Dummy::Stop() {
    stop_flag.store(true);
    for (auto& x: executors) { x.join(); }
}

} // namespace spectrum
