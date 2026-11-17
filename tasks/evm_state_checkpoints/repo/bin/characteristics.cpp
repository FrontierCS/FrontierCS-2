#include <iostream>
#include <spectrum/common/hex.hpp>
#include <ranges>
#include <string_view>
#include <fmt/core.h>
#include <iterator>
#include <stdexcept>
#include <chrono>
#include <glog/logging.h>
#include <string.h>
#include <evmc/evmc.h>
#include <evmc/evmc.hpp>
#include "argparse.hpp"
#include <spectrum/common/glog-prefix.hpp>

#define K std::tuple<evmc::address, evmc::bytes32>

/// @brief mock table for collecting workload characteristics
class MockTable {

    private:
    std::unordered_map<K, evmc::bytes32, spectrum::KeyHasher> inner;
    std::unordered_map<K, std::vector<std::tuple<size_t, char>>, spectrum::KeyHasher> histogram;

    public:
    evmc::bytes32 GetStorage(
        size_t id,
        const evmc::address& addr, 
        const evmc::bytes32& key
    ) {
        auto& entry = histogram[std::make_tuple(addr, key)];
        if (!entry.size() || std::get<0>(entry[entry.size() - 1]) != id) {
            entry.push_back({id, 'g'});
        }
        else if (std::get<1>(entry[entry.size() - 1]) != 'd') {
            std::get<1>(entry[entry.size() - 1]) = 'd';
        }
        return inner[std::make_tuple(addr, key)];
    }

    void SetStorage(
        size_t id,
        const evmc::address& addr, 
        const evmc::bytes32& key, 
        const evmc::bytes32& value
    ) {
        auto& entry = histogram[std::make_tuple(addr, key)];
        if (!entry.size() || std::get<0>(entry[entry.size() - 1]) != id) {
            entry.push_back({id, 'p'});
        }
        else if (std::get<1>(entry[entry.size() - 1]) != 'p') {
            std::get<1>(entry[entry.size() - 1]) = 'd';
        }
        inner[std::make_tuple(addr, key)] = value;
    }

    std::unordered_map<K, std::string, spectrum::KeyHasher> Report(size_t counter) {
        auto collector_map = std::unordered_map<K, std::string, spectrum::KeyHasher>();
        for (const auto& entry: histogram) {
            auto j = size_t{0};
            auto k = std::get<0>(entry);
            auto v = std::get<1>(entry);
            auto collector = std::string();
            for (size_t i = 0; i != counter; i += 1) {
                if (j < v.size() && std::get<0>(v[j]) == i) {
                    auto op = std::get<1>(v[j]);
                    collector.push_back(op == 'g' ? '?' : (op == 'p' ? '!' : 'x'));
                    j += 1;
                }
                else {
                    collector.push_back('-');
                }
            }
            collector_map[k] = collector;
        }
        return collector_map;
    }

};

using namespace spectrum;

int main(int argc, char* argv[]) {
    google::InstallPrefixFormatter(PrefixFormatter);
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 1;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    CHECK(argc == 3) << "We only expect 2 flags. ";
    auto mock_table = MockTable();
    auto statistics = std::make_unique<Statistics>();
    auto workload   = ParseWorkload(argv[1]);
    auto duration   = to<milliseconds>(argv[2]);
    auto stop_flag  = std::atomic<bool>();
    auto start_time = steady_clock::now();
    auto counter   = size_t{0};
    auto handle     = std::thread([&]() {
        while (!stop_flag.load()) {
            auto transaction = workload->Next();
            auto start_time  = steady_clock::now();
            transaction.InstallGetStorageHandler([&](auto addr, auto key){
                return mock_table.GetStorage(counter, addr, key);
            });
            transaction.InstallSetStorageHandler([&](auto addr, auto key, auto value){
                mock_table.SetStorage(counter, addr, key, value);
                return evmc_storage_status::EVMC_STORAGE_ASSIGNED;
            });
            transaction.Execute();
            statistics->JournalExecute();
            statistics->JournalCommit(duration_cast<milliseconds>(steady_clock::now() - start_time).count());
            ++counter;
        }
    });
    // wait for a given duration, stop running, and print statistics
    std::this_thread::sleep_for(duration);
    stop_flag.store(true); handle.join();
    std::cerr << "'?' represents get, '!' represents put, 'x' represents both, '-' represents no-operation" << std::endl;
    std::cerr << statistics->PrintWithDuration(duration_cast<milliseconds>(steady_clock::now() - start_time));
    for (auto& entry: mock_table.Report(counter)) {
        auto& addr  = std::get<0>(std::get<0>(entry));
        auto& key   = std::get<1>(std::get<0>(entry));
        auto& hist  = std::get<1>(entry);
        std::cerr << to_hex(std::span{(uint8_t*)&addr, 20}) << ":" << to_hex(std::span{(uint8_t*)&key, 32}) << std::endl;
        std::cerr << hist << std::endl;
    }
}

#undef K
