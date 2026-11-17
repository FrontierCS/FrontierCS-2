#include <spectrum/common/lock-util.hpp>
#include <spectrum/protocol/abstraction.hpp>
#include <spectrum/workload/abstraction.hpp>
#include <spectrum/common/statistics.hpp>
#include <spectrum/transaction/evm-hash.hpp>
#include <thread>
#include <vector>
#include <atomic>

namespace spectrum {

#define K std::tuple<evmc::address, evmc::bytes32>

using DummyTable = Table<K, evmc::bytes32, KeyHasher>;

class Dummy: public Protocol {

    private:
    DummyTable                  table;
    std::atomic<bool>           stop_flag{false};
    Workload&                   workload;
    size_t                      num_threads;
    Statistics&                 statistics;
    std::vector<std::thread>    executors;

    public:
    Dummy(Workload& workload, Statistics& statistics, size_t num_threads, size_t table_partitions, EVMType evm_type);
    void Start() override;
    void Stop()  override;

};

} // namespace spectrum
