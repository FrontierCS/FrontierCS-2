#include <functional>
#include <spectrum/common/lock-util.hpp>
#include <spectrum/transaction/evm-hash.hpp>
#include <spectrum/transaction/evm-transaction.hpp>
#include <spectrum/workload/abstraction.hpp>
#include <spectrum/protocol/abstraction.hpp>
#include <spectrum/common/statistics.hpp>
#include <tuple>
#include <list>
#include <unordered_map>
#include <optional>
#include <atomic>
#include <memory>
#include <chrono>
#include <barrier>

namespace spectrum
{

#define K std::tuple<evmc::address, evmc::bytes32>
#define T CalvinTransaction

/// @brief calvin tranaction with local read and write set. 
struct CalvinTransaction: public Transaction {
    size_t      id;
    T*          should_wait{nullptr};
    bool        flag_conflict{false};
    std::atomic<bool>   committed{false};
    Prediction          prediction;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    CalvinTransaction(Transaction&& inner, size_t id);
    CalvinTransaction(CalvinTransaction&& tx);
};

/// @brief calvin table entry for first round execution
struct CalvinEntry {
    evmc::bytes32   value   = evmc::bytes32{0};
};

/// @brief calvin table for first round execution
struct CalvinTable: public Table<K, CalvinEntry, KeyHasher> {};

/// @brief calvin table entry for fallback pessimistic execution
struct CalvinLockEntry {
    std::vector<T*>     deps_get;
    std::vector<T*>     deps_put;
};

/// @brief calvin table for fallback pessimistic execution
struct CalvinLockTable: public Table<K, CalvinLockEntry, KeyHasher> {
    CalvinLockTable(size_t partitions);
};

/// @brief calvin protocol master class
class Calvin: public Protocol {

    private:
    Statistics&                         statistics;
    Workload&                           workload;
    CalvinTable                         table;
    CalvinLockTable                     lock_table;
    size_t                              repeat;
    size_t                              num_threads;
    std::atomic<bool>                   stop_flag{false};
    std::barrier<std::function<void()>> barrier;
    std::atomic<size_t>                 confirm_exit{0};
    std::atomic<size_t>                 counter{0};
    std::atomic<bool>                   has_conflict{false};
    std::vector<std::thread>            workers;
    friend class CalvinExecutor;

    public:
    Calvin(Workload& workload, Statistics& statistics, size_t num_threads, size_t table_partitions, size_t repeat);
    void Start() override;
    void Stop() override;

};

/// @brief routines to be executed in various execution stages
class CalvinExecutor {

    private:
    Statistics&                             statistics;
    Workload&                               workload;
    CalvinTable&                            table;
    CalvinLockTable&                        lock_table;
    size_t                                  num_threads;
    std::atomic<size_t>&                    confirm_exit;
    std::atomic<bool>&                      stop_flag;
    std::barrier<std::function<void()>>&    barrier;
    std::atomic<size_t>&                    counter;
    std::atomic<bool>&                      has_conflict;
    size_t                                  repeat;
    size_t                                  worker_id;

    public:
    CalvinExecutor(Calvin& calvin, size_t worker_id);
    void Run();
    void PrepareLockTable(T* tx);
    void Analyze(T* tx);
    void CleanLockTable(T* tx);

};

#undef K
#undef T

} // namespace spectrum
