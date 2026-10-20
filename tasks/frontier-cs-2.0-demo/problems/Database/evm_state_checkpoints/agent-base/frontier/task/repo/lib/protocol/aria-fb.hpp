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
#define T AriaTransaction

/// @brief aria tranaction with local read and write set. 
struct AriaTransaction: public Transaction {
    size_t      id;
    size_t      batch_id;
    bool        flag_conflict{false};
    std::atomic<bool>   committed{false};
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::unordered_map<K, evmc::bytes32, KeyHasher>  local_put;
    std::unordered_map<K, evmc::bytes32, KeyHasher>  local_get;
    AriaTransaction(Transaction&& inner, size_t id, size_t batch_id);
    AriaTransaction(AriaTransaction&& tx);
};

/// @brief aria table entry for first round execution
struct AriaEntry {
    evmc::bytes32   value   = evmc::bytes32{0};
    size_t  batch_id_get    = 0;
    size_t  batch_id_put    = 0;
    T*      reserved_get_tx = nullptr;
    T*      reserved_put_tx = nullptr;
};

/// @brief aria table for first round execution
struct AriaTable: public Table<K, AriaEntry, KeyHasher> {
    void ReserveGet(T* tx, const K& k);
    void ReservePut(T* tx, const K& k);
    bool CompareReservedGet(T* tx, const K& k);
    bool CompareReservedPut(T* tx, const K& k);
};

/// @brief aria table entry for fallback pessimistic execution
struct AriaLockEntry {
    std::vector<T*>     deps_get;
    std::vector<T*>     deps_put;
};

/// @brief aria table for fallback pessimistic execution
struct AriaLockTable: public Table<K, AriaLockEntry, KeyHasher> {
    AriaLockTable(size_t partitions);
};

/// @brief aria protocol master class
class Aria: public Protocol {

    private:
    Statistics&                         statistics;
    Workload&                           workload;
    AriaTable                           table;
    AriaLockTable                       lock_table;
    size_t                              repeat;
    size_t                              num_threads;
    bool                                enable_reordering;
    std::atomic<bool>                   stop_flag{false};
    std::barrier<std::function<void()>> barrier;
    std::atomic<size_t>                 confirm_exit{0};
    std::atomic<size_t>                 counter{0};
    std::atomic<bool>                   has_conflict{false};
    std::vector<std::thread>            workers;
    friend class AriaExecutor;

    public:
    Aria(Workload& workload, Statistics& statistics, size_t num_threads, size_t table_partitions, size_t repeat, bool enable_reordering);
    void Start() override;
    void Stop() override;

};

/// @brief routines to be executed in various execution stages
class AriaExecutor {

    private:
    Statistics&                             statistics;
    Workload&                               workload;
    AriaTable&                              table;
    AriaLockTable&                          lock_table;
    bool                                    enable_reordering;
    size_t                                  num_threads;
    std::atomic<size_t>&                    confirm_exit;
    std::atomic<bool>&                      stop_flag;
    std::barrier<std::function<void()>>&    barrier;
    std::atomic<size_t>&                    counter;
    std::atomic<bool>&                      has_conflict;
    size_t                                  repeat;
    size_t                                  worker_id;

    public:
    AriaExecutor(Aria& aria, size_t worker_id);
    void Run();
    void Execute(T* tx);
    void Reserve(T* tx);
    void Verify(T* tx);
    void Commit(T* tx);
    void PrepareLockTable(T* tx);
    void Fallback(T* tx);
    void CleanLockTable(T* tx);

};

#undef K
#undef T

} // namespace spectrum
