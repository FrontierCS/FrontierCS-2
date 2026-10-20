#include <spectrum/workload/abstraction.hpp>
#include <spectrum/common/lock-util.hpp>
#include <spectrum/protocol/abstraction.hpp>
#include <spectrum/transaction/evm-hash.hpp>
#include <list>
#include <atomic>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <barrier>

namespace spectrum {

using namespace std::chrono;

// some shorthands to prevent prohibitively long names
#define K std::tuple<evmc::address, evmc::bytes32>
#define V SparkleVersionList
#define T SparkleTransaction

struct SparkleTransaction: public Transaction {
    size_t      id;
    size_t      execution_count{0};
    std::vector<std::tuple<K, evmc::bytes32, size_t>>   tuples_get{};
    std::vector<std::tuple<K, evmc::bytes32>>           tuples_put{};
    SpinLock        rerun_flag_mu;
    bool            rerun_flag{false};
    bool            berun_flag{false};
    time_point<steady_clock>    start_time;
    SparkleTransaction(Transaction&& inner, size_t id);
    bool HasRerunFlag();
    void SetRerunFlag(bool flag);
};

struct SparkleEntry {
    evmc::bytes32   value;
    size_t          version;
    // we store raw pointers here because when a transaction is destructed, it always removes itself from table. 
    std::unordered_set<T*>  readers;
};

struct SparkleVersionList {
    T*          tx = nullptr;
    std::list<SparkleEntry> entries;
    // readers that read default value
    std::unordered_set<T*>  readers_default;
};

struct SparkleTable: private Table<K, V, KeyHasher> {

    SparkleTable(size_t partitions);
    void Get(T* tx, const K& k, evmc::bytes32& v, size_t& version);
    void Put(T* tx, const K& k, const evmc::bytes32& v);
    bool Lock(T* tx, const K& k);
    bool Unlock(T* tx, const K& k);
    void RegretGet(T* tx, const K& k, size_t version);
    void RegretPut(T* tx, const K& k);
    void ClearGet(T* tx, const K& k, size_t version);
    void ClearPut(T* tx, const K& k);

};

using SparkleQueue = LockPriorityQueue<T>;
class SparkleExecutor;

class Sparkle: public Protocol {

    private:
    size_t              num_executors;
    Workload&           workload;
    SparkleTable        table;
    Statistics&         statistics;
    std::atomic<size_t> last_executed{1};
    std::atomic<size_t> last_finalized{0};
    std::atomic<bool>   stop_flag{false};
    std::vector<std::thread>    executors{};
    std::barrier<std::function<void()>>  stop_latch;

    friend class SparkleExecutor;

    public:
    Sparkle(Workload& workload, Statistics& statistics, size_t num_executors, size_t table_partitions);
    void Start() override;
    void Stop() override;

};

/// @brief an executor that fetch transactions from queue and execute them
class SparkleExecutor {

    private:
    Workload&               workload;
    SparkleTable&           table;
    Statistics&             statistics;
    std::atomic<size_t>&    last_executed;
    std::atomic<size_t>&    last_finalized;
    std::atomic<bool>&      stop_flag;
    SparkleQueue            queue;
    std::unique_ptr<T>      tx{nullptr};
    std::barrier<std::function<void()>>&           stop_latch;

    public:
    SparkleExecutor(Sparkle& sparkle);
    void Generate();
    void Finalize();
    void ReExecute();
    void Run();

};

#undef T
#undef V
#undef K

} // namespace spectrum
