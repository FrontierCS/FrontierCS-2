#include <spectrum/workload/abstraction.hpp>
#include <spectrum/common/lock-util.hpp>
#include <spectrum/protocol/abstraction.hpp>
#include <spectrum/transaction/evm-hash.hpp>
#include <list>
#include <atomic>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <thread>
#include <barrier>

namespace spectrum {

// some shorthands to prevent prohibitively long names
#define K std::tuple<evmc::address, evmc::bytes32>
#define V SparklePreSchedVersionList
#define T SparklePreSchedTransaction

using namespace std::chrono;

struct SparklePreSchedPutTuple {
    K               key;
    evmc::bytes32   value;
    bool            is_committed;
};

struct SparklePreSchedGetTuple {
    K               key;
    evmc::bytes32   value;
    size_t          version;
    size_t          tuples_put_len;
    size_t          checkpoint_id;
};

struct SparklePreSchedTransaction: public Transaction {
    size_t      id;
    // size_t      should_wait{0};
    SpinLock            rerun_keys_mu;
    std::vector<K>      rerun_keys;
    std::atomic<bool>   berun_flag{false};
    time_point<steady_clock>            start_time;
    std::unordered_map<K, size_t, KeyHasher>    should_wait;
    std::vector<SparklePreSchedGetTuple>       tuples_get{};
    std::vector<SparklePreSchedPutTuple>       tuples_put{};
    SparklePreSchedTransaction(Transaction&& inner, size_t id);
    bool HasWAR();
    size_t ShouldWait(const K& key);
    void SetWAR(const K& key, size_t cause_id, bool pre_schedule);
};

struct SparklePreSchedEntry {
    evmc::bytes32   value;
    size_t          version;
    // we store raw pointers here because when a transaction is destructed, it always removes itself from table. 
    std::unordered_set<T*>  readers;
};

struct SparklePreSchedVersionList {
    T*          tx = nullptr;
    std::list<SparklePreSchedEntry> entries;
    // readers that read default value
    std::unordered_set<T*>  readers_default;
};

struct SparklePreSchedLockTable: private Table<K, V, KeyHasher> {
    SparklePreSchedLockTable(size_t partitions);
    void Get(T* tx, const K& k);
    void Put(T* tx, const K& k);
    void ClearPut(T* tx, const K& k);
    void ClearGet(T* tx, const K& k);

};

struct SparklePreSchedTable: private Table<K, V, KeyHasher> {

    SparklePreSchedTable(size_t partitions);
    void Get(T* tx, const K& k, evmc::bytes32& v, size_t& version);
    void Put(T* tx, const K& k, const evmc::bytes32& v);
    void RegretGet(T* tx, const K& k, size_t version);
    void RegretPut(T* tx, const K& k);
    void ClearGet(T* tx, const K& k, size_t version);
    void ClearPut(T* tx, const K& k);
    bool Lock(T* tx, const K& k);
    bool Unlock(T* tx, const K& k);

};

using SparklePreSchedQueue = LockPriorityQueue<T>;
class SparklePreSchedExecutor;

class SparklePreSched: public Protocol {

    private:
    size_t              num_executors;
    Workload&           workload;
    SparklePreSchedTable       table;
    SparklePreSchedLockTable   lock_table;
    Statistics&         statistics;
    std::atomic<size_t> last_executed{1};
    std::atomic<size_t> last_finalized{0};
    std::atomic<size_t> last_scheduled{0};
    std::atomic<size_t> last_committed{0};
    std::atomic<bool>   stop_flag{false};
    std::vector<std::thread>    executors{};
    std::barrier<std::function<void()>>            stop_latch;
    friend class SparklePreSchedExecutor;

    public:
    SparklePreSched(Workload& workload, Statistics& statistics, size_t num_executors, size_t table_partitions, EVMType evm_type);
    void Start() override;
    void Stop() override;

};

class SparklePreSchedExecutor {

    private:
    Workload&               workload;
    SparklePreSchedTable&          table;
    SparklePreSchedLockTable&  lock_table;
    Statistics&             statistics;
    std::atomic<size_t>&    last_executed;
    std::atomic<size_t>&    last_finalized;
    std::atomic<size_t>&        last_scheduled;
    std::atomic<size_t>&        last_committed;
    std::atomic<bool>&      stop_flag;
    SparklePreSchedQueue           queue;
    std::unique_ptr<T>      tx{nullptr};
    std::barrier<std::function<void()>>&           stop_latch;

    public:
    SparklePreSchedExecutor(SparklePreSched& spectrum);
    void Finalize();
    void Execute();
    void Schedule();
    void ReExecute();
    void Run();

};

#undef T
#undef V
#undef K

} // namespace spectrum
