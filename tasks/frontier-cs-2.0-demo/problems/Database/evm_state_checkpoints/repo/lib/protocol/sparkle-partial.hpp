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

/*
    This is an implementation of "Sparkle: Speculative Deterministic Concurrency Control for Partially Replicated Transactional Data Stores" (Zhongmiao Li, Peter Van Roy and Paolo Romano). 
    (with partial rollback proposed by Spectrum)
 */
namespace spectrum {

// some shorthands to prevent prohibitively long names
#define K std::tuple<evmc::address, evmc::bytes32>
#define V SparklePartialVersionList
#define T SparklePartialTransaction

using namespace std::chrono;

struct SparklePartialPutTuple {
    K               key;
    evmc::bytes32   value;
    bool            is_committed;
};

struct SparklePartialGetTuple {
    K               key;
    evmc::bytes32   value;
    size_t          version;
    size_t          tuples_put_len;
    size_t          checkpoint_id;
};

struct SparklePartialTransaction: public Transaction {
    size_t      id;
    size_t      should_wait{0};
    SpinLock            rerun_keys_mu;
    std::vector<K>      rerun_keys;
    std::atomic<bool>   berun_flag{false};
    time_point<steady_clock>            start_time;
    std::vector<SparklePartialGetTuple>       tuples_get{};
    std::vector<SparklePartialPutTuple>       tuples_put{};
    SparklePartialTransaction(Transaction&& inner, size_t id);
    bool HasWAR();
    void SetWAR(const K& key, size_t cause_id);
};

struct SparklePartialEntry {
    evmc::bytes32   value;
    size_t          version;
    // we store raw pointers here because when a transaction is destructed, it always removes itself from table. 
    std::unordered_set<T*>  readers;
};

struct SparklePartialVersionList {
    T*          tx = nullptr;
    std::list<SparklePartialEntry> entries;
    // readers that read default value
    std::unordered_set<T*>  readers_default;
};

struct SparklePartialTable: private Table<K, V, KeyHasher> {

    SparklePartialTable(size_t partitions);
    void Get(T* tx, const K& k, evmc::bytes32& v, size_t& version);
    void Put(T* tx, const K& k, const evmc::bytes32& v);
    void RegretGet(T* tx, const K& k, size_t version);
    void RegretPut(T* tx, const K& k);
    void ClearGet(T* tx, const K& k, size_t version);
    void ClearPut(T* tx, const K& k);
    bool Lock(T* tx, const K& k);
    bool Unlock(T* tx, const K& k);

};

using SparklePartialQueue = LockPriorityQueue<T>;
class SparklePartialExecutor;

class SparklePartial: public Protocol {

    private:
    size_t              num_executors;
    Workload&           workload;
    SparklePartialTable       table;
    Statistics&         statistics;
    std::atomic<size_t> last_executed{1};
    std::atomic<size_t> last_finalized{0};
    std::atomic<bool>   stop_flag{false};
    std::vector<std::thread>    executors{};
    std::barrier<std::function<void()>>            stop_latch;
    friend class SparklePartialExecutor;

    public:
    SparklePartial(Workload& workload, Statistics& statistics, size_t num_executors, size_t table_partitions, EVMType evm_type);
    void Start() override;
    void Stop() override;

};

class SparklePartialExecutor {

    private:
    Workload&               workload;
    SparklePartialTable&          table;
    Statistics&             statistics;
    std::atomic<size_t>&    last_executed;
    std::atomic<size_t>&    last_finalized;
    std::atomic<bool>&      stop_flag;
    SparklePartialQueue           queue;
    std::unique_ptr<T>      tx{nullptr};
    std::barrier<std::function<void()>>&           stop_latch;

    public:
    SparklePartialExecutor(SparklePartial& spectrum);
    void Finalize();
    void Generate();
    void ReExecute();
    void Run();

};

#undef T
#undef V
#undef K

} // namespace spectrum
