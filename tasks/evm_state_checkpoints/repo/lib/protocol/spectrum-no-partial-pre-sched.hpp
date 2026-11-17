#include <spectrum/workload/abstraction.hpp>
#include <spectrum/common/lock-util.hpp>
#include <spectrum/protocol/abstraction.hpp>
#include <spectrum/transaction/evm-hash.hpp>
#include <list>
#include <atomic>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <thread>
#include <barrier>

namespace spectrum {

// some shorthands to prevent prohibitively long names
#define K std::tuple<evmc::address, evmc::bytes32>
#define V SpectrumNoPartialPreSchedVersionList
#define T SpectrumNoPartialPreSchedTransaction

using namespace std::chrono;

struct SpectrumNoPartialPreSchedPutTuple {
    K               key;
    evmc::bytes32   value;
    bool            is_committed;
};

struct SpectrumNoPartialPreSchedGetTuple {
    K               key;
    evmc::bytes32   value;
    size_t          version;
    size_t          tuples_put_len;
    size_t          checkpoint_id;
};

struct SpectrumNoPartialPreSchedTransaction: public Transaction {
    size_t      id;
    SpinLock            rerun_keys_mu;
    std::vector<K>      rerun_keys;
    std::atomic<bool>   berun_flag{false};
    std::unordered_map<K, size_t, KeyHasher>    should_wait;
    time_point<steady_clock>                    start_time;
    std::vector<SpectrumNoPartialPreSchedGetTuple>       tuples_get{};
    std::vector<SpectrumNoPartialPreSchedPutTuple>       tuples_put{};
    SpectrumNoPartialPreSchedTransaction(Transaction&& inner, size_t id);
    bool    HasWAR();
    void    SetWAR(const K& key, size_t writer_id, bool pre_schedule);
    size_t  ShouldWait(const K& key);
};

struct SpectrumNoPartialPreSchedEntry {
    evmc::bytes32   value;
    size_t          version;
    // we store raw pointers here because when a transaction is destructed, it always removes itself from table. 
    std::unordered_set<T*>  readers;
};

struct SpectrumNoPartialPreSchedVersionList {
    T*          tx = nullptr;
    std::list<SpectrumNoPartialPreSchedEntry> entries;
    // readers that read default value
    std::unordered_set<T*>  readers_default;
};

struct SpectrumNoPartialPreSchedLockTable: private Table<K, V, KeyHasher> {

    SpectrumNoPartialPreSchedLockTable(size_t partitions);
    void Get(T* tx, const K& k);
    void Put(T* tx, const K& k);
    void ClearPut(T* tx, const K& k);
    void ClearGet(T* tx, const K& k);

};

struct SpectrumNoPartialPreSchedTable: private Table<K, V, KeyHasher> {

    SpectrumNoPartialPreSchedTable(size_t partitions);
    void Get(T* tx, const K& k, evmc::bytes32& v, size_t& version);
    void Put(T* tx, const K& k, const evmc::bytes32& v);
    void RegretGet(T* tx, const K& k, size_t version);
    void RegretPut(T* tx, const K& k);
    void ClearGet(T* tx, const K& k, size_t version);
    void ClearPut(T* tx, const K& k);

};

using SpectrumNoPartialPreSchedQueue = LockPriorityQueue<T>;
class SpectrumNoPartialPreSchedExecutor;

class SpectrumNoPartialPreSched: public Protocol {

    private:
    size_t                      num_executors;
    Workload&                   workload;
    SpectrumNoPartialPreSchedTable       table;
    SpectrumNoPartialPreSchedLockTable   lock_table;
    Statistics&                 statistics;
    std::atomic<size_t>         last_executed{1};
    std::atomic<size_t>         last_finalized{0};
    std::atomic<size_t>         last_scheduled{0};
    std::atomic<size_t>         last_committed{0};
    std::atomic<bool>           stop_flag{false};
    std::vector<std::thread>    executors{};
    std::barrier<std::function<void()>>  stop_latch;
    friend class SpectrumNoPartialPreSchedExecutor;

    public:
    SpectrumNoPartialPreSched(Workload& workload, Statistics& statistics, size_t num_executors, size_t table_partitions, EVMType evm_type);
    void Start() override;
    void Stop() override;

};

class SpectrumNoPartialPreSchedExecutor {

    using TP = std::unique_ptr<T>;

    private:
    Workload&                   workload;
    SpectrumNoPartialPreSchedTable&      table;
    SpectrumNoPartialPreSchedLockTable&  lock_table;
    Statistics&                 statistics;
    std::atomic<size_t>&        last_executed;
    std::atomic<size_t>&        last_finalized;
    std::atomic<size_t>&        last_scheduled;
    std::atomic<size_t>&        last_committed;
    std::atomic<bool>&          stop_flag;
    SpectrumNoPartialPreSchedQueue       queue;
    std::list<TP>               idle_queue;
    std::unique_ptr<T>          tx{nullptr};
    std::barrier<std::function<void()>>& stop_latch;

    public:
    SpectrumNoPartialPreSchedExecutor(SpectrumNoPartialPreSched& spectrum);
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
