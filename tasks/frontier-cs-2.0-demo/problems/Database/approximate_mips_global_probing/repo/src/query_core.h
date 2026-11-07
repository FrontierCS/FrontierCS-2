#pragma once
#include <memory>
// Only query projections enter the solver. Data vectors, IDs, truth, counters,
// and result heaps are owned by the harness.
struct Probe { int table, key; double score; };
struct SearchState {
    int result_count, requested_k, candidates_verified, partition_size;
    double kth_score, next_score, approximation_ratio, maximum_inner_product;
    int hash_tables;
    const char* calibration_path;
};
class QueryCore {
public:
    virtual ~QueryCore() {}
    // projected[t*K+j] = a[t*K+j]^T q / ||q||. K is in [1,20].
    virtual void reset(int L, int K, const double* projected)=0;
    // Emit all L*2^K (table,key) pairs exactly once if not terminated.
    // score is the squared-projection QD, never an arbitrary priority.
    virtual bool next(Probe& out)=0;
    virtual bool stop(const SearchState& state)=0;
};
std::unique_ptr<QueryCore> make_query_core();
