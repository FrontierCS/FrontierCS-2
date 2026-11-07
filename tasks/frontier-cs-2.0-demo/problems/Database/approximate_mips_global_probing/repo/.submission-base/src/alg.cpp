#include "query_core.h"
#include <vector>
#include <cmath>
// Runnable exhaustive starter: numeric bucket order, no adaptive termination.
// Replace this with an efficient query-local implementation.
class ExhaustiveCore : public QueryCore {
    int l=0, k=0, cursor=0;
    std::vector<double> projection;
public:
    void reset(int L,int K,const double* p) override {
        l=L; k=K; cursor=0; projection.assign(p,p+L*K);
    }
    bool next(Probe& out) override {
        if(cursor==l*(1<<k)) return false;
        out.table=cursor/(1<<k); out.key=cursor%(1<<k); out.score=0;
        for(int j=0;j<k;++j) {
            double z=projection[out.table*k+j];
            if(((out.key>>(k-1-j))&1)!=(z>0)) out.score+=z*z;
        }
        ++cursor; return true;
    }
    bool stop(const SearchState&) override { return false; }
};
std::unique_ptr<QueryCore> make_query_core() {
    return std::unique_ptr<QueryCore>(new ExhaustiveCore());
}
