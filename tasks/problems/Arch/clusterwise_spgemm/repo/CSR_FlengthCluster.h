#ifndef _CSR_FLENGTH_CLUSTER_H_
#define _CSR_FLENGTH_CLUSTER_H_

#include <stdexcept>
#include "CSR.h"

template<class IT, class NT>
class CSR_FlengthCluster {
public:
  CSR_FlengthCluster() : csr_rows(0), rows(0), cols(0), cluster_sz(0), nnzc(0), rowptr(nullptr), colids(nullptr), values(nullptr) {}
  CSR_FlengthCluster(const CSR<IT, NT> &, IT) : CSR_FlengthCluster() {
    throw std::logic_error("CSR_FlengthCluster is not implemented");
  }
  void make_empty() {
    if (rowptr) my_free<IT>(rowptr);
    if (colids) my_free<IT>(colids);
    if (values) my_free<NT>(values);
    rowptr = nullptr; colids = nullptr; values = nullptr;
    csr_rows = rows = cols = cluster_sz = nnzc = 0;
  }
  ~CSR_FlengthCluster() { make_empty(); }
  bool isEmpty() { return nnzc == 0; }

  IT csr_rows, rows, cols, cluster_sz, nnzc;
  IT *rowptr, *colids;
  NT *values;
};

#endif
