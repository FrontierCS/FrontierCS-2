#ifndef _CSR_VLENGTH_CLUSTER_H_
#define _CSR_VLENGTH_CLUSTER_H_
#include "CSR.h"

// Compatibility scaffold only; the unused alternative implementation is absent.
template<class IT, class NT>
class CSR_VlengthCluster {
public:
  CSR_VlengthCluster() = default;
  CSR_VlengthCluster(const CSR<IT, NT> &, const std::map<IT, std::vector<IT>> &);
  CSR_VlengthCluster(const CSR<IT, NT> &, const std::vector<IT> &);
};
#endif
