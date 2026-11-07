#ifndef CLUSTER_UTILITY_H
#define CLUSTER_UTILITY_H
#include "CSR.h"
#include "CSR_FlengthCluster.h"

// Compatibility declaration only; the conversion implementation is absent.
template<class IT, class NT>
void csr_flength_cluster2csr(
    CSR<IT, NT> &csr,
    const CSR_FlengthCluster<IT, NT> &csr_flength_cluster);
#endif
