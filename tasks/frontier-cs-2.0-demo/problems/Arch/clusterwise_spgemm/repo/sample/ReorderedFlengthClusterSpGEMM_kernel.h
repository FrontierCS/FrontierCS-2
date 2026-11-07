#ifndef REORDERED_FLENGTH_CLUSTER_SPGEMM_KERNEL_H
#define REORDERED_FLENGTH_CLUSTER_SPGEMM_KERNEL_H

// Fixed boundary between the two translation units of the retained program.
//
// ReorderedFlengthClusterSpGEMM.cpp (main, argument handling, matrix loading,
// reordering, CSR construction, flop count, output) includes no editable
// header. ReorderedFlengthClusterSpGEMM_kernel.cpp includes the editable
// headers and defines frontier_clustered_spgemm, which builds the clustered
// representation of A, runs HashSpGEMMCluster and hands the clustered product
// to the caller's consumer while it is still alive. The two files are compiled
// separately and linked; the kernel object may not define code that the main
// object defines.

#include <cstdint>

template <class IT, class NT>
class CSR;

// Fields of the retained CSR_FlengthCluster<int64_t, double> product, with the
// meanings given in the task statement, plus the average multiplication time.
struct FrontierClusterProduct {
  int64_t csr_rows, cols, rows, cluster_sz, nnzc;
  const int64_t *rowptr;
  const int64_t *colids;
  const double *values;
  double ave_msec;
};

typedef int (*FrontierClusterConsumer)(const FrontierClusterProduct *product, void *context);

// Builds CSR_FlengthCluster(A_csr, cluster_size), runs `warmups` untimed and
// then `iters` timed HashSpGEMMCluster calls, calls `consume` exactly once on
// the final product and returns its status.
int frontier_clustered_spgemm(CSR<int64_t, double> &A_csr, CSR<int64_t, double> &B_csr,
                              int64_t cluster_size, int warmups, int iters,
                              FrontierClusterConsumer consume, void *context);

#endif
