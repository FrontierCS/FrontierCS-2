#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <random>

#include <filesystem>
namespace fs = std::filesystem;

#include "../utility.h"
#include "../cluster_utility.h"
#include "../CSR.h"
#include "../CSR_FlengthCluster.h"
#include "../multiply.h"

#include "../hash_mult.h"
#include "../hash_mult_flengthcluster.h"
#include "sample_common.hpp"
#include "ReorderedFlengthClusterSpGEMM_kernel.h"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int64_t

// The clustered section of the retained program: representation construction
// and HashSpGEMMCluster, unchanged except that the product is handed to the
// caller (ReorderedFlengthClusterSpGEMM.cpp) through FrontierClusterProduct.
int frontier_clustered_spgemm(CSR<INDEXTYPE, VALUETYPE> &A_csr, CSR<INDEXTYPE, VALUETYPE> &B_csr,
                              INDEXTYPE cluster_size, int warmups, int iters,
                              FrontierClusterConsumer consume, void *context) {
  const bool sortOutput = false;
  double start, end, msec, ave_msec;

  // create A_csr_flength_cluster from A_csr
  CSR_FlengthCluster<INDEXTYPE, VALUETYPE> A_csr_flength_cluster(A_csr, cluster_size);

  CSR_FlengthCluster<INDEXTYPE, VALUETYPE> C_csr_flength_cluster;

  /* First execution is excluded from evaluation */
  for (int w = 0; w < warmups; ++w) {
    HashSpGEMMCluster<sortOutput>(A_csr_flength_cluster, B_csr, C_csr_flength_cluster, multiplies<VALUETYPE>(), plus<VALUETYPE>());
    C_csr_flength_cluster.make_empty();
  }

  ave_msec = 0;
  for (int i = 0; i < iters; ++i) {
    start = omp_get_wtime();
    HashSpGEMMCluster<sortOutput>(A_csr_flength_cluster, B_csr, C_csr_flength_cluster, multiplies<VALUETYPE>(), plus<VALUETYPE>());
    end = omp_get_wtime();
    msec = (end - start) * 1000;
    ave_msec += msec;
    if (i < iters - 1) {
      C_csr_flength_cluster.make_empty();
    }
  }
  ave_msec /= iters;

  const FrontierClusterProduct product = {
      C_csr_flength_cluster.csr_rows, C_csr_flength_cluster.cols, C_csr_flength_cluster.rows,
      C_csr_flength_cluster.cluster_sz, C_csr_flength_cluster.nnzc,
      C_csr_flength_cluster.rowptr, C_csr_flength_cluster.colids, C_csr_flength_cluster.values,
      ave_msec};
  const int status = consume(&product, context);
  C_csr_flength_cluster.make_empty();
  return status;
}
