#ifndef HASH_MULT_FLENGTHCLUSTER_H
#define HASH_MULT_FLENGTHCLUSTER_H

#include <stdexcept>
#include "CSR.h"
#include "CSR_FlengthCluster.h"

template<bool sortOutput, typename IT, typename NT,
         typename MultiplyOperation, typename AddOperation>
void HashSpGEMMCluster(const CSR_FlengthCluster<IT, NT> &,
                       const CSR<IT, NT> &,
                       CSR_FlengthCluster<IT, NT> &,
                       MultiplyOperation, AddOperation) {
  throw std::logic_error("HashSpGEMMCluster is not implemented");
}

#endif
