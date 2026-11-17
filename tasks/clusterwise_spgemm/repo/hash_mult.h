#ifndef HASH_MULT_H
#define HASH_MULT_H
#include <functional>
#include <string>
#include "CSR.h"

// Compatibility declarations only. The unused row-wise implementation is absent.
template <bool vectorProbing, bool sortOutput, typename IT, typename NT,
          typename MultiplyOperation, typename AddOperation>
void RowSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b,
               CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop,
               std::string out_nnz_freq_outfile, bool first_pass = false);

template <bool sortOutput, typename IT, typename NT,
          typename MultiplyOperation, typename AddOperation>
void RowSpGEMM(const CSR<IT, NT> &a, const CSR<IT, NT> &b,
               CSR<IT, NT> &c, MultiplyOperation multop, AddOperation addop);
#endif
