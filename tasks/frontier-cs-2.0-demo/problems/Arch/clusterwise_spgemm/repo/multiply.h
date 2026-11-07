#include "CSR.h"
#include <omp.h>
#include <algorithm>

template <typename IT, typename NT>
long long int get_flop(const CSR<IT,NT> & A, const CSR<IT,NT> & B)
{
    long long int flops = 0; // total flops (multiplication) needed to generate C
    long long int tflops=0; //thread private flops

    for (IT i=0; i < A.rows; ++i) {       // for all rows of A
        long long int locmax = 0;
        for (IT j=A.rowptr[i]; j < A.rowptr[i + 1]; ++j) { // For all the nonzeros of the ith column
            long long int inner = A.colids[j]; // get the row id of B (or column id of A)
            long long int npins = B.rowptr[inner + 1] - B.rowptr[inner]; // get the number of nonzeros in A's corresponding column
            locmax += npins;
        }
        tflops += locmax;
    }
    flops += tflops;
    return (flops * 2);
}
