// The judge's inner products for the products a workload's instance printed: reads a vector file
// (the starter's `_new` format: three 32-bit header words, N, then N rows of `dim` floats) and, on
// standard input, lines `query_id count id...`; prints, one per line, the float32 inner product of
// each id's vector and the query's, by the starter's own `cal_inner_product`, at the precision
// that the harness prints a product.
#include "basis.h"
#include <cstdio>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream in(argv[1], std::ios::binary);
    unsigned int header[3] = {};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || header[1] == 0 || header[2] == 0) return 3;
    const size_t rows = header[1], dim = header[2];
    std::vector<float> values(rows * dim);
    in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
    if (!in) return 3;
    long query, count;
    while (std::scanf("%ld %ld", &query, &count) == 2) {
        if (query < 0 || static_cast<size_t>(query) >= rows) return 4;
        for (long j = 0; j < count; ++j) {
            long id;
            if (std::scanf("%ld", &id) != 1 || id < 0 || static_cast<size_t>(id) >= rows) return 4;
            std::printf("%.9g\n", cal_inner_product(&values[id * dim], &values[query * dim], static_cast<int>(dim)));
        }
    }
    return 0;
}
