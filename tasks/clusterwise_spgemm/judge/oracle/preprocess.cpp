#include "ordered_oracle.hpp"
#include <filesystem>
#include <iostream>
using independent::Index;
using independent::require;

static std::ofstream output(const std::filesystem::path& p) {
    std::ofstream out(p, std::ios::binary);
    require(out.good(), "cannot open expected output");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    return out;
}
template<class T> static void append(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
int main(int argc, char** argv) {
    try {
        require(argc == 9, "usage: preprocess matrix order output_dir rows input_nnz output_nnz operations capture_bytes");
        const uint16_t endian = 1;
        require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "little-endian host required");
        independent::Limits limits{std::stoll(argv[4]),std::stoll(argv[5]),std::stoll(argv[6]),std::stoll(argv[7]),std::stoll(argv[8])};
        auto b = independent::read_matrix(argv[1], limits);
        auto order = independent::read_order(argv[2], b.rows);
        Index operations = independent::operation_count(b, limits), nnz = 0, symbolic = 0;
        Index groups = order.size()/8;
        std::filesystem::path root(argv[3]);
        require(std::filesystem::is_directory(root) && std::filesystem::is_empty(root), "expected directory must be empty");
        auto pointers = output(root / "rowptr.i64le"), columns = output(root / "columns.i64le");
        auto values = output(root / "values.f64le");
        append(pointers, nnz);
        independent::GroupOracle oracle(b);
        for (Index g = 0; g < groups; ++g) {
            oracle.evaluate(b, order, g);
            Index size = oracle.columns.size(); symbolic += size;
            require(48 + 8*(groups+1+9*symbolic) <= limits.capture_bytes, "capture limit");
            for (Index lane = 0; lane < 8; ++lane) {
                for (Index c : oracle.columns) {
                    double value = oracle.values[c][lane];
                    require(std::isfinite(value), "nonfinite oracle output");
                    if (value != 0.0) {
                        require(nnz < limits.output_nnz, "exact numeric nnz limit");
                        append(columns, c); append(values, value); ++nnz;
                    }
                }
                append(pointers,nnz);
            }
        }
        pointers.close(); columns.close(); values.close();
        auto metadata = output(root / "metadata.json");
        metadata << "{\"rows\":" << order.size() << ",\"original_rows\":" << b.rows
                 << ",\"groups\":" << groups << ",\"width\":8,\"numeric_nnz\":" << nnz
                 << ",\"symbolic_columns\":" << symbolic << ",\"scalar_operations\":" << operations
                 << ",\"capture_bytes\":" << 48+8*(groups+1+9*symbolic) << "}\n";
        metadata.close();
    } catch (const std::exception& error) {
        std::cerr << "ORACLE_FAILURE: " << error.what() << '\n';
        return 1;
    }
}
