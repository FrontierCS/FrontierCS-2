#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace independent {
using Index = int64_t;
constexpr Index width = 8;
constexpr double epsilon = static_cast<double>(0.000001f);
struct Limits {
    Index rows = 4000000, input_nnz = 40000000, output_nnz = 80000000;
    Index operations = 2000000000, capture_bytes = 7516192768;
};
struct Matrix {
    Index rows = 0;
    std::vector<Index> ptr, col;
    std::vector<double> val;
};
inline void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
inline bool data_line(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        auto p = line.find_first_not_of(" \t\r");
        if (p != std::string::npos && line[p] != '%') return true;
    }
    return false;
}
inline Matrix read_matrix(const std::string& path, const Limits& limit) {
    std::ifstream in(path);
    require(in.good(), "cannot open matrix");
    std::string line, magic, object, format, field, symmetry, extra;
    std::getline(in, line);
    std::istringstream banner(line);
    require(bool(banner >> magic >> object >> format >> field >> symmetry) && !(banner >> extra), "invalid banner");
    require(magic == "%%MatrixMarket" && object == "matrix" && format == "coordinate", "unsupported matrix format");
    require(field == "real" || field == "integer" || field == "pattern", "unsupported field");
    require(symmetry == "general" || symmetry == "symmetric", "unsupported symmetry");
    require(data_line(in, line), "missing dimensions");
    Index n, m, count;
    std::istringstream dimensions(line);
    require(bool(dimensions >> n >> m >> count) && !(dimensions >> extra), "invalid dimensions");
    require(n > 0 && n == m && n <= limit.rows, "dimension limit");
    require(count > 0 && count <= limit.input_nnz, "input nnz limit");
    using Entry = std::tuple<Index, Index, double>;
    std::vector<Entry> entries;
    entries.reserve(static_cast<size_t>(std::min(limit.input_nnz, count * (symmetry == "symmetric" ? 2 : 1))));
    for (Index i = 0; i < count; ++i) {
        require(data_line(in, line), "truncated matrix");
        std::istringstream row(line);
        Index r, c;
        double v = 1.0;
        require(bool(row >> r >> c), "invalid coordinate");
        if (field != "pattern") {
            require(bool(row >> v) && std::isfinite(v), "invalid numeric value");
            v += 1.0;
        }
        require(!(row >> extra) && std::isfinite(v), "trailing token or nonfinite transformed value");
        require(r > 0 && c > 0 && r <= n && c <= n, "coordinate range");
        entries.emplace_back(r-1, c-1, v);
        if (symmetry == "symmetric" && r != c) entries.emplace_back(c-1, r-1, v);
        require(entries.size() <= static_cast<size_t>(limit.input_nnz), "expanded input nnz limit");
    }
    require(!data_line(in, line), "extra matrix entries");
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b));
    });
    Matrix matrix;
    matrix.rows = n;
    matrix.ptr.resize(n+1, 0);
    matrix.col.reserve(entries.size());
    matrix.val.reserve(entries.size());
    Index last_r = -1, last_c = -1;
    for (const auto& [r,c,v] : entries) {
        require(r != last_r || c != last_c, "duplicate expanded coordinate");
        ++matrix.ptr[r+1];
        matrix.col.push_back(c);
        matrix.val.push_back(v);
        last_r = r; last_c = c;
    }
    std::partial_sum(matrix.ptr.begin(), matrix.ptr.end(), matrix.ptr.begin());
    return matrix;
}
inline std::vector<Index> read_order(const std::string& path, Index n) {
    std::ifstream in(path);
    require(in.good(), "cannot open order");
    std::vector<Index> order;
    std::vector<bool> seen(n, false);
    Index r;
    while (true) {
        in >> std::ws;
        if (in.eof()) break;
        require(bool(in >> r), "malformed permutation token");
        require(r >= 0 && r < n && !seen[r], "invalid permutation");
        seen[r] = true;
        order.push_back(r);
    }
    require(in.eof() && order.size() == static_cast<size_t>(n), "incomplete permutation");
    while (order.size() % width) order.push_back(n++);
    return order;
}
inline Index operation_count(const Matrix& b, const Limits& limits) {
    Index paths = 0;
    for (Index k : b.col) {
        paths += b.ptr[k+1] - b.ptr[k];
        require(paths <= limits.operations / 2, "raw operation limit");
    }
    return paths * 2;
}

// Dense addressing is independent of the reference's hash tables and bins.
// Only touched columns are initialized; each group is released before the next.
class GroupOracle {
    const Matrix& b;
    std::vector<Index> generation;
    Index serial = 0;
public:
    std::vector<std::array<double, width>> values;
    std::vector<Index> columns;
    explicit GroupOracle(const Matrix& matrix)
        : b(matrix), generation(matrix.rows, 0), values(matrix.rows) {}
    void evaluate(const Matrix& a, const std::vector<Index>& order, Index group) {
        ++serial;
        columns.clear();
        std::array<Index, width> cursor{}, end{};
        for (Index lane = 0; lane < width; ++lane) {
            Index row = order[group*width+lane];
            if (row < a.rows) { cursor[lane] = a.ptr[row]; end[lane] = a.ptr[row+1]; }
        }
        while (true) {
            Index k = std::numeric_limits<Index>::max();
            for (Index lane = 0; lane < width; ++lane)
                if (cursor[lane] != end[lane]) k = std::min(k, a.col[cursor[lane]]);
            if (k == std::numeric_limits<Index>::max()) break;
            std::array<double, width> av{};
            for (Index lane = 0; lane < width; ++lane)
                if (cursor[lane] != end[lane] && a.col[cursor[lane]] == k)
                    av[lane] = a.val[cursor[lane]++];
            for (Index p = b.ptr[k]; p < b.ptr[k+1]; ++p) {
                Index c = b.col[p];
                bool first = generation[c] != serial;
                if (first) { generation[c] = serial; columns.push_back(c); }
                for (Index lane = 0; lane < width; ++lane) {
                    if (first) values[c][lane] = av[lane] * b.val[p];
                    else if (std::abs(av[lane]) >= epsilon) {
                        double product = av[lane] * b.val[p];
                        values[c][lane] = product + values[c][lane];
                    }
                }
            }
        }
        std::sort(columns.begin(), columns.end());
    }
};
}
