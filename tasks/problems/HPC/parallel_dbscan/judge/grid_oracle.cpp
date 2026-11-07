// Private correctness oracle for the large walk workloads (beat spec v3).
// Exact DBSCAN with the same closed distance test and output as
// exact_oracle.cpp, but points are bucketed into a uniform grid whose side is
// slightly larger than epsilon, so every epsilon-neighbour lies in one of the
// 3^d adjacent cells and only those cells are scanned exhaustively. Core
// connectivity is a serial union-find over core-core pairs; every border
// membership is listed. Independent of the DBSCAN project sources.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

struct KeyHash {
  size_t operator()(const std::vector<int64_t> &k) const {
    uint64_t h = 1469598103934665603ULL;
    for (int64_t v : k) { h ^= (uint64_t)v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); h *= 1099511628211ULL; }
    return (size_t)h;
  }
};

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  std::ifstream input(argv[1]);
  std::string header;
  int dim;
  if (!(input >> header >> dim) || header != "dim" || dim < 1 || dim > 16) return 2;
  std::vector<double> points;
  double value;
  while (input >> value) points.push_back(value);
  if (points.empty() || points.size() % dim) return 2;
  const long n = (long)(points.size() / dim);
  const int minpts = std::stoi(argv[3]);
  const double eps = std::stod(argv[2]), radius2 = eps * eps;
  if (!(eps > 0) || minpts < 1) return 2;
  auto near = [&](long i, long j) {
    double distance = 0;
    for (int d = 0; d < dim; ++d) {
      double delta = points[i * dim + d] - points[j * dim + d];
      distance += delta * delta;
      if (distance > radius2) return false;
    }
    return distance <= radius2;
  };
  // Bucket points by floor(x / side). side exceeds eps by a relative 1e-9, so
  // rounding cannot move two points within eps more than one cell apart.
  const double side = eps * (1.0 + 1e-9);
  std::unordered_map<std::vector<int64_t>, long, KeyHash> cellOf;
  std::vector<std::vector<int64_t>> keys;
  std::vector<std::vector<long>> members;
  std::vector<long> pointCell(n);
  std::vector<int64_t> key(dim);
  for (long i = 0; i < n; ++i) {
    for (int d = 0; d < dim; ++d) key[d] = (int64_t)std::floor(points[i * dim + d] / side);
    auto found = cellOf.find(key);
    long c;
    if (found == cellOf.end()) {
      c = (long)keys.size();
      cellOf.emplace(key, c);
      keys.push_back(key);
      members.emplace_back();
    } else c = found->second;
    members[c].push_back(i);
    pointCell[i] = c;
  }
  const long cells = (long)keys.size();
  long offsets = 1;
  for (int d = 0; d < dim; ++d) offsets *= 3;
  std::vector<std::vector<long>> adjacent(cells);
#pragma omp parallel for schedule(dynamic, 64)
  for (long c = 0; c < cells; ++c) {
    std::vector<int64_t> probe(dim);
    for (long o = 0; o < offsets; ++o) {
      long rest = o;
      for (int d = 0; d < dim; ++d) { probe[d] = keys[c][d] + (rest % 3) - 1; rest /= 3; }
      auto found = cellOf.find(probe);
      if (found != cellOf.end()) adjacent[c].push_back(found->second);
    }
  }
  std::vector<char> core(n, 0);
#pragma omp parallel for schedule(dynamic, 256)
  for (long i = 0; i < n; ++i) {
    long count = 0;
    for (long c : adjacent[pointCell[i]]) {
      for (long j : members[c]) {
        count += near(i, j);
        if (count >= minpts) break;
      }
      if (count >= minpts) break;
    }
    core[i] = count >= minpts;
  }
  std::vector<long> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](long x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  // A cell whose core members already share one root keeps that property:
  // unions only merge components. Such a cell is skipped when its root equals
  // the current point's root; this prunes only already-connected pairs.
  std::vector<std::vector<long>> coreMembers(cells);
  for (long c = 0; c < cells; ++c)
    for (long j : members[c]) if (core[j]) coreMembers[c].push_back(j);
  std::vector<char> uniform(cells, 0);
  std::vector<long> checkedAt(cells, -1);
  long unions = 0;
  auto isUniform = [&](long c) {
    if (uniform[c]) return true;
    if (checkedAt[c] == unions) return false;
    checkedAt[c] = unions;
    const std::vector<long> &list = coreMembers[c];
    long first = list.empty() ? -1 : root(list[0]);
    for (long j : list) if (root(j) != first) return false;
    uniform[c] = 1;
    return true;
  };
  for (long i = 0; i < n; ++i) if (core[i]) {
    long ri = root(i);
    for (long c : adjacent[pointCell[i]]) {
      if (coreMembers[c].empty()) continue;
      if (isUniform(c) && root(coreMembers[c][0]) == ri) continue;
      for (long j : coreMembers[c]) {
        if (j >= i) continue;
        long rj = root(j);
        if (ri == rj || !near(i, j)) continue;
        parent[std::max(ri, rj)] = std::min(ri, rj);
        ++unions;
        ri = root(i);
      }
    }
  }
  std::string line;
  std::vector<long> labels;
  for (long i = 0; i < n; ++i) {
    labels.clear();
    if (core[i]) labels.push_back(root(i));
    else for (long c : adjacent[pointCell[i]]) for (long j : members[c])
      if (core[j] && near(i, j)) labels.push_back(root(j));
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    line = std::to_string(i);
    line += core[i] ? " core" : labels.empty() ? " noise" : " border";
    for (long label : labels) { line += ' '; line += std::to_string(label); }
    line += '\n';
    std::cout << line;
  }
}
