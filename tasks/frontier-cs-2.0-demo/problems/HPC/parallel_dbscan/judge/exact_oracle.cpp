// Private correctness oracle: exhaustive distances, serial union-find, all borders.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  std::ifstream input(argv[1]);
  std::string header;
  int dim;
  if (!(input >> header >> dim) || header != "dim" || dim < 1) return 2;
  std::vector<double> points;
  double value;
  while (input >> value) points.push_back(value);
  if (points.size() % dim) return 2;
  int n = points.size() / dim, minpts = std::stoi(argv[3]);
  double eps = std::stod(argv[2]), radius2 = eps * eps;
  auto near = [&](int i, int j) {
    double distance = 0;
    for (int d = 0; d < dim; ++d) {
      double delta = points[i * dim + d] - points[j * dim + d];
      distance += delta * delta;
      if (distance > radius2) return false;
    }
    return distance <= radius2;
  };
  std::vector<int> core(n), parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](int x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  for (int i = 0; i < n; ++i) {
    int count = 0;
    for (int j = 0; j < n && count < minpts; ++j) count += near(i, j);
    core[i] = count >= minpts;
  }
  for (int i = 0; i < n; ++i) if (core[i]) {
    for (int j = 0; j < i; ++j) if (core[j] && near(i, j)) {
      int a = root(i), b = root(j);
      parent[std::max(a, b)] = std::min(a, b);
    }
  }
  for (int i = 0; i < n; ++i) {
    std::set<int> memberships;
    if (core[i]) memberships.insert(root(i));
    else for (int j = 0; j < n; ++j)
      if (core[j] && near(i, j)) memberships.insert(root(j));
    std::cout << i << ' ' << (core[i] ? "core" : memberships.empty() ? "noise" : "border");
    for (int label : memberships) std::cout << ' ' << label;
    std::cout << '\n';
  }
}
