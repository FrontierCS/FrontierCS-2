#include <iostream>
#include "graph.h"
#include "sample.hh"
#define VERTEX_INDUCED
int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <graph> <k> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();
  int k = atoi(argv[2]);
  assert(k > 3);
  eidType num_samples = atoi(argv[3]);
  std::cout << "num_samples: " << num_samples << "\n";

  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";

  auto num_samples_per_thread = num_samples / num_threads;
 
  std::random_device rd;
  rd_engine gens[num_threads];

  for(int t = 0; t < num_threads; t++) {
    gens[t] = rd_engine(rd());
  }

#ifdef VERTEX_INDUCED
  auto m = g.init_edgelist();
#else
  auto m = g.V();
#endif
  std::cout << "sample space: " << m << "\n";
  std::uniform_int_distribution<eidType> sample_dist(0, m-1);

  Timer t;
  t.Start();
  #pragma omp parallel reduction(+ : counter)
  {
  //std::uniform_int_distribution<eidType> sample_dist(0, m-1);
  //std::random_device rd;
  //rd_engine gen(rd());

#ifdef VERTEX_INDUCED
  // vertex-induced
  for (eidType i = 0; i < num_samples_per_thread; i++) {
    auto &gen = gens[omp_get_thread_num()];
    auto eid = sample_dist(gen);
    auto v0 = g.get_src(eid);
    auto d0 = g.get_degree(v0);
    if (d0 < 3) continue;
    auto v1 = g.get_dst(eid);
    auto y0n1 = difference_set(g.N(v0), g.N(v1), v1);
    auto c0 = y0n1.size();
    if (c0 < 2) continue;
    auto idx0 = random_select_single<vidType>(0, c0-1, gen);
    auto v2 = y0n1[idx0];
    auto c1 = difference_num(y0n1, g.N(v2), v2);
    if (c1 > 0) counter += c0 * c1;
  }
#else
  // edge-induced
  for (eidType i = 0; i < num_samples_per_thread; i++) {
    auto &gen = gens[omp_get_thread_num()];
    auto v0 = sample_dist(gen);
    auto n = g.get_degree(v0);
    if (n > 2) counter += (n-2)*(n-1)*n/6;
  }
#endif
  }
  // scale down by number of samples
  uint64_t total = counter * m / num_samples;
  t.Stop();
  std::cout << "runtime = " << t.Seconds() << " sec\n";
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
}

