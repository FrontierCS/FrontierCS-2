#include "graph.h"
#include "sample.hh"
#define VERTEX_INDUCED

void sample_diamond(Graph &g, eidType num_samples, uint64_t &total, uint64_t m);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <graph> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();
  int64_t num_samples = atoi(argv[2]);

  Timer t;
  auto m = g.init_edgelist(true);

  t.Start();
  uint64_t total = 0;
  sample_diamond(g, num_samples, total, m);
  t.Stop();
  std::cout << "Runtime = " << t.Seconds() << " sec\n";
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
}

void sample_diamond(Graph &g, eidType num_samples, uint64_t &total, uint64_t m) {
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
 
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 

  std::random_device rd;
  rd_engine gens[num_threads];

  for(int t = 0; t < num_threads; t++) {
    gens[t] = rd_engine(rd());
  }

  #pragma omp parallel reduction(+ : counter)
  {
  //std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  //std::random_device rd;
  //rd_engine gen(rd());
#ifdef VERTEX_INDUCED
  //vertex-induced
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    //auto eid = edge_dist(gen);
    auto &gen = gens[omp_get_thread_num()];
    auto eid = edge_dist(gen);
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto d0 = g.get_degree(v0);
    if (d0 < 3) continue;
    auto d1 = g.get_degree(v1);
    if (d1 < 3) continue;
    auto y0y1 = g.N(v0) & g.N(v1);
    auto c0 = y0y1.size();
    if (c0 < 2) continue;
    auto idx0 = random_select_single<vidType>(0, c0-1, gen);
    auto v2 = y0y1[idx0];
    auto c1 = difference_num(y0y1, g.N(v2), v2);
    if (c1 < 1) continue;
    counter += double(c0) * c1;
  }
#else
  // edge-induced
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    //auto eid = edge_dist(gen);
    auto eid = edge_dist(gens[omp_get_thread_num()]);
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto d0 = g.get_degree(v0);
    if (d0 < 3) continue;
    auto d1 = g.get_degree(v1);
    if (d1 < 3) continue;
    auto n = intersection_num(g.N(v0), g.N(v1));
    if (n > 1) counter += n * (n-1) / 2;
  }
#endif
  }
  total = counter * m / num_samples;
}
