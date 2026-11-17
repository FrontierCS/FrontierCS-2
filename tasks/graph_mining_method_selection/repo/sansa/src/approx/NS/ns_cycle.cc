#include <iostream>
#include "graph.h"
#include "sample.hh"

void sample_cycle(Graph &g, int k, eidType num_samples, uint64_t &total);
void sample_4cycle_vertex_induced(Graph &g, eidType num_samples, uint64_t &total);

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

  Timer t;
  t.Start();
  uint64_t total = 0;
  //sample_cycle(g, k, num_samples, total);
  sample_4cycle_vertex_induced(g, num_samples, total);
  t.Stop();
 
  std::cout << "runtime = " << t.Seconds() << " sec\n";
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
}

void sample_cycle(Graph &g, int k, eidType num_samples, uint64_t &total) {
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";

  auto m = g.init_edgelist(true);
  std::cout << "sample space: " << m << "\n";
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads;
 
  std::random_device rd;
  rd_engine gens[num_threads];
  for(int t = 0; t < num_threads; t++)
    gens[t] = rd_engine(rd());

  Timer t;
  t.Start();
  #pragma omp parallel reduction(+ : counter)
  {
  //std::random_device rd;
  //rd_engine gen(rd());
  for (eidType i = 0; i < num_samples_per_thread; i++) {
    auto &gen = gens[omp_get_thread_num()];
    auto eid = edge_dist(gen);
    double scale = 1.;
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VertexSet vs;
    vs.add(v0);
    vs.add(v1);
    sort_vertexset(vs);
    vidType v = v1;
    vidType c;
    for (int j = 2; j < k; j++) {
      if (j == k-1) {
        VertexSet candidate_set;
        difference_set(candidate_set, g.N(v), vs, v1);
        scale *= intersection_num(candidate_set, g.N(v0));
        break;
      } else {
        VertexSet candidate_set;
        difference_set(candidate_set, g.N(v), vs, v0);
        c = candidate_set.size();
        if (c < 1) { scale = 0; break; }
        auto id = random_select_single(0, c-1, gen);
        v = candidate_set[id];
      }
      if (c == 0) {
        scale = 0;
        break;
      }
      vs.add(v);
      sort_vertexset(vs);
      scale *= c;
    }
    counter += scale;
  }
  }
  // scale down by number of samples
  total = counter * m / num_samples;
  t.Stop();
}

void sample_4cycle_vertex_induced(Graph &g, eidType num_samples, uint64_t &total) {
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";

  auto m = g.init_edgelist(true);
  std::cout << "sample space: " << m << "\n";
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads;
 
  std::random_device rd;
  rd_engine gens[num_threads];
  for(int t = 0; t < num_threads; t++)
    gens[t] = rd_engine(rd());

  Timer t;
  t.Start();
  #pragma omp parallel reduction(+ : counter)
  {
  for (eidType i = 0; i < num_samples_per_thread; i++) {
    auto &gen = gens[omp_get_thread_num()];
    auto eid = edge_dist(gen);
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto d0 = g.get_degree(v0);
    auto d1 = g.get_degree(v1);
    if (d0 < 2 || d1 < 2) continue;

    auto y0n1 = difference_set(g.N(v0), g.N(v1), v1);
    auto c0 = y0n1.size();
    if (c0 == 0) continue;
    auto idx0 = random_select_single(0, c0-1, gen);
    auto v2 = y0n1[idx0];

    auto y1y2 = intersection_set(g.N(v1), g.N(v2), v0);
    auto c1 = difference_num(y1y2, g.N(v0), v0);
    if (c1 > 0) counter += c0 * c1;
  }
  }
  total = counter * m / num_samples;
  t.Stop();
}
