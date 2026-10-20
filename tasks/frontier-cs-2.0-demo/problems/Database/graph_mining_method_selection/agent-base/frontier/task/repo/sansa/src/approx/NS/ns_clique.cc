#include "graph.h"
#include "sample.hh"
//#define PRUNING

void sample_clique_base(Graph &g, int k, eidType num_samples, uint64_t &total);
void sample_clique_prune_mo(Graph &g, int k, eidType num_samples, uint64_t &total);
void sample_clique_prune_mo_sb(Graph &g, int k, eidType num_samples, uint64_t &total);
void sample_4clique(Graph &g, eidType num_samples, uint64_t &counter);

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <graph> <k> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();

  int k = atoi(argv[2]);
  assert(k > 2);
  int64_t num_samples = atoi(argv[3]);

  Timer t;
  t.Start();
  uint64_t total = 0;
#ifdef PRUNING
  sample_clique_prune_mo_sb(g, k, num_samples, total);
#else
  sample_clique_base(g, k, num_samples, total);
#endif
  t.Stop();
  std::cout << "Runtime = " << t.Seconds() << " sec\n";
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
}

bool is_clique(Graph &g, std::vector<vidType> vs) {
  for (size_t i = 2; i < vs.size(); i++) {
    auto v = vs[i];
    for (size_t j = 0; j < i; j++) {
      auto u = vs[j];
      if (!g.is_connected(v, u)) return false;
    }
  }
  return true;
}

// no pruning; same as ASAP
void sample_clique_base(Graph &g, int k, eidType num_samples, uint64_t &total) {
  std::cout << "running sample_clique base version\n";
  auto m = g.E();
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 

  Timer t;
  t.Start();
  g.init_simple_edgelist();
  t.Stop();
  std::cout << "Preprocessing time = " << t.Seconds() << " sec\n";

  t.Start();
  #pragma omp parallel reduction(+ : counter)
  {
  std::random_device rd;
  rd_engine my_rng(rd());
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    auto eid = edge_dist(my_rng);
    std::vector<vidType> sampled_vertices;
    auto scale = connected_subgraph_sampler(g, k, eid, my_rng, sampled_vertices);
    if (scale == 0) continue;
    //assert(sampled_vertices.size() == k);
    //std::set<vidType> s(sampled_vertices.begin(), sampled_vertices.end());
    //assert(s.size() == k);
    if (is_clique(g, sampled_vertices))
      counter += double(scale);
  }
  }
  // scale down by number of samples
  total = uint64_t(counter * m / num_samples);

  // symmetry breaking
  int multiplicity = 2;
  assert(k >= 3);
  for (int i = 3; i <= k; i++) {
    multiplicity *= i;
  }
  total = total / multiplicity;
  std::cout << "multiplicity = " << multiplicity << "\n";
  std::cout << "Sampling time = " << t.Seconds() << " sec\n";
}

// use matching order to prune
void sample_clique_prune_mo(Graph &g, int k, eidType num_samples, uint64_t &total) {
  auto m = g.E();
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 

  Timer t;
  t.Start();
  g.init_simple_edgelist();
  t.Stop();
  std::cout << "Preprocessing time = " << t.Seconds() << " sec\n";

  t.Start();
  #pragma omp parallel reduction(+ : counter)
  {
  std::random_device rd;
  rd_engine gen(rd());
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    auto eid = edge_dist(gen);
    double scale = 1.;
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VertexSet temp[2];
    intersection(g.N(v0), g.N(v1), temp[0]);
    vidType c = temp[0].size();
    if (c == 0) continue;
    if (k == 3) {
      counter += scale * c;
      continue;
    }
    auto idx0 = random_select_single<vidType>(0, c-1, gen);
    vidType v = temp[0][idx0];
    scale *= c;
    for (int j = 2; j < k-1; j++) {
      temp[(j+1)%2].clear();
      if (j == k - 2) {
        c = intersection_num(g.N(v), temp[j%2]);
      } else {
        intersection(g.N(v), temp[j%2], temp[(j+1)%2]);
        c = temp[(j+1)%2].size();
        if (c == 0) { scale = 0; break; }
        auto id = random_select_single<vidType>(0, c-1, gen);
        v = temp[(j+1)%2][id];
      }
      if (c == 0) { scale = 0; break; }
      scale *= c;
    }
    counter += scale;
  }
  }
  // scale down by number of samples
  total = uint64_t(counter * m / num_samples);
  // symmetry breaking
  int multiplicity = 2;
  assert(k >= 3);
  for (int i = 3; i <= k; i++) {
    multiplicity *= i;
  }
  total = total / multiplicity;
  std::cout << "Sampling time = " << t.Seconds() << " sec\n";
}

// use matching order and symmetry breaking to prune
void sample_clique_prune_mo_sb(Graph &g, int k, eidType num_samples, uint64_t &total) {
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";

  Timer t;
  t.Start();
  g.orientation();
  g.init_simple_edgelist();
  t.Stop();
  std::cout << "Preprocessing time = " << t.Seconds() << " sec\n";

  auto m = g.E();
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 

  t.Start();
  #pragma omp parallel reduction(+ : counter)
  {
  std::random_device rd;
  rd_engine gen(rd());
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    auto eid = edge_dist(gen);
    double scale = 1.;
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VertexSet temp[2];
    intersection(g.N(v0), g.N(v1), temp[0]);
    vidType c = temp[0].size();
    if (c == 0) continue;
    if (k == 3) {
      counter += scale * c;
      continue;
    }
    auto idx0 = random_select_single<vidType>(0, c-1, gen);
    vidType v = temp[0][idx0];
    scale *= c;
    for (int j = 2; j < k-1; j++) {
      temp[(j+1)%2].clear();
      if (j == k - 2) {
        c = intersection_num(g.N(v), temp[j%2]);
      } else {
        intersection(g.N(v), temp[j%2], temp[(j+1)%2]);
        c = temp[(j+1)%2].size();
        if (c == 0) { scale = 0; break; }
        auto id = random_select_single<vidType>(0, c-1, gen);
        v = temp[(j+1)%2][id];
      }
      if (c == 0) { scale = 0; break; }
      scale *= c;
    }
    counter += scale;
  }
  }
  // scale down by number of samples
  total = counter * m / num_samples;
  t.Stop();
  std::cout << "Sampling time = " << t.Seconds() << " sec\n";
}

void sample_4clique(Graph &g, eidType num_samples, uint64_t &total) {
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
 
  Timer t;
  t.Start();
  g.orientation();
  g.init_simple_edgelist();
  t.Stop();
  std::cout << "Preprocessing time = " << t.Seconds() << " sec\n";

  auto m = g.E();
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 

  t.Start();
  #pragma omp parallel reduction(+ : counter)
  {
  std::random_device rd;
  rd_engine gen(rd());
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    auto eid = edge_dist(gen);
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto y0y1 = g.N(v0) & g.N(v1);
    auto d1 = y0y1.size();
    if (d1 < 1) continue;
    auto idx1 = random_select_single<vidType>(0, d1-1, gen);
    auto v2 = y0y1[idx1];
    counter += intersection_num(y0y1, g.N(v2)) * g.E() * d1;
  }
  }
  total = counter / num_samples;
  t.Stop();
  std::cout << "Sampling time = " << t.Seconds() << " sec\n";
}
