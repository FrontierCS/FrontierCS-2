#include "graph.h"
#include "sample.hh"

void sample_house_base(Graph &g, eidType num_samples, uint64_t &total);
void sample_house(Graph &g, eidType num_samples, uint64_t &total);

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <graph> <num_samples>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  Graph g(argv[1]);
  g.print_meta_data();
  int64_t num_samples = atoi(argv[2]);
  uint64_t total = 0;

  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
 
  //Timer t;
  //t.Start();
  sample_house_base(g, num_samples, total);
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n\n";

  sample_house(g, num_samples, total);
  std::cout << "Estimated count " << FormatWithCommas(total) << "\n";
  //t.Stop();
  //std::cout << "Runtime = " << t.Seconds() << " sec\n";
}

struct Edge {
  vidType src;
  vidType dst;
};

bool prune(int round, Graph &g, Edge extending_edge, std::vector<vidType> &chain) {
           //vidType &node1, vidType &node2, vidType &node3, vidType &node4, vidType &node5) {
  vidType node1 = 0;
  if (round == 0) {
    auto node2 = extending_edge.src;
    auto node3 = extending_edge.dst;
    if (node2 == chain[0]) node1 = chain[1];
    else node1 = chain[0];
    if (!g.is_connected(node1, node3)) return true;
    chain[0] = node1;
    chain[1] = node2;
    chain.push_back(node3);
  } else if (round == 1) {
    auto node3_ = extending_edge.src;
    auto node4 = extending_edge.dst;
    if (node3_ == chain[0])
      chain.insert(chain.begin(), node4);
    else
      chain.push_back(node4);
  } else if (round == 2) {
    auto node4_ = extending_edge.src;
    auto node5 = extending_edge.dst;
    if (node4_ == chain[0])
      chain.insert(chain.begin(), node5);
    else    
      chain.push_back(node5);
  }
  return false;
}

uint64_t sample_edge(Graph &g, eidType eid, default_random_engine & rand_generator, Edge &extending_edge,
                     std::set<vidType> &open_vs, std::vector<vidType> &vlist) {
  uint64_t total_d_extending_edge = 0;
  vector<vidType> open_set_neighbor;
  for (auto u : open_vs) {
    auto begin_idx = g.edge_begin(u);
    auto end_idx = g.edge_end(u);
    for (auto idx = begin_idx; idx < end_idx; idx++) {
      auto w = g.getEdgeDst(idx);
      if (std::find(vlist.begin(), vlist.end(), w) != vlist.end()) continue;
      total_d_extending_edge++;
      open_set_neighbor.push_back(idx);
    }
  }
  if (total_d_extending_edge == 0) return 0;
  uniform_int_distribution <uint32_t> next_edge_distribution(0, total_d_extending_edge - 1);
  auto edge_idx = open_set_neighbor[next_edge_distribution(rand_generator)];
  auto u0 = g.get_src(edge_idx);
  auto u1 = g.get_dst(edge_idx);
  open_vs.erase(u0);
  open_vs.emplace(u1);
  //vlist.push_back(u1);
  extending_edge.src = u0;
  extending_edge.dst = u1;
  return total_d_extending_edge;
}

void sample_house_base(Graph &g, eidType num_samples, uint64_t &total) {
  std::cout << "running ns-base for house\n";
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  auto num_samples_per_thread = num_samples / num_threads; 

  Timer t;
  t.Start();
  auto m = g.init_edgelist();
  t.Stop();
  std::cout << "Preprocessing time = " << t.Seconds() << " sec\n";
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  double counter = 0;
  uint64_t hit_count = 0;

  t.Start();
  #pragma omp parallel reduction(+ : counter,hit_count)
  {
  std::random_device rd;
  rd_engine my_rng(rd());
  //default_random_engine my_rng(rd());
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    auto eid = edge_dist(my_rng);
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    std::set<vidType> open_vs;
    open_vs.insert(v0);
    open_vs.insert(v1);
    std::vector<vidType> path_vertices;
    path_vertices.push_back(v0);
    path_vertices.push_back(v1);

    double scale = 1.0;
    for (int j = 0; j < 3; j++) {
      Edge edge;
      auto num = sample_edge(g, eid, my_rng, edge, open_vs, path_vertices);
      if (num == 0 || prune(j, g, edge, path_vertices)) { scale = 0; break; }
      scale *= num;
    }
    if (scale == 0) continue;
    if (!g.is_connected(path_vertices[0], path_vertices[4])) continue;
    counter += scale;
    hit_count ++;
  }
  }
  total = counter * m / num_samples / 16;
  float hit_rate = float(hit_count) / num_samples;
  t.Stop();
  std::cout << "Sampling time = " << t.Seconds() << " sec\n";
  std::cout << "hit_rate = " << hit_rate << " \n";
}

void sample_house(Graph &g, eidType num_samples, uint64_t &total) {
  std::cout << "running ns-prune for house\n";
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  auto num_samples_per_thread = num_samples / num_threads; 

  Timer t;
  t.Start();
  auto m = g.reinit_edgelist(true);
  t.Stop();
  std::cout << "Preprocessing time = " << t.Seconds() << " sec\n";
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  double counter = 0;
  uint64_t hit_count = 0;;

  t.Start();
  #pragma omp parallel reduction(+ : counter,hit_count)
  {
  std::random_device rd;
  rd_engine gen(rd());
  for (int64_t i = 0; i < num_samples_per_thread; i++) {
    auto eid = edge_dist(gen);
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto d0 = g.get_degree(v0);
    if (d0 < 3) continue;
    auto d1 = g.get_degree(v1);
    if (d1 < 3) continue;
    auto y0y1 = g.N(v0) & g.N(v1);
    auto c0 = y0y1.size();
    if (c0 < 1) continue;
    auto idx0 = random_select_single<vidType>(0, c0-1, gen);
    auto v2 = y0y1[idx0];
    auto u = v1 > v2 ? v2 : v1;
    auto w = v1 > v2 ? v1 : v2;
    VertexSet vs12;
    vs12.add(u);
    vs12.add(w);
    auto candidate_v3 = difference_set(g.N(v0), vs12);
    auto c1 = candidate_v3.size();
    if (c1 < 1) continue;
    auto idx1 = random_select_single<vidType>(0, c1-1, gen);
    auto v3 = candidate_v3[idx1];
    u = v0 > v2 ? v2 : v0;
    w = v0 > v2 ? v0 : v2;
    VertexSet vs02;
    vs02.add(u);
    vs02.add(w);
    auto candidate_v4 = difference_set(g.N(v1), vs02);
    auto c2 = intersection_num(g.N(v3), candidate_v4);
    if (c2 < 1) continue;
    counter += c0 * c1 * c2;
    hit_count += 1;
  }
  }
  total = counter * m / num_samples;
  float hit_rate = float(hit_count) / num_samples;
  t.Stop();
  std::cout << "Sampling time = " << t.Seconds() << " sec\n";
  std::cout << "hit_rate = " << hit_rate << " \n";
}
