#include "graph.h"
#include "sample.hh"
#include "pattern.h"

uint64_t max(uint64_t a, uint64_t b, uint64_t c) {
    return (a >= b) ? ((a >= c) ? a : c) : ((b >= c) ? b : c);
}

uint64_t min(uint64_t a, uint64_t b, uint64_t c) {
    return (a <= b) ? ((a <= c) ? a : c) : ((b <= c) ? b : c);
}


inline int64_t diff(uint64_t first, uint64_t second) {
    uint64_t abs_diff = (first > second) ? (first - second): (second - first);
    return abs_diff;
}


void sample_clique(Graph* g, int k, eidType num_samples, uint64_t &total,  double err_target, double std_devs, uint64_t &real, double &total_ns);
void sample_house(Graph* g, eidType num_samples, uint64_t &total,  double err_target, double std_devs, uint64_t &real, double &total_ns);
void sample_pattern(Graph* g, int k, eidType num_samples, uint64_t &total, Pattern* p,  double err_target, double std_devs, uint64_t &real,double &total_ns, Timer& t);

struct Edge {
  vidType src;
  vidType dst;
};

int main(int argc, char* argv[]) {
  if (argc < 7) {
    std::cout << "Usage: " << argv[0] << " <graph> <number_samples_granularity> <pattern_k> <pattern_name> <err_target> <std_devs> <DAG> <real>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000 1 12534960\n";
    exit(1);
  }
  std::cout << "DAG?: " << atoi(argv[7]) + "\n";
  Graph g(argv[1], atoi(argv[7]));
  g.print_meta_data();

  int ns = atoi(argv[2]);
  int k = atoi(argv[3]);


  //double err_target = atod(argv[5]);
  assert(k > 2);

  Pattern p(argv[4]);

  double e = atof(argv[5]);
  double std = atof(argv[6]);
  uint64_t real = atoll(argv[8]);

  // TODO: should use init or simple?
  // init simple works for DAG
  // init_edgelist might be correct for normal?? unsure...


  Timer t;
  uint64_t true_count;
  double total_ns = 0;
  sample_pattern(&g,k,ns, true_count, &p, e, std, real, total_ns, t);
  std::cout << "runtime = " << t.Seconds() << " sec\n";
  std::cout << "estimated count: " << true_count << "\n";
  std::cout << "samples: " << total_ns << "\n";
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



void sample_clique(Graph* g, int k, eidType num_samples, uint64_t &total, double err_target, double std_devs, uint64_t &real, double &total_ns) {
  auto m = g->E();
  std::cout << "m: " << m <<"\n";
  std::cout << "k: " << k <<"\n";

  double counter = 0;
  double counter_sq = 0;
  double hits = 0;

  double avg;
  int num_threads = 1;


  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
  auto num_samples_per_thread = num_samples / num_threads; 

  // std::random_device rd;
  // rd_engine gen(rd());

  std::random_device rd;
  rd_engine gens[num_threads];

  for(int t = 0; t < num_threads; t++) {
    gens[t] = rd_engine(rd());
  }
    
  Timer t;
  t.Start();
  for (int64_t j = 0; j < 1000000000; j++) {
      
      #pragma omp parallel for reduction(+ : counter, counter_sq) schedule(dynamic, 64)
      for (int64_t i = 0; i < num_samples; i++) {
        std::uniform_int_distribution<eidType> edge_dist(0, m-1);
        auto eid = edge_dist(gens[omp_get_thread_num()]);
        
        std::vector<vidType> sampled_vertices;
        auto scale = connected_subgraph_sampler(*g, k, eid, gens[omp_get_thread_num()], sampled_vertices);
        if (scale == 0) continue;
        //assert(sampled_vertices.size() == k);
        //std::set<vidType> s(sampled_vertices.begin(), sampled_vertices.end());
        //assert(s.size() == k);
        if (is_clique(*g, sampled_vertices)) {
            counter += double(scale)*double(m);
            counter_sq += double(scale)*double(m) * double(scale)*double(m);
            hits += 1;
        }
    }
    
    total_ns = num_samples * (j+1);
    avg = counter/total_ns;
    
    auto var = counter_sq/total_ns - avg*avg;
    auto std_dev = std::sqrt(var / total_ns) ;
    double error = std_devs*std_dev/avg;

    if(error < err_target) {
        break;
    }
  }

  int multiplicity = 2;
  assert(k >= 3);
  for (int i = 3; i <= k; i++) {
    multiplicity *= i;
  }
  total = avg / multiplicity;
  
  std::cout << "hits: " << hits << " hit_rate: " << hits/total_ns << "\n";
}





void sample_house(Graph* g, eidType num_samples, uint64_t &total,double err_target, double std_devs, uint64_t &real, double &total_ns) {
  auto m = g->E();//g->init_edgelist(true);
  double counter = 0;
  double counter_sq = 0;
  double hits = 0;

  double avg;
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

  for (int64_t j = 0; j < 1000000000; j++) {
    #pragma omp parallel for reduction(+ : counter, counter_sq) schedule(dynamic, 1)
    for (int64_t i = 0; i < num_samples; i++) {
    auto eid = edge_dist(gens[omp_get_thread_num()]);
    auto v0 = g->get_src(eid);
    auto v1 = g->get_dst(eid);
    std::set<vidType> open_vs;
    open_vs.insert(v0);
    open_vs.insert(v1);
    std::vector<vidType> path_vertices;
    path_vertices.push_back(v0);
    path_vertices.push_back(v1);

    double scale = 1.0;
    for (int j = 0; j < 3; j++) {
      Edge edge;
      auto num = sample_edge(*g, eid, gens[omp_get_thread_num()], edge, open_vs, path_vertices);
      if (num == 0 || prune(j, *g, edge, path_vertices)) { scale = 0; break; }
      scale *= num;
    }
    if (scale == 0) continue;
    if (!g->is_connected(path_vertices[0], path_vertices[4])) continue;
    counter += scale * m;
    counter_sq += scale * m *scale * m;
    hits ++;
    }

    total_ns = num_samples * (j+1);
    avg = counter/total_ns;
    
    auto var = counter_sq/total_ns - avg*avg;
    auto std_dev = std::sqrt(var / total_ns) ;
    double error = std_devs*std_dev/avg;

    if(error < err_target) {
        break;
    }
  }
  
  total = avg / 16;
  std::cout << "hits: " << hits << " hit_rate: " << hits/total_ns << "\n";
}




void sample_pattern(Graph* g, int k, eidType num_samples, uint64_t &total, Pattern* p,  double err_target, double std_devs, uint64_t &real, double &total_ns, Timer &t) {
  std::cout << "here " << k << " ns: "<< num_samples << "sampling: " << p->get_name() << "\n";
  if(p->is_clique() || p->is_4clique() || p->is_triangle() || p->is_5clique() || p->is_6clique() || p->is_9clique()) {
    std::cout << "CLIQUE\n";
    g->init_simple_edgelist();
    t.Start();
    sample_clique(g,k,num_samples,total, err_target, std_devs,real,total_ns);
    t.Stop();
  } else if (p->is_house()) {
    g->init_edgelist(true);
    std::cout << "HOUSE\n";
    t.Start();
    sample_house(g,num_samples, total,  err_target, std_devs,real,total_ns);
    t.Stop();
  } 
}

