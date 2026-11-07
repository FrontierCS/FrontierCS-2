#include "graph.h"
#include "sample.hh"
#include "pattern.h"

uint64_t max(uint64_t a, uint64_t b, uint64_t c) {
    return (a >= b) ? ((a >= c) ? a : c) : ((b >= c) ? b : c);
}

uint64_t min(uint64_t a, uint64_t b, uint64_t c) {
    return (a <= b) ? ((a <= c) ? a : c) : ((b <= c) ? b : c);
}



void sample_clique(Graph* g, int k, eidType num_samples, uint64_t &total);
void sample_house(Graph* g, eidType num_samples, uint64_t &total);
void sample_pattern(Graph* g, int k, eidType num_samples, uint64_t &total, Pattern* p);
void sample_path_sb(Graph* g, int k, eidType num_samples, uint64_t &total);


int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout << "Usage: " << argv[0] << " <graph> <number_samples> <pattern_k> <pattern_name> <DAG>\n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph 3 1000\n";
    exit(1);
  }
  std::cout << "DAG?" << atoi(argv[5]);
  Graph g(argv[1], atoi(argv[5]));
  g.print_meta_data();

  int ns = atoi(argv[2]);
  int k = atoi(argv[3]);
  assert(k > 2);

  Pattern p(argv[4]);

  // TODO: should use init or simple?
  // init simple works for DAG
  // init_edgelist might be correct for normal?? unsure...
  g.init_simple_edgelist();


  Timer t;
  t.Start();
  uint64_t true_count;
  sample_pattern(&g,k,ns, true_count, &p);
  t.Stop();
  std::cout << "runtime = " << t.Seconds() << " sec\n";
  std::cout << "estimated count: " << true_count << "\n";
}

void sample_clique(Graph* g, int k, eidType num_samples, uint64_t &total) {
  auto m = g->E();
  std::cout << "m: " << m <<"\n";
  std::cout << "k: " << k <<"\n";

  double counter = 0;
  uint64_t hits = 0;

  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 
  std::random_device rd;
  rd_engine gen(rd());

  std::cout << "here?\n";
  Timer t;
  t.Start();
  #pragma omp parallel for reduction(+ : counter, hits)
  for (int64_t i = 0; i < num_samples; i++) {
    auto eid = edge_dist(gen);
    double scale = m;
    auto v0 = g->get_src(eid);
    auto v1 = g->get_dst(eid);
    VertexSet temp[2];
    intersection(g->N(v0), g->N(v1), temp[0]);
    vidType c = temp[0].size();
    if (c == 0) continue;
    if (k == 3) {
      hits += 1;
      counter += scale * c;
      continue;
    }
    auto idx0 = random_select_single<vidType>(0, c-1, gen);
    vidType v = temp[0][idx0];
    scale *= c;
    for (int j = 2; j < k-1; j++) {
      temp[(j+1)%2].clear();
      if (j == k - 2) {
        c = intersection_num(g->N(v), temp[j%2]);
      } else {
        intersection(g->N(v), temp[j%2], temp[(j+1)%2]);
        c = temp[(j+1)%2].size();
        if (c == 0) { scale = 0; break; }
        auto id = random_select_single<vidType>(0, c-1, gen);
        v = temp[(j+1)%2][id];
      }
      if (c == 0) { scale = 0; break; }
      scale *= c;
    }
    counter += scale;
    hits += 1;
  }
  std::cout << "successful samples = " << hits << " sccess_rate = " << double(hits) / double(num_samples) << "\n";

  // scale down by number of samples
  total = counter / num_samples;
}

void sample_path_sb(Graph* g, int k, eidType num_samples, uint64_t &total) {
  //g->init_simple_edgelist();
  eidType m = g->E();
  __int128_t counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
  //assert(num_samples%num_threads == 0);
  auto num_samples_per_thread = num_samples / num_threads;
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  std::random_device rd;
  rd_engine gen(rd());


  #pragma omp parallel for reduction(+ : counter)
  for (eidType i = 0; i < num_samples; i++) {
    auto eid = edge_dist(gen);
    uint64_t scale = m;
    auto v0 = g->get_src(eid);
    auto v1 = g->get_dst(eid);
    VertexSet vs;
    vs.add(v0);
    vs.add(v1);
    sort_vertexset(vs);
    vidType v = v1;
    for (int j = 2; j < k; j++) {
      vidType c = 0;
      if (j == k-1) {
        c = difference_num(g->N(v), vs, v0); // v0 is used for symmetry breaking
      } else {
        VertexSet candidate_set;
        difference_set(candidate_set, g->N(v), vs);
        c = candidate_set.size();
        if (c == 0) { scale = 0; break; }
        auto id = random_select_single(0, c-1, gen);
        v = candidate_set[id];
        vs.add(v);
        sort_vertexset(vs);
      }
      if (c == 0) { scale = 0; break; }
      scale *= c;
    }
    counter += scale;
  }
  
  total = counter / num_samples; // scale down by number of samples
}


void sample_house(Graph* g, eidType num_samples, uint64_t &total) {
  auto m = g->E();//g->init_edgelist(true);
  double counter = 0;
  uint64_t hits = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
 
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 

  std::random_device rd;
  rd_engine gen(rd());
  #pragma omp parallel for reduction(+ : counter, hits)
  for (int64_t i = 0; i < num_samples; i++) {
    auto eid = edge_dist(gen);
    auto v0 = g->get_src(eid);
    auto v1 = g->get_dst(eid);
    auto d0 = g->get_degree(v0);
    if (d0 < 3) continue;
    auto d1 = g->get_degree(v1);
    if (d1 < 3) continue;
    auto y0y1 = g->N(v0) & g->N(v1);
    auto c0 = y0y1.size();
    if (c0 < 1) continue;
    auto idx0 = random_select_single<vidType>(0, c0-1, gen);
    auto v2 = y0y1[idx0];
    auto u = v1 > v2 ? v2 : v1;
    auto w = v1 > v2 ? v1 : v2;
    VertexSet vs12;
    vs12.add(u);
    vs12.add(w);
    auto candidate_v3 = difference_set(g->N(v0), vs12);
    auto c1 = candidate_v3.size();
    if (c1 < 1) continue;
    auto idx1 = random_select_single<vidType>(0, c1-1, gen);
    auto v3 = candidate_v3[idx1];
    u = v0 > v2 ? v2 : v0;
    w = v0 > v2 ? v0 : v2;
    VertexSet vs02;
    vs02.add(u);
    vs02.add(w);
    auto candidate_v4 = difference_set(g->N(v1), vs02);
    auto c2 = intersection_num(g->N(v3), candidate_v4);
    if (c2 < 1) continue;
    double scale = m;
    counter += scale * c0 * c1 * c2;
    hits += 1;
  }
  
  std::cout << "successful samples = " << hits << " sccess_rate = " << double(hits) / double(num_samples) << "\n";
  total = counter / num_samples;
}

void sample_dumbbell(Graph* g, eidType num_samples, uint64_t &total) {
  auto m = g->E();
  std::cout << "m = " << m << "\n";
  double counter = 0;
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
 
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
  auto num_samples_per_thread = num_samples / num_threads; 
  int64_t success_sample = 0;

  std::random_device rd;
  rd_engine gen(rd());
  #pragma omp parallel for reduction(+ : counter)
  for (int64_t i = 0; i < num_samples; i++) {
    auto eid = edge_dist(gen);
    auto v0 = g->get_src(eid);
    auto v1 = g->get_dst(eid);
    auto d0 = g->get_degree(v0);
    if (d0 < 3) continue;
    auto d1 = g->get_degree(v1);
    if (d1 < 3) continue;
    VertexSet vs1;
    vs1.add(v1);
    auto y0n1 = difference_set(g->N(v0), vs1);
    auto c0 = y0n1.size();
    if (c0 < 1) continue;
    auto idx0 = random_select_single<vidType>(0, c0-1, gen);
    auto v2 = y0n1[idx0];
    auto y0y2 = intersection_set(y0n1, g->N(v2), v2);
    auto c1 = y0y2.size();
    if (c1 < 1) continue;
    auto idx1 = random_select_single<vidType>(0, c1-1, gen);
    auto v3 = y0y2[idx1];

    VertexSet vs0;
    vs0.add(v0);
    vs0.add(v2);
    vs0.add(v3);
    auto y1n0 = difference_set(g->N(v1), vs0);
    auto c2 = y1n0.size();
    if (c2 < 1) continue;
    auto idx2 = random_select_single<vidType>(0, c2-1, gen);
    auto v4 = y1n0[idx2];
    auto y1y4 = intersection_set(y1n0, g->N(v4), v4);
    auto c3 = y1y4.size();
    if (c3 < 1) continue;
    //auto idx3 = random_select_single<vidType>(0, c3-1, gen);
    //auto v5 = y1y4[idx3];
    //std::cout << "sample succeed!\n";
    success_sample += 1;

    double scale = m;
    counter += scale * c0 * c1 * c2 * c3;
  }
  std::cout << "successful samples = " << success_sample << " sccess_rate = " << double(success_sample) / double(num_samples) << "\n";
  total = counter / num_samples;
}


void sample_pattern(Graph* g, int k, eidType num_samples, uint64_t &total, Pattern* p) {
  std::cout << "here " << k << " ns: "<< num_samples << "\n";
  if(p->is_4clique() || p->is_triangle() || p->is_5clique() || p->is_6clique() || p->is_9clique()) {
    std::cout << "clique\n";
    sample_clique(g,k,num_samples,total);
  } else if (p->is_house()) {
    sample_house(g,num_samples, total);
  } else  if (p->is_triangletriangle()) {
    sample_dumbbell(g,num_samples, total);
  } else {
    sample_clique(g,k,num_samples,total);

    //sample_path_sb(g,k,num_samples, total);
  }
}

