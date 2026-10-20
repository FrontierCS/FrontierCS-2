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


void sample_clique(Graph* g, int k, eidType num_samples, uint64_t &total,  double err_target, double std_devs, uint64_t &real);
void sample_house(Graph* g, eidType num_samples, uint64_t &total,  double err_target, double std_devs, uint64_t &real);
void sample_pattern(Graph* g, int k, eidType num_samples, uint64_t &total, Pattern* p,  double err_target, double std_devs, uint64_t &real);
void sample_path_sb(Graph* g, int k, eidType num_samples, uint64_t &total,  double err_target, double std_devs,uint64_t &real);


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
  t.Start();
  uint64_t true_count;
  sample_pattern(&g,k,ns, true_count, &p, e, std, real);
  t.Stop();
  std::cout << "runtime = " << t.Seconds() << " sec\n";
  std::cout << "estimated count: " << true_count << "\n";
}

void sample_clique(Graph* g, int k, eidType num_samples, uint64_t &total, double err_target, double std_devs, uint64_t &real) {
  auto m = g->E();
  std::cout << "m: " << m <<"\n";
  std::cout << "k: " << k <<"\n";

  double counter = 0;
  double counter_sq = 0;
  uint64_t avg;
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

      #pragma omp parallel for reduction(+ : counter, counter_sq)
      for (int64_t i = 0; i < num_samples; i++) {
        std::uniform_int_distribution<eidType> edge_dist(0, m-1);
        auto eid = edge_dist(gens[omp_get_thread_num()]);
        //std::cout << "omp: " << omp_get_thread_num() << "\n";
        //std::cout << "eid: " << eid << "\n";
        double scale = m;
        auto v0 = g->get_src(eid);
        auto v1 = g->get_dst(eid);
        VertexSet temp[2];
        intersection(g->N(v0), g->N(v1), temp[0]);
        vidType c = temp[0].size();
        if (c == 0) continue;
        if (k == 3) {
        counter += scale * c;
        counter_sq += scale*c*scale*c;
        continue;
        }
        auto idx0 = random_select_single<vidType>(0, c-1, gens[omp_get_thread_num()]);
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
            auto id = random_select_single<vidType>(0, c-1, gens[omp_get_thread_num()]);
            v = temp[(j+1)%2][id];
        }
        if (c == 0) { scale = 0; break; }
        scale *= c;
        }
        counter += scale;
        counter_sq += scale*scale;
      }
    
    auto total_ns = num_samples * (j+1);
    std::cout << "total samples: " << total_ns << "\n";
    avg = counter/total_ns;
    
    auto var = counter_sq/total_ns - avg*avg;
    auto std_dev = std::sqrt(var / total_ns) ;
    double error = std_devs*std_dev/avg;

    std::cout << "error: " << error << "\n";
    std::cout << "real error: " << diff(avg,real)/(1.0*real) << "\n";
    std::cout << "current count: " << avg << "\n";

    if(error < err_target) {
        break;
    }
  }
  
  // scale down by number of samples
  total = avg;
  
}

void sample_clique_lock(Graph* g, int k, eidType num_samples, uint64_t &total, double err_target, double std_devs, uint64_t &real) {
  auto m = g->E();
  std::cout << "m: " << m <<"\n";
  std::cout << "k: " << k <<"\n";

  double counter = 0;
  double counter_sq = 0;
  uint64_t avg;
  int num_threads = 1;

  omp_lock_t writelock;
  omp_init_lock(&writelock);

  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP (" << num_threads << " threads)\n";
  auto num_samples_per_thread = num_samples / num_threads; 

  std::random_device rd;
  rd_engine gen(rd());
  std::uniform_int_distribution<eidType> edge_dist(0, m-1);
    
  Timer t;
  t.Start();
  for (int64_t j = 0; j < 1000000000; j++) {


      #pragma omp parallel for reduction(+ : counter, counter_sq)
      for (int64_t i = 0; i < num_samples; i++) {

        omp_set_lock(&writelock);
        auto eid = edge_dist(gen);
        omp_unset_lock(&writelock);
      
        double scale = m;
        auto v0 = g->get_src(eid);
        auto v1 = g->get_dst(eid);
        VertexSet temp[2];
        intersection(g->N(v0), g->N(v1), temp[0]);
        vidType c = temp[0].size();
        if (c == 0) continue;
        if (k == 3) {
        counter += scale * c;
        counter_sq += scale*c*scale*c;
        continue;
        }
        omp_set_lock(&writelock);
        auto idx0 = random_select_single<vidType>(0, c-1, gen);
        omp_unset_lock(&writelock);
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
            omp_set_lock(&writelock);
            auto id = random_select_single<vidType>(0, c-1, gen);
            omp_unset_lock(&writelock);
            v = temp[(j+1)%2][id];
        }
        if (c == 0) { scale = 0; break; }
        scale *= c;
        }
        counter += scale;
        counter_sq += scale*scale;
      }
    
    auto total_ns = num_samples * (j+1);
    std::cout << "total samples: " << total_ns << "\n";
    avg = counter/total_ns;
    
    auto var = counter_sq/total_ns - avg*avg;
    auto std_dev = std::sqrt(var / total_ns) ;
    double error = std_devs*std_dev/avg;

    std::cout << "error: " << error << "\n";
    std::cout << "real error: " << diff(avg,real)/(1.0*real) << "\n";
    std::cout << "current count: " << avg << "\n";

    if(error < err_target) {
        break;
    }
  }
  
  // scale down by number of samples
  total = avg;
  
}

void sample_path_sb(Graph* g, int k, eidType num_samples, uint64_t &total,double err_target, double std_devs, uint64_t &real) {
  //g->init_simple_edgelist();
  eidType m = g->E();

  double counter = 0;
  double counter_sq = 0;
  double avg = 0;
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

  for (int64_t j = 0; j < 1000000000; j++) {
    //#pragma omp parallel for reduction(+ : counter)
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
      counter_sq += ((double) scale)*((double) scale);
    }
    std::cout << "counter: " << counter <<" counter_sq: " << counter_sq<< "\n";
    auto total_ns = num_samples * (j+1);
    std::cout << "total samples: " << total_ns << "\n";
    avg = counter/total_ns;
    std::cout << "avg: " << avg << "\n";

    auto var = counter_sq/total_ns - avg*avg;
    auto std_dev = std::sqrt(var / total_ns) ;
    std::cout << "var: " << var << " std: " << std_dev << "\n";


    double error = std_devs*std_dev/avg;
    std::cout << "error: " << error << "\n";
    std::cout << "real error: " << diff(avg,real)/(1.0*real) << "\n";
    std::cout << "current count: " << avg << "\n";

    // if(error < err_target) {
    //     break;
    // }
    if(total_ns >= 1000000000) {
        break;
    }
  }
  
  total = avg; // scale down by number of samples
}


void sample_house(Graph* g, eidType num_samples, uint64_t &total,double err_target, double std_devs, uint64_t &real) {
  auto m = g->E();//g->init_edgelist(true);
  double counter = 0;
  double counter_sq = 0;
  uint64_t avg;
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
  for (int64_t j = 0; j < 1000000000; j++) {
    //#pragma omp parallel for reduction(+ : counter)
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
      counter_sq += (scale * c0 * c1 * c2)*(scale * c0 * c1 * c2);

    }
      auto total_ns = num_samples * (j+1);
      std::cout << "total samples: " << total_ns << "\n";
      avg = counter/total_ns;
      
      // SYM BREAK
      //avg = avg / 2;
      // SYM BREAK var
      //auto var = counter_sq/total_ns/2 - avg*avg;

      auto var = counter_sq/total_ns - avg*avg;
      auto std_dev = std::sqrt(var / total_ns) ;

      double error = std_devs*std_dev/avg;
      std::cout << "error: " << error << "\n";
      std::cout << "real error: " << diff(avg,real)/(1.0*real) << "\n";
      std::cout << "current count: " << avg << "\n";

      if(error < err_target) {
          break;
      }
  
  }
  
  total = avg;
}

void sample_dumbbell(Graph* g, eidType num_samples, uint64_t &total,double err_target, double std_devs, uint64_t real) {
  auto m = g->E();
  std::cout << "m = " << m << "\n";
  double counter = 0;
  double counter_sq = 0;
  double avg;
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

  for (int64_t j = 0; j < 1000000000; j++) {

    //#pragma omp parallel for reduction(+ : counter)
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
      counter_sq += (scale * c0 * c1 * c2 * c3)*(scale * c0 * c1 * c2 * c3);
    }
    //std::cout << "counter: " << counter <<" counter_sq: " << counter_sq<< "\n";
      auto total_ns = num_samples * (j+1);

      std::cout << "total samples: " << total_ns << "\n";
      avg = counter/total_ns;

      // // SYM BREAK
      avg = avg/2.2;

      auto var = counter_sq/total_ns/2.2 - avg*avg;
      auto std_dev = std::sqrt(var / total_ns) ;

      double error = std_devs*std_dev/avg;
      std::cout << "error: " << error << "\n";
      std::cout << "real error: " << diff(avg,real)/(1.0*real) << "\n";
      std::cout << "current count: " << avg << "\n";

      if(error < err_target) {
          break;
      }

      // if(total_ns >= 10000000) {
      //   break;
      // }
  }

  std::cout << "successful samples = " << success_sample << " success_rate = " << double(success_sample) / double(num_samples) << "\n";
  total = avg;
}


void sample_pattern(Graph* g, int k, eidType num_samples, uint64_t &total, Pattern* p,  double err_target, double std_devs, uint64_t &real) {
  std::cout << "here " << k << " ns: "<< num_samples << "sampling: " << p->get_name();
  if(p->is_4clique() || p->is_triangle() || p->is_5clique() || p->is_6clique() || p->is_9clique()) {
    g->init_simple_edgelist();
    sample_clique(g,k,num_samples,total, err_target, std_devs,real);
  } else if (p->is_house()) {
    g->init_edgelist(true);
    sample_house(g,num_samples, total,  err_target, std_devs,real);
  } else  if (p->is_triangletriangle()) {
    g->init_edgelist(true);
    std::cout << "DUMBBELL\n";
    sample_dumbbell(g,num_samples, total,  err_target, std_devs,real);
  } else {
    g->init_simple_edgelist();
    sample_path_sb(g,k,num_samples, total,  err_target, std_devs, real);
  }
}

