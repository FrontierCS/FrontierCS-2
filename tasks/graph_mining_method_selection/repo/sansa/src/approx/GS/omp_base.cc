#include "graph.h"
#include "pattern.h"
#include "sampler.h"

void SansaSolver(Graph &g, Pattern &p, Sampler &s, uint64_t &total, int delta, int dag, int chunk_size) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP Approximate GPM running (%d threads) ...\n", num_threads);
  uint64_t counter = 0;
  Timer t;
    
  printf("Running Graph Sample Procedure (%d threads) ...\n", num_threads);
  t.Start();
  s.GetGraph(g, delta, p);
  t.Stop();
  std::cout << "subgraph sample time = " <<  t.Seconds() << " seconds, delta = " << delta << "\n";

  g.print_meta_data();

  t.Start();
  std::cout << "Running the baseline implementation\n";
  if (p.is_house()) {
    #include "house.h"
  } else if (p.is_pentagon()) {
    #include "pentagon.h"
  } else if (p.is_rectangle()) {
    #include "rectangle.h"
  } else if (p.is_diamond()) {
    #include "diamond.h"
  } else if(p.is_triangle()) {
    #include "triangle.h"
  } else if(p.is_9clique()) {
    #include "9clique.h"
  } else if(p.is_8clique()) {
    #include "8clique.h"
  } else if(p.is_5clique()) {
    #include "5clique.h"
  } else if(p.is_4clique()) {
    #include "4clique.h"
  } else if(p.is_5path()) {
    #include "5path.h"
  } else if (p.is_6clique()) {
    #include "6clique.h"
  } else if (p.is_triangletriangle()) {
    #include "triangletriangle.h"
  } else if (p.is_3motif()) {
    int num_patterns=2;
    std::vector<uint64_t> total_vec(2, 0);
    #include "3motif.h"
    for (int i = 0; i < num_patterns; i++){
      std::cout << "pattern " << i << ": " << s.Scale(total_vec[i]) << "\n";
      counter += total_vec[i];
    }
  } else if(p.is_4motif()) {
    int num_patterns=6;
    std::vector<uint64_t> total_vec(num_patterns, 0);
    #include "4motif.h"
    for (int i = 0; i < num_patterns; i++) {
      std::cout << "pattern " << i << ": " << s.Scale(total_vec[i]) << "\n";
      counter += total_vec[i];
    }
  }

  printf("scale factor: %ld\n", s.scale_factor);

  total = s.Scale(counter);
  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";
  return;
}