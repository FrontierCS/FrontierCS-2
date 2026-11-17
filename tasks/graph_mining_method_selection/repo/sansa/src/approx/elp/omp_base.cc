#include "graph.h"
#include "pattern.h"
#include "sampler.h"


uint64_t max(uint64_t a, uint64_t b, uint64_t c) {
    return (a >= b) ? ((a >= c) ? a : c) : ((b >= c) ? b : c);
}

uint64_t min(uint64_t a, uint64_t b, uint64_t c) {
    return (a <= b) ? ((a <= c) ? a : c) : ((b <= c) ? b : c);
}

uint64_t ExactCount(Graph& g, Pattern& p, Sampler& s, int dag, int chunk_size) {
  
  uint64_t counter = 0;

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
  } else if(p.is_5clique()) {
    #include "5clique.h"
  } else if(p.is_4clique()) {
    #include "4clique.h"
  } else if(p.is_5path()) {
    #include "5path.h"
  } else if (p.is_6clique()) {
    #include "6clique.h"
  }

  printf("p_hat = %ld * %lu = %lu \n", s.scale_factor, counter, s.Scale(counter));

  return s.Scale(counter);
}

float ELP(Graph* g, Sampler &s, Pattern &p, uint64_t count, float error, uint64_t scale_factor, uint64_t sgmaxDegree, int dag, int chunk_size) {
   int c = 1000;
   uint64_t avg_last = INT_MAX;
   uint64_t range_last = INT_MAX;
   float scalar = 0; 
   while(true) {

    std::cout << "try " << c << "\n";
    
    Graph* g1 = new Graph();
    Graph* g2 = new Graph();
    Graph* g3 = new Graph();
    
    s.GetGraph(*g, g1, c, p);
    s.GetGraph(*g, g2, c, p);
    s.GetGraph(*g, g3, c, p);

    uint64_t o1 = ExactCount(*g1,p,s,dag,chunk_size); 
    uint64_t o2 = ExactCount(*g2,p,s,dag,chunk_size);
    uint64_t o3 = ExactCount(*g3,p,s,dag,chunk_size);

    std::cout << o1 << " "<< o2 << " " << o3 <<"\n";

    uint64_t avg = (o1 + o2 + o3) / 3.0 ;
    uint64_t range = (max(o1,o2,o3) - min(o1,o2,o3));

    float error_ = (avg>avg_last ? avg-avg_last : avg_last-avg) / (avg*1.0);

    std::cout << "avg: " << avg << " range: "<< range << " error: " << error_ <<"\n";


      if(range_last / (avg_last * 1.0) < 0.1 && error_ < error && range / (avg * 1.0) < .10) {
        std::cout << sgmaxDegree << "^ " << p.get_nodes()-1 << "= " << std::pow(sgmaxDegree,p.get_nodes()-1);
        std::cout << c << "^ " << -p.get_nodes()+1 << "= " << std::pow(c,-p.get_nodes()+1);
        scalar = ((error_ *error_) * avg  * (std::pow(c,-p.get_nodes()+1)));
        printf("found scalar: %f\n", scalar);

        std::cout << p.to_string() << ": " << scalar*1.0 << "\n";
        break;
      }

      c /= 2;
      avg_last = avg;
      range_last = range;
   }

  
  // BRO they use the wrong scaling for pattern count on sparsified graph in arya!!!
   return (scalar * 1.0)*scale_factor / (count*(error*error));
}


void SansaSolver(Graph &g, Pattern &p, Sampler &s, uint64_t &total, int delta, int dag, int chunk_size) {
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("OpenMP Approximate GPM running (%d threads) ...\n", num_threads);
  Timer t;
    
  printf("Running Graph Sample Procedure (%d threads) ...\n", num_threads);
  t.Start();
  Graph* sg = new Graph();
  s.GetGraph(g, sg, delta, p);
  uint64_t sgmaxDegree = sg->compute_max_degree();
  std::cout << "subgraph max degree: " << sgmaxDegree << "\n";
  t.Stop();
  std::cout << "subgraph sample time = " <<  t.Seconds() << " seconds, delta = " << delta << "\n";

  g.print_meta_data();

  t.Start();
  std::cout << "Running the baseline implementation\n";
  //   uint64_t counter = 0;
  // if (p.is_house()) {
  //   #include "house.h"
  // } else if (p.is_pentagon()) {
  //   #include "pentagon.h"
  // } else if (p.is_rectangle()) {
  //   #include "rectangle.h"
  // } else if (p.is_diamond()) {
  //   #include "diamond.h"
  // } else if(p.is_triangle()) {
  //   #include "triangle.h"
  // } else if(p.is_9clique()) {
  //   #include "9clique.h"
  // } else if(p.is_5clique()) {
  //   #include "5clique.h"
  // } else if(p.is_4clique()) {
  //   #include "4clique.h"
  // } else if(p.is_5path()) {
  //   #include "5path.h"
  // } else if (p.is_6clique()) {
  //   #include "6clique.h"
  // }

  // printf("scale factor: %ld\n", s.scale_factor);

  // total = s.Scale(counter);

  uint64_t count = ExactCount(*sg, p, s, dag, chunk_size);

  t.Stop();
  std::cout << "runtime = " <<  t.Seconds() << " seconds\n";

  float scale = ELP(sg, s, p, count, 0.1, s.Scale(1), sgmaxDegree, dag, chunk_size);
  std::cout << "scale factor = " <<  scale << "\n";
  return;
}

