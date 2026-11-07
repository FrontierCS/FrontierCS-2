// Copyright 2020 Massachusetts Institute of Technology
// Contact: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "pattern.h"
#include "sampler.h"

void SansaSolver(Graph &g, Pattern &p, Sampler &s, uint64_t &total, int delta, int dag, int chunk_size);

int main(int argc, char **argv) {
  if(argc < 3) {
    std::cerr << "usage: " << argv[0] << " <graph prefix> <pattern> <delta> [chunk_size(1024)] [dag(0)] \n";
    printf("Example: %s /graph_inputs/mico/graph rectangle\n", argv[0]);
    exit(1);
  }
  std::cout << "Subgraph Listing/Counting (undirected graph only)\n";
  Pattern patt(argv[2]);
  int delta = atoi(argv[3]);
  int dag = 0;
  std::cout << "Pattern: " << patt.get_name() << "\n";
  int chunk_size = 1024;
  if (argc > 4) chunk_size = atoi(argv[4]);
  if (argc > 5) dag = atoi(argv[5]);

 
  Graph g(argv[1], dag);

  auto test_sg = new Graph*[3];

  for(int i =0; i< 3; i++) {
    test_sg[i] = new Graph();
  }

  MultiColorSampler sm;
  sm.GetMultiGraph(g, test_sg,3, delta, patt);
  
  

  g.print_meta_data();


  ColorSampler s;

  std::cout << "Color sampler is use...\n";

  uint64_t h_total = 0;
  SansaSolver(g, patt, s, h_total, delta, dag, chunk_size);
  std::cout << "total_num = " << h_total << "\n";
  std::cout << "--------------------\n";
  return 0;
}

