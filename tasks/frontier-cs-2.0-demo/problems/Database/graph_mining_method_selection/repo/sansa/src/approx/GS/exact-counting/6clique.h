// ad-hoc 6-clique (use DAG)
  printf("6clique\n");
  #pragma omp parallel for schedule(dynamic, chunk_size) reduction(+:counter)
  for (vidType v1 = 0; v1 < g.V(); v1++) {
    uint64_t local_counter = 0;
    auto y1 = g.N(v1);
    for (auto v2 : y1) {
      auto y1y2 = intersection_set(y1, g.N(v2));
      for (auto v3 : y1y2) {
        auto y1y2y3 = intersection_set(y1y2, g.N(v3));
         for (auto v4 : y1y2y3) {
          auto y1y2y3y4 = intersection_set(y1y2y3, g.N(v4));
          for (auto v5 : y1y2y3y4) {
            local_counter += intersection_num(y1y2y3y4, g.N(v5));
          }
         }
      }
    }
    counter += local_counter;
  }
if(!dag) {
  counter = counter / (6*5*4*3*2*1);
}
