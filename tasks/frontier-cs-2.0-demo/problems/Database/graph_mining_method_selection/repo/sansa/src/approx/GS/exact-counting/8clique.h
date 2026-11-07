std::cout << "running 8clique...\n";

#pragma omp parallel for schedule(dynamic, chunk_size) reduction(+:counter)
  for (vidType v1 = 0; v1 < g.V(); v1++) {
    //auto tid = omp_get_thread_num();
    uint64_t local_counter = 0;
    auto y1 = g.N(v1);
    for (auto v2 : y1) {
      auto y1y2 = intersection_set(y1, g.N(v2));
      for (auto v3 : y1y2) {
        auto y1y2y3 = intersection_set(y1y2, g.N(v3));
        for (auto v4 : y1y2y3) {
            auto y1y2y3y4 = intersection_set(y1y2y3, g.N(v4));
            for (auto v5 : y1y2y3y4) {
              auto y1y2y3y4y5 = intersection_set(y1y2y3y4, g.N(v5));
              for (auto v6 : y1y2y3y4y5) {
                auto y1y2y3y4y5y6 = intersection_set(y1y2y3y4y5, g.N(v6));
                for (auto v7 : y1y2y3y4y5y6) {
                    local_counter += intersection_num(y1y2y3y4y5y6, g.N(v7));
                }
              }
            }
	      }
      }
    }
       counter += local_counter;
  }
  if(!dag) {
      counter = counter / (9*8*7*6*5*4*3*2*1);
  }
