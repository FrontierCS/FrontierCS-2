  #pragma omp parallel for reduction(+ : counter) schedule(dynamic, chunk_size)
  for (vidType u = 0; u < g.V(); u ++) {
    auto adj_u = g.N(u);
    for (auto v : adj_u) {
      counter += (uint64_t)intersection_num(adj_u, g.N(v));
    }
  }

  if(!dag) {
    counter = counter / (3*2*1);
  }
