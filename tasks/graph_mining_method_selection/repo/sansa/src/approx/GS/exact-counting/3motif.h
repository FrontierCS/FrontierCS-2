std::vector<std::vector<uint64_t>> global_counters(num_threads);
for (int tid = 0; tid < num_threads; tid ++) {
    auto &local_counters = global_counters[tid];
    local_counters.resize(num_patterns);
    std::fill(local_counters.begin(), local_counters.end(), 0);
}
#pragma omp parallel
  {
    auto &counter = global_counters.at(omp_get_thread_num());
    #pragma omp for schedule(dynamic,1)
    for(vidType v0 = 0; v0 < g.V(); v0++) {
      VertexSet y0 = g.N(v0);
      VertexSet y0f0 = bounded(y0,v0);
      for(vidType idx1 = 0; idx1 < y0.size(); idx1++) {
        vidType v1 = y0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        counter[0] += difference_num(y0, y1, v1);
      }
      for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
        vidType v1 = y0f0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        counter[1] += intersection_num(y0f0, y1, v1);
      }
    }
}

for (int tid = 0; tid < num_threads; tid++)
    for (int pid = 0; pid < num_patterns; pid++)
      total_vec[pid] += global_counters[tid][pid];

