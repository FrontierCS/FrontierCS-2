std::vector<std::vector<uint64_t>> global_counters(num_threads);
for (int tid = 0; tid < num_threads; tid ++) {
    auto &local_counters = global_counters[tid];
    local_counters.resize(num_patterns);
    std::fill(local_counters.begin(), local_counters.end(), 0);
} 
  #pragma omp parallel
  {
    auto &counter2 = global_counters.at(omp_get_thread_num());
    #pragma omp for schedule(dynamic,1)
    for(vidType v0 = 0; v0 < g.V(); v0++) {
      VertexSet y0 = g.N(v0);
      VertexSet y0f0 = bounded(y0,v0);
      for(vidType idx1 = 0; idx1 < y0.size(); idx1++) {
        vidType v1 = y0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        VertexSet y0n1f1 = difference_set(y0, y1, v1);
        for(vidType idx2 = 0; idx2 < y0n1f1.size(); idx2++) {
          vidType v2 = y0n1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter2[0] += difference_num(y0n1f1, y2, v2);
        }
      }
      for(vidType idx1 = 0; idx1 < y0f0.size(); idx1++) {
        vidType v1 = y0f0.begin()[idx1];
        VertexSet y1 = g.N(v1);
        VertexSet y0y1 = intersection_set(y0, y1);
        VertexSet y0f0y1f1 = intersection_set(y0f0, y1, v1);
        VertexSet n0y1; difference_set(n0y1,y1, y0);
        VertexSet n0f0y1; difference_set(n0f0y1,y1, y0);
        VertexSet y0n1 = difference_set(y0, y1);
        VertexSet y0f0n1f1 = difference_set(y0f0, y1, v1);
        for(vidType idx2 = 0; idx2 < y0y1.size(); idx2++) {
          vidType v2 = y0y1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter2[4] += difference_num(y0y1, y2, v2);
          VertexSet n0n1y2; counter2[2] += difference_num(difference_set(n0n1y2,y2, y0), y1);
        }
        for(vidType idx2 = 0; idx2 < y0f0y1f1.size(); idx2++) {
          vidType v2 = y0f0y1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter2[5] += intersection_num(y0f0y1f1, y2, v2);
        }
        for(vidType idx2 = 0; idx2 < y0n1.size(); idx2++) {
          vidType v2 = y0n1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter2[1] += difference_num(n0y1, y2);
        }
        for(vidType idx2 = 0; idx2 < y0f0n1f1.size(); idx2++) {
          vidType v2 = y0f0n1f1.begin()[idx2];
          VertexSet y2 = g.N(v2);
          counter2[3] += intersection_num(n0f0y1, y2, v0);
        }
      }
    }
  }

for (int tid = 0; tid < num_threads; tid++) {
    for (int pid = 0; pid < num_patterns; pid++) {
      total_vec[pid] += global_counters[tid][pid];
    }
}