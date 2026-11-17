- `geomean_speedup`: Geometric mean over the train workloads of the naive/candidate median
  loop-latency ratio (unitless).
- `lloyd_loop_latency_ms`: Candidate median two-step loop latency per train workload, in
  milliseconds; sorted, without workload labels.
- `labeled_inertia`: Candidate labeled inertia of the final labels and centroids per train
  workload; sorted, without workload labels.
- `label_mismatch_fraction`: Candidate fraction of final labels that are not an exact nearest
  centroid per train workload; sorted, without workload labels.
