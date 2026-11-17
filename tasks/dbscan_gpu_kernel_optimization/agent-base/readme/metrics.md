- `geomean_speedup`: Geometric mean over the train workloads of the median paired speedup over
  the naive starter (unitless).
- `dbscan_latency_ms`: Candidate median call latency per train workload, in milliseconds;
  sorted, without workload labels.
- `adjusted_rand_index`: Candidate minimum timed ARI against the naive labels per train
  workload; sorted, without workload labels.
- `noise_mismatch_fraction`: Candidate maximum timed noise-mask mismatch fraction per train
  workload; sorted, without workload labels.
