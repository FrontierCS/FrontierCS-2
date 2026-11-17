- `dev_min_throughput_ops_per_second`: Minimum candidate throughput over the public train
  workloads, in operations per second; 0 if any workload is invalid.
- `dev_max_latency_picoseconds`: Maximum candidate worst response latency (cycles times clock
  period) over the public train workloads, in picoseconds; -1 if any workload is invalid.
- `dev_max_clock_period_picoseconds`: Maximum candidate modelled clock period over the public
  train configurations, in picoseconds; -1 if any workload is invalid.
- `dev_min_throughput_ops_per_cycle`: Minimum candidate completed operations per simulated cycle
  over the public train workloads; 0 if any workload is invalid.
- `dev_max_latency_cycles`: Maximum candidate worst response latency over the public train
  workloads, in simulated cycles; -1 if any workload is invalid.
- `dev_valid_fraction`: Fraction of the public train workloads that are fully correct.
