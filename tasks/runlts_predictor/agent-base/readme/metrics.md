- `train_br_mis_pki_mean`: Candidate mean BrMisPKI (four-decimal printed rate) over the train
  traces.
- `train_cyc_wp_pki_mean`: Candidate mean CycWpPKI (four-decimal printed rate) over the train
  traces.
- `all_metrics_progress`: Counts and percentages of train traces on which both prediction
  comparisons clear together: passed, failed, total, pass_percent, fail_percent.
- `category_means`: Per application category (compress, fp, infra, int, media, web): number of
  train traces and the candidate's mean br_mis_pki and cyc_wp_pki from exact counters.
- `candidate_failure`: 0 for a measured result; otherwise the failing stage of a scored zero: 1
  C++ parse error in the source audit, 2 logical storage over 1,572,864 bits, 3 source audit
  over its limit, 4 build failed or timed out, 5 a simulation failed or timed out, 6 malformed
  simulator output, 7 instruction counts differ from the trace's, 8 another candidate failure.
