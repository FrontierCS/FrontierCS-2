- `raw_quality`: Candidate quality Q = 100 * (0.8 * mean branch accuracy + 0.2 * mean
  transaction-path ratio) on the scored epoch.
- `branch_accuracy`: Candidate mean branch-direction accuracy over the scored records, in [0,
  1].
- `transaction_path_accuracy`: Candidate mean transaction-path satisfaction ratio over the
  scored records; normal-category ratios may exceed 1.
- `observations`: Number of scored report records.
