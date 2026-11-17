- `candidate_public_server_compute_time_ns_median`: Median candidate server compute time over
  the 1,000 measured queries of the train experiment, in nanoseconds; -1 when unavailable.
- `candidate_public_online_time_ms_median`: Candidate end-to-end online time of the train
  experiment, in milliseconds per query; -1 when unavailable.
- `candidate_public_preprocessing_seconds_large`: Candidate client preprocessing time of the
  train experiment, in seconds; -1 when unavailable.
- `candidate_public_online_communication_bytes_large`: Candidate mean request-plus-response
  bytes per query in the train experiment; -1 when unavailable.
- `candidate_public_client_storage_bytes_large`: Candidate client storage in the train
  experiment (the larger of the peak live heap readings), in bytes; -1 when unavailable.
- `candidate_public_validity_code`: First failed check: 0 valid; 3 build; 4 client crashed,
  broke the harness protocol or timed out; 5 hint export differs; 6 request is not the reduction
  of the reference request; 7 wrong retrieved value or failed server call; 8 client process
  memory limit; 9 server replay check; 10 evaluation time limit; 11 failed an additional query;
  12 primary-hint audit failed; -1 no verdict (codes 1 and 2 are now shared rejections).
