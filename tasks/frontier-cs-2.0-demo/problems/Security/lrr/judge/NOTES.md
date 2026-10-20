# Qualification notes

The formal task was reduced to 100 cases to accelerate qualification. Selection is fixed before GPU/model runs: the first 10 records of each development stratum and first 40 of each held-out stratum from the frozen 1,000-case reconstruction. All selected records inherit previously verified native Python compilability and constructor coverage. Original paper membership is unavailable.

The prior 1,000-case CPU calibration stopped after 327 retained inferences; it produced no full-suite score. Its source and partial outputs remain historical evidence. A 32-case CPU timing profile completed in 523.3 seconds; this is not a GPU estimate.

The GPU launcher preflight passed. Actual full GPU qualification is pending. The model bytes and engine source revision are pinned; positive complete layer offload is required at runtime. The eight paper-related metric rules, sample counts and reference patch are unchanged. No successful beating solver is a prerequisite, but execution failures are not numerical scores.
