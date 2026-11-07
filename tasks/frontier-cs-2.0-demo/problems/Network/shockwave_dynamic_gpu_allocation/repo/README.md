# Dynamic GPU Allocation

Write the round-by-round GPU allocation policy of a simulated cluster scheduler for training jobs
whose batch size and throughput change during execution. Lower makespan, average job-completion
time (JCT) and finish-time unfairness give a higher score. A run is valid only if it completes
every job with a legal allocation in every round.

## Task

The tree is a reduced copy of the Shockwave scheduler (NSDI 2023, built on Gavel). Its
trace-driven simulator runs the real scheduling path on CPU, advancing training in virtual time
on four simulated V100s. Jobs
(language modeling, image classification, recommendation, translation) adapt their batch size
with Accordion or gradient-noise-scale (`gns`). `ShockwaveScheduler.round_schedule` and its
helper module (`shockwave_helper.py` in `scheduler/`) have been removed; supply a policy.

## What you may change

- `scheduler/shockwave.py` (`ShockwaveScheduler`)
- `shockwave_helper.py` in `scheduler/` (removed from the starter; you may recreate it)
- new or changed modules under `scheduler/solver_extensions/`

At most 20 files may change. Everything else is fixed: arrivals, round advancement, worker
placement, job progress, throughput lookup, batch-size adaptation, completion, waiting delay,
metric calculation and result serialization. The policy may keep state across lifecycle calls
and rounds, but the caller consumes only the list returned by `round_schedule`. Your code must
not change any other simulator behaviour or output (for example by patching simulator objects at
run time).

## Interface

`ShockwaveScheduler.__init__(ngpus, gram, init_metadata, future_nrounds, round_duration,
solver_preference, solver_rel_gap, solver_num_threads, solver_timeout, n_epoch_vars_max,
logapx_bases, logapx_origin, k, lam, rhomax)` receives an ordered mapping of live integer job
IDs to `JobMetaData` objects. The lifecycle methods (`add_metadata`, `remove_metadata`,
`increment_round_ptr`, `schedule_progress`, `deschedule_waiting_delay`, `set_resolve`,
`clear_resolve`) must stay usable by `scheduler.Scheduler`.

The judge's configuration is `future_rounds: 20`, `gpu_ram: 16`, `solver_rel_gap: 0.001`,
`solver_num_threads: 24`, `solver_timeout: 15`, `k: 0.001`, `lambda: 12.0`, `rhomax: 1.0` and
log-approximation bases `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. The retained scheduler also passes
`ngpus = 4`, `round_duration = 120`, `solver_preference = ["GUROBI"]`, `n_epoch_vars_max = 30`
and `logapx_origin = {0.0: 0.000001}`.

`round_schedule()` returns the list of currently active job IDs to run in the next round: no
duplicates, total `nworkers` of the selected jobs at most `self.ngpus`, and nonempty while live
jobs remain. With live jobs 3, 7 and 9 demanding 1, 1 and 2 workers:

```python
>>> scheduler.round_schedule()
[3, 7, 9]
```

The simulator records this round as `{3: (0,), 7: (1,), 9: (2, 3)}` (worker IDs 0-3).

A trace has one job per line with 12 tab-separated fields:

```text
job_type  command  working_directory  num_steps_arg  needs_data_dir  total_steps  scale_factor  mode  priority_weight  SLO  duration_seconds  arrival_seconds
LM (batch size 10)  python3 main.py --cuda --data %s/wikitext2 --batch_size 10  language_modeling  --steps  1  134583  1  gns  1  -1.000000  1418  0
```

`scale_factor` is the job's GPU demand, `mode` is `accordion` or `gns`, `needs_data_dir` is an
integer boolean, and a negative `SLO` means none. A job's duration becomes the sum of its
per-epoch durations from the throughput table; no training command runs.

The throughput table is JSON keyed by worker type, then by the string `(job_type, scale_factor)`;
`null` maps to the isolated iterations per second, another job key to the pair's colocated
throughputs:

```json
{"v100": {"('LM (batch size 10)', 1)": {"null": 94.91094148,
  "('ResNet-18 (batch size 16)', 1)": [67.64173010537195, 32.353384328946916]}}}
```

For each workload the judge runs, from `scheduler/`:

```text
python3 scripts/drivers/simulate_scheduler_with_trace.py --trace_file <trace> --policy shockwave
  --throughputs_file <tacc_throughputs.json> --cluster_spec 4:0:0 --num_gpus_per_server 1:1:1
  --seed 0 --solver GUROBI --time_per_iteration 120 --config <shockwave_config.json>
  --pickle_output_dir <output>
```

and reads `makespan`, `avg_jct`, `jct_list`, `finish_time_fairness_list` and
`per_round_schedule` from `<output>/shockwave_<trace-name>_simulation.pickle`.

## Correctness

Any of these on any trace makes the whole evaluation score 0:
- the gRPC stub build (the `protoc` command of `frontier-build.sh`, 120 s limit) fails;
- the driver runs longer than 240 s, or exits nonzero (the untouched starter raises
  `NotImplementedError`);
- the result pickle is missing or malformed;
- not every job completes: `jct_list` and `finish_time_fairness_list` must each have exactly 8
  values, and `per_round_schedule` must be nonempty;
- an allocation round is invalid: each round must be a nonempty dictionary whose keys are job IDs
  0-7, each job gets exactly its `scale_factor` distinct workers from 0-3, at most 4 workers are
  used, and no worker appears twice;
- makespan, average JCT, any JCT or any rho is negative or non-finite.

## Workload

One workload is one eight-job trace, simulated once on four V100 workers (one GPU per server)
with the same throughput table, configuration, seed `0` and 120-second rounds.

| suite | traces | adaptation per trace (accordion / gns) |
|---|---|---|
| train | 2 | 3/5, 4/4 |
| final | 2 | 4/4, 5/3 |

- The four traces are disjoint, content-distinct selections from the project's canonical
  synthetic dynamic trace, each with arrivals shifted to start at 0; they share no record. Each
  has four 1-GPU, two 2-GPU and two 4-GPU jobs.
- The suites are stratified by duration and arrival pattern. Across all traces, record
  durations are 1,069-16,778 s and arrivals 0-2,050 s; files are 1,213-1,295 bytes.
- The throughput table (TACC measurements, 1,239,492 bytes) has 83 records under each of `k80`,
  `p100`, `v100`, `v100_unconsolidated`, `p100_unconsolidated` and 77 under
  `k80_unconsolidated`; only `v100` is used.
- Traces, the table and the configuration file are not in the Agent image.

## Scoring

Per trace `w`, with `M` the makespan (s), `J` the average JCT (s), `R` the maximum of the eight
job rho values and `U = round(100 * count(rho > 1.05) / 8, 2)`:

```text
S_w   = 40 * 12000 / (12000 + M) + 40 * 7000 / (7000 + J)
      + 15 * 0.55 / (R + 0.05) + 5 * (1 - U / 100)
score = (S_1 + S_2) / 2          # unrounded, over the suite's 2 traces
```

A job's rho is its JCT / (isolated per-epoch runtime sum * max(1, 8/4)), rounded to five
decimals, so `R >= 0.5` (fairness term maximum 15).

Anchors: the untouched starter scores 0. A serial FIFO policy that returns only the first live
job scores `29.580852` on train.

**Beats the reference** when, on both final traces, all four of makespan, average JCT, worst rho
and unfair-job percentage are strictly lower than the withheld reference policy's values: 8
comparisons, no tolerance; a tie or 7 of 8 fails. This criterion does not change the score.

## Running locally

```sh
cd /workspace/repo && bash frontier-build.sh                # generate the gRPC stubs
bash frontier-bench.sh --trace_file ... --policy shockwave ...  # the driver, arguments as above
```

`frontier-bench.sh` runs the driver from `scheduler/` with your arguments. Write your own trace,
throughput and configuration files in the formats above; the loader writes a pickle next to the
trace, so its directory must be writable.

## Environment

Python 3.12 on CPU with cvxpy 1.9.2, gurobipy 13.0.3, mosek 11.2.3, numpy 2.5.2, scipy 1.18.1 and
grpcio-tools 1.83.1. The judge runs each trace with `PYTHONHASHSEED=0` and `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1.

## Notes

The 240 s limit covers all policy and solver time of one trace. The simulator (fixed seed,
virtual time) is deterministic for a deterministic policy.
