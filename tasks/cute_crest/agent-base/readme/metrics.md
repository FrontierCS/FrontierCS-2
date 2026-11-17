- `dev_branches_covered_mean`: Mean over the 30 development runs of the branches covered when
  the run ended.
- `dev_coverage_curve`: List of [iteration, mean branches covered] at fixed checkpoints over the
  30 development runs; a run that ended before a checkpoint counts with its final coverage.
- `dev_search_overhead_seconds_mean`: Mean over the 30 development runs of the search overhead
  (strategy CPU plus constraint-solving CPU), in seconds.
- `dev_solver_attempts_mean`: Mean over the 30 development runs of the constraint-solver
  attempts, feasible and infeasible.
- `dev_runs_ended_early`: Number of the 30 development runs that ended before the iteration
  budget was spent (a cap was reached or Run returned).
- `dev_killed_executions_mean`: Mean over the 30 development runs of the program launches that
  crashed or hit the per-launch time limit.
