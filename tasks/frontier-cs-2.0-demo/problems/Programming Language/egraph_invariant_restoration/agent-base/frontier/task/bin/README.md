# E-Graph Invariant Restoration testbed usage

Scientific requirements are in [/frontier/task/repo/README.md](../repo/README.md). The [task entry](../README.md) defines workspace setup and archive creation. Use the Candidate-installed `frontier-submit`; this CPU task has no GPU helper or SSH access.

All changes belong below `/workspace`. `prepare_workspace.sh` creates `/workspace/repo` and commits its baseline. `make_submission.sh` creates `/workspace/payload/solution.patch`; it does not submit the tar. The tar must be uncompressed POSIX format, at most 1048576 bytes, and contain exactly the regular file `payload/solution.patch` with safe relative paths.

`frontier-submit train` evaluates the three public development workloads. Each accepted request consumes one of 100 calls, including verifier failures. Approved feedback contains the independent candidate runtime score, candidate-only median runtime, and aggregate beat-spec counts; reference values and per-workload identities stay private.

`frontier-submit store-final` saves one Driver-owned snapshot without training or ending work. A successful replacement supersedes it. There is no save-count limit. `frontier-submit final` irreversibly selects an explicit archive. If no explicit final exists, normal exit, answer timeout, or a trusted process crash may select the saved archive. Cancellation and platform failure do not. Final independently re-evaluates the three public workloads plus the three hidden workloads after cleanup.

Budgets: formal answering 18000 seconds; separately generated short trials use 1800 seconds. Independent final has 1800 seconds. Train allows 100 calls and 120000 cumulative verifier seconds. Agent and Verifier each receive 8 CPUs and 4 GiB RAM; the evaluator fixes eight build jobs; each measured candidate and reference process uses one Rust test thread and one Rayon thread. Agent workspace is 4 GiB. Model access allows 1024 sessions with 64 MiB requests and 8 MiB responses. Total audit allowance is 32 GiB; verifier evidence is capped at 4 MiB. There is no required solver score or early-stop rule.
