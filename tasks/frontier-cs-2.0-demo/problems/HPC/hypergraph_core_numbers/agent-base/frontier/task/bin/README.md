# Hypergraph Core Numbers testbed usage

Scientific requirements are in [/frontier/task/repo/README.md](../repo/README.md). The [task entry](../README.md) defines workspace setup and archive creation. Use the Candidate-installed `frontier-submit`; this CPU task has no GPU helper or SSH access.

All changes belong below `/workspace`. `prepare_workspace.sh` creates `/workspace/repo` and commits its baseline. `make_submission.sh` creates `/workspace/payload/solution.patch`; it does not submit the tar. The tar must be uncompressed POSIX format, at most 262144 bytes, and contain exactly the regular file `payload/solution.patch` with safe relative paths.

`frontier-submit train` evaluates the two development workloads. Each accepted request consumes one of 100 calls, including verifier failures. Approved feedback contains the independent candidate runtime score, candidate-only median runtime, and aggregate beat-spec counts; reference values and per-workload identities stay private.

`frontier-submit store-final` saves one Driver-owned snapshot without training or ending work. A successful replacement supersedes it. There is no save-count limit. `frontier-submit final` irreversibly selects an explicit archive. If no explicit final exists, normal exit, answer timeout, or a trusted process crash may select the saved archive. Cancellation and platform failure do not. Final runs independently after cleanup on both feedback workloads and the two hidden graded workloads.

Budgets: formal answering 18000 seconds; separately generated short trials use 1800 seconds. Independent final has 1800 seconds. Train allows 100 calls and 30000 cumulative verifier seconds. Agent and Verifier each receive 8 CPUs and 4 GiB RAM; the evaluator fixes `OMP_NUM_THREADS=8` and `OMP_DYNAMIC=FALSE` for candidate and reference. Agent workspace is 2 GiB. Model access allows 1024 sessions with 64 MiB requests and 8 MiB responses. Total audit allowance is 32 GiB; verifier evidence is capped at 4 MiB. There is no required solver score or early-stop rule.
