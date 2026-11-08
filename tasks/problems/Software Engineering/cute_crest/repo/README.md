# Search Strategy for Concolic Testing

Implement the search strategy of the CREST concolic tester: the policy that decides which
condition of which earlier execution to negate next. Cover as many branches as possible within
4000 program executions per run while spending little CPU time on the search. The score
measures coverage; beating the reference needs both more coverage and less search overhead.

## Task

A concolic tester runs an instrumented program on concrete inputs, records the input-dependent
conditions along the path taken, and negates one of them, solving the constraints, to get inputs
for a different path. Each step costs one execution. The tester is CREST (Burnim and Sen,
"Heuristics for Scalable Dynamic Test Generation", ASE 2008); its search strategies have been
replaced by an interface, and you implement one strategy behind it. The reference is the paper's
control-flow-directed search.

## What you may change

- Only ASCII `.cc`/`.h` files directly in `src/frontier_solution/` (names `[A-Za-z0-9_]+`, ≤ 32
  files, at most 256 KiB each), all `.cc` linked; `strategy.cc` must define
  `frontier::MakeStrategy()`. No deletions.
- Includes: `"strategy/strategy_api.h"`, your own files, and only `<algorithm> <array> <bitset>
  <cassert> <cfloat> <climits> <cmath> <cstddef> <cstdint> <cstdlib> <cstring> <deque>
  <functional> <initializer_list> <iterator> <limits> <list> <map> <memory> <new> <numeric>
  <optional> <queue> <set> <stack> <string> <tuple> <type_traits> <unordered_map>
  <unordered_set> <utility> <vector>` and `<assert.h> <float.h> <limits.h> <math.h> <stddef.h>
  <stdint.h> <stdlib.h> <string.h>`, each line exactly `#include <h>`/`"h"` plus an optional `//`
  comment; directives only `#include`, `#define`, `#undef`, conditionals.
- Anywhere in the file, including comments and strings: no line splice (`\` at end of line),
  `??`, `%:` or `R"`. No token pasting (`##`).
- Outside comments and literals, no identifier `asm`, `__asm`, `__asm__`, `__attribute`,
  `__attribute__`, `__declspec`, `syscall`, `_Pragma`, `thread_local`, `__thread`, nor any
  starting `__builtin`, `__sync` or `__atomic`.
- Objects may reference only the strategy interface, allocation, C string/math helpers and
  standard-container/exception support (no I/O, time, processes, threads, environment or `rand`).
- Everything in a namespace or `static` (no global-namespace variable or `extern "C"` function);
  no namespace-scope object with a constructor or dynamic initialiser; nothing runs before
  `MakeStrategy()`.

## Interface

`src/strategy/strategy_api.h` is the whole interface. You implement:

```cpp
namespace frontier {
class Strategy {
 public:
  virtual ~Strategy() {}
  virtual void Run(Host& host) = 0;
};
Strategy* MakeStrategy();   // define this in src/frontier_solution/
}
```

`Run` drives one run through the `Host`:

```cpp
const Execution* Host::Restart();
const Execution* Host::Negate(const Execution* e, size_t k);
void             Host::Release(const Execution* e);
const Program&   Host::program() const;
int  Host::budget() const;           int  Host::iterations_used() const;
bool Host::covered(int branch) const; int Host::num_covered() const;
uint32_t Host::Random();             uint32_t Host::Uniform(uint32_t n);
```

- `Restart()` runs the program on fresh random inputs and returns the execution. It costs one iteration.
- `Negate(e, k)` keeps the first `k` constraints of `e`, negates constraint `k`, solves and runs the program on the solution: one iteration, returning the new execution, which may not follow the predicted path. If infeasible it returns `NULL` at no iteration cost; a constraint identical to an earlier one on the path is answered infeasible without a solver attempt.
- `Release(e)` gives an execution back. At most 64 executions may be held at a time.

An `Execution` is what the strategy sees of a run of the program:

```cpp
struct Execution {
  int id;
  bool completed;                        // false: killed or crashed before reporting a trace
  std::vector<int> path;                 // branch ids in order, with kCallId (-1) and kReturnId (-2) markers
  std::vector<uint32_t> constraint_pos;  // positions in `path` of the input-dependent branches
};
```

Only constraint branches can be negated; the strategy never sees the constraints, inputs or solver. `Program` (see the header) gives the branch ids, `paired[b]` (the other arm), `function_of[b]` and `successors[b]` (branch-level CFG; calls followed, returns not). Ids are relabelled with a fresh secret permutation every run, so only structure carries over. `Host::Random()`/`Uniform(n)` are the only randomness, seeded per run.

Example: a strategy that negates the last constraint of each new execution:

```cpp
void Run(Host& host) {
  const Execution* e = host.Restart();
  while (true) {
    if (e->constraint_pos.empty()) { host.Release(e); e = host.Restart(); continue; }
    const Execution* n = host.Negate(e, e->constraint_pos.size() - 1);
    host.Release(e);
    e = n ? n : host.Restart();
  }
}
```

When the iteration budget is spent, `Restart()` or `Negate()` does not return and the run ends. If `Run` returns earlier, the run ends with the coverage reached so far.

## Correctness

The strategy runs in its own process with 1 GiB of address space and no file system, clock,
network, environment or child processes; a forbidden system call ends it. Each of these scores
the submission 0 and passes nothing:
- a rule of "What you may change" is broken (reported as a rejection);
- the strategy does not compile or link;
- in any single run, a strategy fault: a crash, a forbidden system call or running out of memory,
  an invalid argument to the `Host` or more than 64 executions held, a protocol fault, or an
  unexpected exit of the strategy process.

Ending a run early (a cap below, or returning from `Run`) is not a fault: the run keeps the
coverage it reached. A failure of the reference or of the judge is an evaluation failure without
a score.

## Workload

One run is one process of the search harness on one program with one seed:
- The budget is 4000 iterations; one iteration is one launch of the program under test.
- Every launch is limited to 10 seconds and 2 GB of address space. A killed or crashed launch
  still costs its iteration and contributes only the coverage it reported before stopping.
- Coverage is the number of distinct branches executed over the whole run.
- Search overhead is the CPU time of your strategy process plus the CPU time the harness spends
  solving constraints for your `Negate` requests; executing the program under test is excluded.

Safety caps end a run early and keep it valid: wall-clock time 320 seconds on the development program
(each final program has its own budget); search overhead up to that wall budget; 160,000 solver
attempts (feasible and infeasible); 225,000 `Negate` requests.

| suite | programs | runs |
| --- | --- | --- |
| development (train) | GNU grep 2.2, 3982 branches (in your workspace) | 30 seeds × (candidate + reference) |
| final | two other real C programs, unnamed and not in your workspace; one is an order of magnitude larger | 30 seeds each × (candidate + reference) |

Candidate and reference run on the same seeds with the same secret relabelling, interleaved on
the same machine, up to 8 runs at a time. Development runs are reproducible; one final program is
not. A strategy tuned to the development program's ids or shape will not transfer.

## Scoring

Per graded program, with `C` the candidate's mean coverage over the 30 seeds, `B` the fixed mean
coverage of bounded depth-first search (depth 12) on that program, and `T` its branch count:

```text
program_score = 100 * max(0, C - B) / (T - B)
score         = mean of program_score over the graded programs
```

For the development program `B` = 1057 and `T` = 3982. A strategy no better than bounded DFS
scores 0; covering every branch would score 100; realistic strategies stay far below 100. The
score uses no reference measurement and no search overhead.

**Beats the reference** when, on every graded program, both of these hold, with `cov`, `ovh` the
per-seed coverage (branches) and search overhead (whole milliseconds) of candidate `c` and
reference `r` over the same 30 seeds:

```text
branches_covered:        mean(cov_c) - mean(cov_r) > 3 * s_cov * sqrt(2 / 30)
search_overhead_seconds: mean(ovh_r) - mean(ovh_c) > 3 * s_ovh * sqrt(2 / 30)
s^2 = (var(candidate) + var(reference)) / 2   (sample variances over the 30 seeds)
```

Exact integer arithmetic; equal means or a difference inside the margin fail. No program or
measurement offsets another.

## Running locally

```sh
cd /workspace/repo
./frontier-build.sh              # builds the tester and instruments grep 2.2 (about a minute, once)
python3 run_local.py --seeds 1-8 # runs your strategy under the judge's limits and source/object checks
```

`run_local.py` prints, per seed, the branches covered, the search overhead and how the run ended.
The grep 2.2 source is at `/frontier/task/benchmark-src/grep-2.2`.

## Environment

The judge's search machine has 8 CPUs and 32 GiB of memory, and each run's strategy process runs
there in an isolated instance of its own; your workspace has 8 CPUs and 16 GiB. CREST is pinned
at commit 3e5ececd with offline OCaml 4.02.3 and Z3 4.15.3.0. A development evaluation of a
strategy in the reference's cost range takes about 15 minutes, a final evaluation about an hour.

## Notes

For scale: with the reference's own spread, the coverage margin was about 2 % of its coverage on
the development program and 2 % to 4 % on the final programs, and the overhead margin about 7 %
(development), 5 % and 30 % (final). Margins come from each evaluation's own runs, so they also
move with your strategy's spread.
