# EVM State Checkpoints

Replace the checkpoint-capable EVM stack in Spectrum's execution-state header with a faster
one. A valid backend restores the saved code position, stack height and every 256-bit item,
leaves every other live checkpoint intact and reclaims its storage. The score rewards higher
committed throughput and lower checkpoint creation time, restoration time and retained memory
than the backend the header ships with.

## Task

Spectrum (Zhang et al., "Spectrum: Speedy and Strictly-Deterministic Smart Contract
Transactions for Blockchain Ledgers", PVLDB 17; C++20, upstream commit `e67c73cc`) executes a
consensus-ordered stream of EVM transactions concurrently while preserving their serial order,
and rolls back EVM bytecode position and operand stack from checkpoints after conflicts. The
header holds a working but slow checkpoint backend; supply a better one.

## What you may change

- Only `lib/evmcow/execution_state.hpp`: the definitions of `StackTop` and `Checkpoint`, their
  private state, and the bodies for construction, stack access, mutation, copying, moving,
  destruction, checkpoint creation and restoration.
- The public declarations used by Spectrum's sources and by the judge's correctness driver must
  keep compiling with their semantics: those in the Interface section and the members the driver
  uses (below).
  Everything outside the header (EVM instructions, transaction processing, workload generation,
  statistics formatting, build configuration) is fixed; no file may be added or deleted.
- Added lines must not contain, case-insensitively: `#define`, `malloc`, `calloc`, `realloc`,
  `aligned_alloc`, `mmap`, `sbrk`, `operator new`, `operator delete`, `Statistics`,
  `checkpoint_bench`, `FRONTIER_`, `mallctl`, `jemalloc` (rejected as
  `content_policy`). Allocate through `new`/`delete` so that retained memory is measured. The
  judge runs the programs built from the header without new processes (threads only),
  environment or access to its files.

## Interface

- `StackTop(uint256* begin, uint256* end)` constructs an empty stack over the given stack space;
  default construction must stay valid. Public `StackTop::height` is the item count.
- `push(value)` adds an item (also when `value` refers into the same stack), `pop()` removes and
  returns the former top, `top()` returns the top.
- `get(index)` and `operator[](index)` return a const reference to the item `index` below the
  top (0 = top); `get_mut(index)` a mutable reference to it. References from `pop()`, `top()`,
  `get()`, `get_mut()` and `operator[]` must stay usable across later `pop()`/`get_mut()` calls
  in the same instruction.
- `ExecutionState::save_checkpoint()` returns a copyable, movable `Checkpoint` holding
  `position.value().code_it` and the whole stack; `load_checkpoint(const Checkpoint&)` restores
  code iterator, height and every value, visible to every access path at once.
- Creating, loading, copying, moving or destroying one checkpoint must not change another; each
  stays valid until destroyed. After a load and later stack changes, only checkpoints created
  before it are loaded again.
- The driver creates the state as `new ExecutionState()`, sets `state->stack_top =
  StackTop(&state->stack_space.m_stack_space[0],
  &state->stack_space.m_stack_space[state->stack_space.limit])` and `state->position =
  Position{code + i}`, and keeps `std::vector<Checkpoint>`; these members must stay.

```cpp
// bottom-to-top stack [10, 20, 30]
state.position = Position{code + 7};
auto saved = state.save_checkpoint();
state.stack_top.get_mut(1) = 99;  // [10, 99, 30]
state.position = Position{code + 12};
state.load_checkpoint(saved);     // [10, 20, 30], position code + 7
```

## Correctness

Everything below must hold, or the evaluation scores 0:

- The header compiles in the correctness driver (`g++ -O3 -DNDEBUG -std=c++20`, 60 s) and in
  Spectrum (`cmake -S <tree> -B <tree>/build -DNDEBUG=ON`, then `cmake --build <tree>/build
  --target bench -j 4`; 600 s per measured build).
- Each driver trace (`<driver> 400 <seed>`, 30 s) exits normally and prints six integers:
  create ns, restore ns, retained bytes per checkpoint, live checkpoints (must be 16), checksum,
  end-to-end ns per round. Create, restore and end-to-end must be in `(0, 1,000,000,000)`
  ns, retained bytes in `(0, 67,108,864]`, and tracked live allocation must return to zero.
- Each Spectrum run (`bench Spectrum:2:9973:CHECKPOINTED <record> 1s <seed>`, 20 s) passes
  every restore check (height, code position, `top()`, every item) and prints
  `commit <positive> tx/s` and `checkpoint create_ns <x> restore_ns <x> retained_bytes <x>
  creates <n> restores <n> retained_samples <n>` on standard error, with every value positive.

## Workload

| Suite | Spectrum records | Seeds |
| --- | --- | --- |
| train | `YCSB:10000:0.9`, `Smallbank:10000:0.9` | 1907, 3253 |
| final | `YCSB:15000:0.9`, `Smallbank:15000:0.9`, `TPCC:100:5` | fixed, not disclosed |

Records are `workload:domain:parameter`: YCSB and SmallBank take a key/account-domain size and
a Zipf exponent; `TPCC:100:5` has item domain 100 and five order lines
per NewOrder, with warehouse and district fixed to 0. Calls are generated in memory from the
record and seed for one second.

A workload is one Spectrum record (five interleaved runs, medians) or one checkpoint-micro
workload per suite seed (five traces; the first enters the score, all five the beat check):
train has 4 workloads, final 6. A trace first checks rollbacks at heights 1, 16, 64, 65, 200
and 1,000, then runs 400 rounds of: push 1,000 values; 16 checkpoints, each timed as
`save_checkpoint()` plus storing it (create time) and, with its 24 indexed writes, as
end-to-end time; copy- and move-construct one checkpoint and restore it (`load_checkpoint()`
alone is the restore time; with the 64-access mix after it, end-to-end time); a full oracle
check; a restore of an older checkpoint with its access mix (end-to-end); erasure in seed
order. Throughput is rounds per second of end-to-end time. Retained bytes are
`ceil(peak new-tracked requested bytes / 16)`, tracked from just before `new ExecutionState()`
(inline stack space, nested `new`/`new[]` and the vector of 16 checkpoints count, a shared
allocation once), maximum over rounds.

## Scoring

```text
for every workload w:
  Q_w = T_w / (M_w * sqrt(C_w * R_w))^(1/3)
  (T throughput: Spectrum median TPS, micro rounds/s of the first trace;
   C, R create/restore ns; M retained bytes per checkpoint; Spectrum: medians over runs)
Q_C       = exp(mean_w ln Q_w)
unbounded = 100 * log10(Q_C / Q_B)        (0 unless Q_C > 0 and Q_B > 0)
score     = 100 * tanh(max(0, unbounded) / 136)
```

`Q_B` is the shipped backend measured in the same evaluation. Matching it scores 0; 2x, 10x and
100x composite quality score 21.8, 62.6 and 90.0. Changes under about 4 unbounded points are
within timing noise.

**Beats the reference** when, on every workload, it has `correctness` and, for each of
`committed_tps` (throughput), `checkpoint_create_ns`, `checkpoint_restore_ns` and
`retained_bytes_per_checkpoint`, all five candidate measurements strictly better than all five
reference measurements (minimum candidate throughput above the maximum reference
throughput; maximum candidate value below the minimum reference value for the other three).

## Running locally

Before configuring, copy the prebuilt dependencies (see `/frontier/task/README.md`), then:

```sh
cd /workspace/repo && cmake -S . -B build -DNDEBUG=ON && cmake --build build --target bench -j 4
d=build/_deps; LD_LIBRARY_PATH=$d/fmtlib/fmt/install/lib:$d/gflags/gflags/install/lib:$d/google/glog/install/lib:$d/jemalloc/jemalloc/install/lib \
  build/bin/bench Spectrum:2:9973:CHECKPOINTED YCSB:10000:0.9 1s 1907
```

## Notes

The records and seeds stand for the three workload families: do not special-case the listed
records or seeds.
