# Register-aware Branch Prediction

Supply the missing register-value component of a TAGE-SC-L style conditional branch predictor
(RUNLTS) so that the cycle-aware simulator counts fewer branch mispredictions and fewer wrong-path
cycles per 1,000 instructions, within a fixed 192 KiB storage budget.

## Task

The CBP 2025 simulator models fetch, decode and execution over instruction traces. The predictor
returns one direction per conditional branch, keeps its state through the supplied callbacks, and
may use only what those callbacks have supplied at that point. The register-value contribution to
the statistical corrector (RUNLTS) has been removed; implement it inside the retained predictor
(its design is yours).

## What you may change

- `my_cond_branch_predictor.cc`, only inside the nine regions below; every byte between them
  (whitespace and line endings included) is protected. Intervals are zero-based UTF-8 byte
  offsets in the pristine file (end excluded, SHA256
  `ce35248019bedb52575c390232f9615b776bd149143589935570296cdeac9bfd`); after an edit, find later
  regions by their anchors, kept intact with their braces.
- New files under `solver_extensions/`, included from those regions with quoted
  `#include "solver_extensions/..."`. Every other file is fixed.

| Region | Pristine byte interval | Stable location | Contract |
| --- | --- | --- | --- |
| `state` | `[437, 437)` | Immediately after `namespace nRUNLTS {` and its two newlines; before `static constexpr int LogMaxNewlyCounters = 16;`. | Namespace-level component types, constants and fixed state. Do not change retained predictor definitions. |
| `snapshot_fields` | `[19653, 19653)` | Inside `struct cbp_hist_t`, after `int perceptron_sum_at_prediction;` and before the blank line preceding `// Begin Conventional Histories`. | Additional component checkpoint fields. One active copy counts toward storage; genuine in-flight copies use the existing exemption. |
| `storage` | `[22895, 22895)` | After `inter += 2 * CONFWIDTH; // the 2 counters in the choser` and its following blank line; before `printf("(TotalSC %d) ", inter);`. | Optional declarations only under the same access rules. Leave empty if unused: the independent audit determines storage and no change to the printed total is required. Retained `inter` is not a writable output. |
| `helpers` | `[27449, 27449)` | Inside `RUNLTS`, after `} // end init_histories` and its following blank line; before `// the index functions for the tagged tables uses path history as in the OGEHL predictor`. | Component member state and helper methods. New helpers belong here or in an included utility file, not between the protected callbacks. |
| `prediction` | `[34223, 34275)` | After the assignment to `pv.pred_inter` and its following blank line; before `// Compute the SC prediction`. | Replace the inactive `int extra_weight = 0; pv.LSUM = 0;` initializer. Produce `pv.LSUM` and a local `int extra_weight` for the retained corrector. Read only currently available callback/history inputs and your component state. |
| `training` | `[45566, 45566)` | After `IMLI_components.update(pv.LSUM, resolveDir, PC, hist_to_use.BrIMLI, hist_to_use.TaIMLI);` and its following blank line; before the closing braces preceding `void TAGE_allocation`. | Component training inside the existing corrector-update condition. `resolveDir` is available here; preserve the surrounding training condition and order. |
| `snapshot_capture` | `[49565, 49565)` | Immediately inside `bool predict(uint64_t seq_no, uint8_t piece, UINT64 PC)`, before `pred_time_histories.emplace(...)`. | Populate your additional fields of `active_hist` before the retained checkpoint copy. Preserve prediction/checkpoint ordering. |
| `decode` | `[50881, 50881)` | Only the body of `void decode_notify(uint64_t seq_no, uint8_t piece, uint64_t dst_reg)`. | Observe this destination overwrite using the given identity. No executed value is supplied here. |
| `execute` | `[50983, 50983)` | Only the body of `void execute_notify(uint64_t seq_no, uint8_t piece, uint64_t dst_reg, uint64_t value)`. | Observe the executed scalar value and producer identity; update only component state. Preserve protection against superseded producers. |

Retained names usable where already in scope: `PC`, `seq_no`, `piece`, `dst_reg`, `value`, `resolveDir`, `hist_to_use`, `active_hist`, `pv`, `LSUM`, `extra_weight`, `pred_time_predict`, `MYRANDOM`, `ctrupdate`; their existing fields stay protected. Retained inputs are read-only: write only new component state, `pv.LSUM` and your local `extra_weight`. A resolution label must not influence an earlier prediction.

Calls: your own new helpers, plus only `min`, `max`, `abs`, `labs`, `llabs`, `fill`, `at`, `size`, `begin`, `end`, `std::array` construction/assignment/index operators, ordinary assertions and the retained `MYRANDOM`/`ctrupdate`. Use checked `std::array::at`. `solver_extensions/` files obey the same rules. Declare types and functions before use.

Rejected (in the regions and `solver_extensions/` files, comments included
where stated): a backslash, `??`, `%:` or a raw string literal anywhere; a comment, string or
character literal not closed inside its region or file; any preprocessor directive other than a
quoted include of an existing `solver_extensions/` file; outside comments (string literals
included), any identifier starting with `__` or with `exec` (a prefix: `executed` is rejected
too), `template`, and the whole words `asm` `extern` `thread_local` `volatile` `reinterpret_cast`
`const_cast` `new` `delete` `malloc` `calloc` `realloc` `free` `printf` `fprintf` `sprintf` `puts`
`fopen` `open` `read` `write` `system` `fork` `dlsym` `syscall` `mmap` `getenv` `ifstream`
`ofstream` `fstream` `cout` `cerr` `clog` `exit` `abort` `longjmp` `signal` `raise` `atexit`
`dlopen` `memcpy` `memmove` `memset` `vector` `unordered_map` `function` `shared_ptr` `unique_ptr`;
templates, lambdas, raw pointers or arrays, raw-array indexing, address-of or dereference,
C-style/const/reinterpret/pointer/reference casts, dynamic allocation, mutable reference aliases,
functions returning a pointer or reference, compiler builtins, and unresolved, indirect or
non-listed external calls; references to retained names outside the list or writes to retained
state other than `pv.LSUM` and `extra_weight`; persistent state whose logical width cannot be
audited (use fixed-width scalars, bit-fields, structs of those and `std::array` with a constant
extent; no unions, inheritance or empty types). Changing bytes outside the regions, or another
retained file, is also rejected.

## Interface

The evaluator builds with `make -j1 'OPT=-O3 -D_GLIBCXX_ASSERTIONS'` and runs `./cbp <trace.gz>`
(entrypoint `lib/cbp.cc`), no flags. Callbacks (`u64 = uint64_t`, `u8 = uint8_t`):

```cpp
void setup();
bool predict(u64 seq_no, u8 piece, u64 PC);
void update(u64 seq_no, u8 piece, u64 PC, bool resolveDir, bool predDir, u64 nextPC);
void history_update(u64 seq_no, u8 piece, u64 PC, int brtype, bool pred_dir, bool resolve_dir, u64 nextPC);
void TrackOtherInst(u64 PC, int brtype, bool pred_dir, bool resolve_dir, u64 nextPC);
void decode_notify(u64 seq_no, u8 piece, u64 dst_reg);
void execute_notify(u64 seq_no, u8 piece, u64 dst_reg, u64 value);
void terminate();
```

`predict` returns `true` for taken. `seq_no` is the simulator's micro-operation number
(`num_uop`), not a PC or record index; `piece` counts the cracked pieces of one instruction from
zero (so pieces also have successive `seq_no`). Use the `(seq_no, piece)` pair to match
prediction-time state and producer notifications.

Order: `setup` first, `terminate` last; each conditional prediction is immediately followed by
the retained speculative history update; a destination's `decode_notify` precedes that producer's
execution, but instructions execute out of program order; at execution the eligible destination
value is forwarded first, then `update` is called for a conditional branch; branches without
destinations get no `decode_notify`/`execute_notify`; within a cycle due decode,
address-generation, execution and retirement events run in that order. Do not assume an older
producer has executed when a later branch is predicted, and never let an older producer re-enable
a value superseded by a newer overwrite.

Register names 0–30 are general registers, 31 the stack pointer, 32–63 SIMD, 64 the flags, 65 the
zero register; only 0–64 are forwarded. A 16-byte SIMD output becomes two `uint64_t` callbacks (low
then high), each with its register name and own `(seq_no, piece)`. Candidate code must not read
traces.

## Correctness

A run scores 0 when the patched source has C++ errors in the audit's C++17 parse, the declared
logical storage exceeds 1,572,864 bits (192 KiB, including RNG, thresholds, register state,
tables and selectors; genuine in-flight history copies keep the original checkpoint exemption;
storage is audited from declared widths, not printed totals), the source audit exceeds its 60 s or 4,096 MiB limit,
the build fails or exceeds 180 s, a simulation fails, is killed by a resource limit or exceeds its
2,400 s watchdog, the output lacks exactly one well-formed copy of each original measurement
section with consistent counters, or the instruction counts differ from the trace's expected
counts. Do not change the retained predictor state or use future values, branch-label oracles,
trace-specific answers or private evaluator information.

## Workload

105 original CBP traces in six application categories (compress 8, fp 14, infra 16, int 37,
media 4, web 26). Each run processes a complete trace, warmup included, in a fresh process; the
measured suffix sums trailing whole epochs until their instruction count exceeds half of the full
count, with no reset at the midpoint.

| suite | traces |
|---|---|
| train | 24, a fixed category-stratified subset |
| final | all 105, rerun afresh (includes the train subset) |

Measured: integer counts (mispredictions, wrong-path cycles, instructions) from a deterministic
simulator; each simulation is one single-threaded process limited to 4,096 MiB of address space.
Treat traces as application workloads, not inputs to recognize.

## Scoring

```text
score = 100 / sqrt((1 + B) * (1 + W / 100))
```

`B` and `W` are the equal-weight means over the suite of the simulator's four-decimal printed
BrMisPKI and CycWpPKI. The score lies in [0, 100] and uses no reference. Anchor: the inactive starter component scores
about 31.3 on the train subset (mean BrMisPKI 2.864, CycWpPKI 163.4).

**Beats the reference** when, on **every** final trace, all three hold: `br_mis_pki`
(mispredictions per instruction) strictly lower than the private reference's and `cyc_wp_pki`
(wrong-path cycles per instruction) strictly lower, both compared as exact integer counters with
no tolerance (a tie fails); and `logical_storage_bits <= 1,572,864`. A one-call shift of the shared random stream moves one trace's mispredictions by
about 1% (up to 6.6%) and wrong-path cycles by about 0.7%, so winning every trace needs a clearly
larger improvement. A valid submission keeps its score when the criterion fails.

## Running locally

`make` builds `./cbp`; `./cbp <trace.gz>` runs a trace (the graded traces are private). `bash
make_submission.sh` writes the predictor diff to `solution.patch`.
