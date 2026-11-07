# BBQ bitmap traversal

Implement the missing `BBQLevelLX` logic in BBQ's Python-to-SystemVerilog priority-queue
generator so the generated pipeline keeps exact queue behaviour and reaches a higher packet rate
(operations per cycle times clock frequency).

## Task

BBQ (NSDI 2024) is a hardware integer priority queue for packet scheduling. The generator
(`hardware/bbq/generator/bbq.py`) emits synthesizable queue RTL for a pipeline of levels; the
bitmap-traversal level `BBQLevelLX` is yours. The priority-bucket stage carries a documented author
correction and stays fixed, as do the RAM/FIFO adapter and the simulation assertions. You may
change dependency tracking, forwarding, traversal, speculation and pipeline latency of the LX
level, while preserving the public generator interface and queue semantics.

## What you may change

- `hardware/bbq/generator/bbq_level_lx.py`
- new importable helper files under `hardware/bbq/generator/solver_extensions/`
- Every other file stays unchanged (neighbouring stages, testbench, adapter, `frontier_cpu/`).
  Do not submit generated RTL, objects, caches or binaries.

## Interface

`BBQLevelLX` keeps the class and method interface the generator (`bbq.py`, `bbq_level.py`) calls,
and produces RTL compatible with the supplied pipeline. The evaluator observes only external
responses, simulated cycles and the synthesized clock period. Design rules for the emitted RTL:

The generated design must be single-clock synchronous logic. Every register is
clocked by the positive edge of `clk`. Latches, asynchronous set or reset,
negative-edge or derived clocks, combinational loops, tristates and `x`/`z`
literals are not permitted. Do not emit synthesis attributes, compiler
directives or system tasks from your logic, and do not instantiate RAM or FIFO
primitives beyond those the supplied generator emits; register arrays you
declare are synthesized to registers and multiplexers and timed as logic.
Every RAM or FIFO primitive whose output the generated design reads must
survive synthesis: a design in which synthesis removes such a primitive is
invalid, because the timed netlist would no longer be the simulated circuit.
A primitive whose outputs nothing reads may be removed.
No signal may be driven from more than one place: assigning the same variable
in two `always` blocks, in an `always` block and a continuous `assign`, or with
both blocking and non-blocking assignments makes the design invalid (Verilator
reports `MULTIDRIVEN` or `BLKANDNBLK` as an error, and Yosys reports multiple
conflicting drivers). A design outside these rules has no clock period and its
workloads are invalid.

## Correctness

The generated queue supports enqueue, dequeue-min and dequeue-max. Preserve
priority selection, payload conservation, exact response counts, capacity,
empty/reset behavior and the upstream testbench contracts. Requests must be
handled correctly across adjacent operations and subtree/bucket transitions.
Adjacent same-bucket dequeues may follow the supplied stage's coloring order;
isolated FIFO and upstream regression behavior remain required. Dropped,
duplicated, incorrect or extra responses invalidate the affected workload.

The generated `bbq.sv` is checked as raw text, comments included: it is invalid if it contains the
whole words `DPI`, `force`, `release`, `bind`, `program` or `interface` (so a comment such as
`// release the slot` fails), any `` `include ``, `` `define `` or `` `undef `` (other than the
supplied `` // `define DEBUG ``), a module other than `bbq`, more than one `endmodule`, the text
`trace_tb`, an `import` other than `heap_ops::`, a system task other than `$clog2`, `$display`,
`$error`, `$finish`, `$bits`, `$signed`, `$unsigned`, `$countones`, `$onehot`, `$onehot0`,
`$isunknown`, `$size`, `$high`, `$low`, or more than 8,000,000 bytes. With comments stripped, only
`` `ifdef SIM ``, `` `ifdef DEBUG `` and `` `endif `` directives are allowed, and the words
`negedge`, `always_latch`, `specify`, `primitive` and `deassign` are rejected.

Per-step limits: generator 20 s; each Verilator compile 120 s; each upstream regression 10 s;
each trace run 15 s; sv2v 60 s (output at most 16 MB); each Yosys run 600 s (elaborated JSON at
most 256 MB).

A workload is invalid (scores 0 and fails both comparisons) when its responses are wrong or
missing, a response arrives more than 128 cycles after its request, the design breaks a design rule above (it then has no clock period), the build,
simulation or synthesis fails, or its operation count differs from the reference's. All workloads
fail when the enabled upstream queue regressions fail.

## Workload

One workload is a deterministic operation trace on one configuration (depth, bitmap width,
capacity, seed). The evaluator generates and compiles your RTL, runs the upstream regressions,
simulates the traces in Verilator and synthesizes the same RTL for its clock period.

| suite | workloads |
|---|---|
| train | 4 public cases in `frontier_cpu/public_cases.json` (depth 2–3, width 4–8, capacity 63–255) |
| final | the 4 public cases rerun plus held-out cases with independent seeds that extend the bitmap-width axis beyond any width in train; do not assume only the public shapes or seeds |

Timed: simulated cycles and the modelled clock period, not host compile or simulation speed.
The trace also contains untimed correctness probes and resets; only the timed requests count
toward cycles and latency. The period comes from sv2v, then Yosys `synth -flatten` and a fixed
ABC script mapping to 6-input lookup tables, timed by a fixed delay model; undriven or unmapped
nets in the netlist make the design invalid:

```text
period_ps = 300 + max over timing paths of (start offset + 500 * lookup tables on the path)
```

Paths start at register outputs, module inputs and RAM/FIFO outputs, and end at
register inputs (data, enable, synchronous reset), RAM/FIFO inputs and module
outputs. The start offset is 1000 ps at the output of a RAM without output
register, 500 ps at FIFO status flags and 0 elsewhere. Every structural path
counts, whether or not it can be exercised; there are no multicycle or false
paths. There is no placement, routing or random seed, so the period is a
deterministic function of the generated source.

## Scoring

```text
efficiency(w) = 100 * offered_span(w) * 800 ps / (cycles(w) * period_ps(w))   (0 if w invalid)
score         = mean of efficiency over the suite's workloads
```

`offered_span` is the offered schedule's length in cycles, `cycles` the completion span (pipeline
fill and drain included), 800 ps the model's period for one lookup table between registers.
Anchors: 100 is the offered schedule at zero latency and 800 ps; the reference scores about 26.7
on train.

**Beats the reference** when, on **every** workload, both hold: `throughput_ops_per_second`
(`count / (cycles * period)`) strictly higher than the reference's (equal fails), and
`latency_seconds` (worst response latency, `latency_cycles * period`) no greater than the
reference's. Exact integer comparisons, no tolerance.

## Running locally

```sh
cd /workspace/repo
debug_dir=$(mktemp -d "${TMPDIR:-/tmp}/bbq-debug.XXXXXX")
python3 frontier_cpu/runner.py --workdir "$debug_dir" --official
python3 frontier_cpu/timing.py --rtl "$debug_dir/build_2_4_63/bbq.sv" \
  --workdir "$debug_dir/timing_check" --width 4 --capacity 63
```

The runner prints, per public case, `period_ps`, `throughput_ops_per_second` and `latency_ps`
exactly as the evaluator computes them, with lookup-table, register and memory-bit counts;
`timing.py` times one generated file and shows its critical-path endpoints.

## Environment

Python 3 with toolz 1.0.0, Verilator 5.032, sv2v 0.0.13 and Yosys 0.69 (YoWASP); CPU only.
