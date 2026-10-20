# BBQ bitmap traversal

Implement the missing `BBQLevelLX` logic in the Python-to-SystemVerilog queue
generator. Your implementation must preserve integer-priority queue behavior
while improving the generated pipeline's packet rate: operations per cycle
multiplied by clock frequency.

## Problem
The working tree is `/workspace/repo`. You may modify
`hardware/bbq/generator/bbq_level_lx.py` and add importable helper files under
`hardware/bbq/generator/solver_extensions/`. Keep every other supplied file
unchanged. The generator entrypoint is `hardware/bbq/generator/bbq.py`.
Preserve the public Python interface and produce compatible synthesizable queue
RTL for the supplied pipeline. You may change the selected module's latency
and implementation; the evaluator observes external responses, cycles and the
synthesized design's clock period.

Use Python, toolz and Verilator for local CPU simulation, and the supplied
`frontier_cpu/timing.py` (sv2v and Yosys) for the clock-period estimate. No GPU,
vendor FPGA toolchain or physical FPGA measurement is required. The supplied priority-bucket
stage includes a documented author correction; it remains outside your editable
surface. The independent RAM/FIFO adapter and native SIM assertions are fixed.

## Correctness
The generated queue supports enqueue, dequeue-min and dequeue-max. Preserve
priority selection, payload conservation, exact response counts, capacity,
empty/reset behavior and the upstream testbench contracts. Requests must be
handled correctly across adjacent operations and subtree/bucket transitions.
Adjacent same-bucket dequeues may follow the supplied stage's coloring order;
isolated FIFO and upstream regression behavior remain required. Dropped,
duplicated, incorrect or extra responses invalidate the affected workload.

## Workload
The evaluator generates and compiles your RTL, runs the enabled upstream queue
regressions, and checks deterministic operation traces at multiple depths,
bitmap widths and capacities. For each configuration it also synthesizes the
same generated RTL and computes its minimum clock period. Public case parameters are in
`frontier_cpu/public_cases.json`. Final evaluation freshly reruns the public
cases and additional private cases. Do not assume only the public shapes or
seeds will be used. The graded suite is not the feedback suite: it includes
fresh public-case reruns plus held-out cases. The final suite extends the
public distribution along the bitmap-width scale axis, with independent seeds.
The graded suite includes bitmap widths beyond any in feedback.

## Scoring
The paper measures a hardware queue's performance as the product of two
factors: its maximum clock frequency, set by the critical path, and the number
of operations it can issue per cycle. This task measures both. Cycles come from
the Verilator trace. The minimum clock period comes from a fixed open-source
flow applied, for every graded configuration, to the same generated RTL that is
simulated: sv2v converts the SystemVerilog, Yosys synthesizes and flattens it,
a fixed ABC script maps it to 6-input lookup tables, and a fixed delay model
times the mapped netlist:

```text
period_ps = 300 + max over timing paths of (start offset + 500 * lookup tables on the path)
```

Paths start at register outputs, module inputs and RAM/FIFO outputs, and end at
register inputs (data, enable, synchronous reset), RAM/FIFO inputs and module
outputs. The start offset is 1000 ps at the output of a RAM without output
register, 500 ps at FIFO status flags and 0 elsewhere. Every structural path
counts, whether or not it can be exercised; there are no multicycle or false
paths. There is no placement, routing or random seed, so the period is a
deterministic function of the generated source. It is an estimate under this
model, not a vendor timing report or a physical FPGA measurement.

For every graded workload, success requires both:

- strictly higher throughput in operations per second,
  `completed operations / (cycles * period)`, than the reference; and
- no greater worst response latency in seconds,
  `worst latency in cycles * period`, than the reference.

These requirements apply separately to every workload. The measured finite
interval includes pipeline fill and drain; one input operation per cycle is the
interface ceiling. Matching the reference's throughput does not satisfy the
strict improvement requirement.

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

The continuous score is the mean of
`100*(offered_span*800 ps)/(completion_span*period)` over the graded workloads,
with invalid workloads contributing zero. It is the measured operations per
second as a percentage of the model ceiling: the offered schedule completed
with zero latency at 800 ps, the model's period for one lookup table between
registers. This score is separate
from the all-workload throughput/latency success condition. Host compile,
simulation and synthesis speed is not scored.
Maximise the continuous score while satisfying every required metric comparison.

## Design space
Within the selected module and helper namespace, you may change dependency
tracking, forwarding, traversal, speculation and pipeline latency while preserving
the public generator interface and queue semantics. Neighboring stages and
measurement code remain fixed.

## Calibration
Measurements use simulated cycles, externally observed responses and the
deterministic modelled clock period. Compile and synthesis wall time is a
resource cost, not score noise. A higher continuous score does
not imply that every per-workload reference comparison passes; inspect the
reported aggregate criterion progress as well.

## Development and submission
The submission budget is in `/frontier/task/README.md`. That count is rendered from
the task configuration. Development feedback provides public absolute metrics
and aggregate progress toward the throughput and latency criteria.

Work on the permitted source files, then run `/frontier/task/make_submission.sh` to create
`/workspace/payload/solution.patch` relative to the supplied starter tree. The final deliverable
is that unified text patch; it must contain only permitted source changes and
must apply cleanly to a pristine starter. Include new helper files in the patch.
Do not submit generated RTL, compiled objects, caches, binaries or edits to the
testbench, adapter, neighboring stages or evaluator. Keep build output outside
the submitted source tree.

For local public-case debugging, the supplied runner can build and exercise a
completed implementation:

```sh
cd /workspace/repo
debug_dir=$(mktemp -d "${TMPDIR:-/tmp}/bbq-debug.XXXXXX")
python3 frontier_cpu/runner.py --workdir "$debug_dir" --official
/frontier/task/make_submission.sh
```

For each public case the runner prints `period_ps`, `throughput_ops_per_second`
and `latency_ps`, computed exactly as the evaluator computes them, together
with lookup-table, register and memory-bit counts. To time one generated file
and see its critical path endpoints:

```sh
python3 frontier_cpu/timing.py --rtl "$debug_dir/build_2_4_63/bbq.sv" \
  --workdir "$debug_dir/timing_check" --width 4 --capacity 63
```

Use the evaluation/submission command provided by the
runtime for scored calls and final selection. The task does not expose private
reference values or private final-case details.

## Source provenance
This task is derived from BBQ with separately documented author corrections to the retained PB stage and selected LX reference. The starter retains the selected LX interface; its implementation is solver-owned. The reference is author-modified, not upstream-exact. CPU simulation and the modelled clock period establish neither physical FPGA performance nor vendor-library equivalence.
