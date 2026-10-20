# Native queue integration revision - 2026-09-12

This is source integration into the existing cand_000 after native Stage6's
same-interpreter report-forgery finding. The previous candidate is retained in
`checkpoints/queue-native-integration-r1-20260912/task-before` under the native
run. The published task version, historical
reference results and pipeline state are preserved. No model, build, GPU or
Evaluation is run by this source revision.

## Execution boundary

The runner restores `reference.patch` into a trusted upstream spine and still
requires its complete exact source lock. It validates each submitted tree
against that spine, admitting changes only to `frechet_distance/queue.py` and
`frechet_distance/frontier_solution/`. Only those files enter a separate source
snapshot. Neither submitted tree nor snapshot enters the parent import path or
working directory. Reports, checkpoints, model assets and evaluation code stay
outside the actor's namespace.

Both candidate and reference phases launch `queue_observer_entry.py`. It checks
the trusted spine and observer, installs a trusted FeatureQueue proxy before
any upstream import in both train and eval, then invokes the original observer.
The parent retains generation, representation extraction, loss, optimizer,
checkpoint ownership, held-out scorer and all validity decisions. The actor
provides bounded tensor-only queue operations and first-order vector products.
Queue checkpoint restoration closes each temporary worker after checking state.

`judge_deps/queue_runtime.json` pins 16 private source/tool files. Fourteen are
byte-identical to the accepted r3 package, including actor, TSYNC filter, wire,
proxy, namespace, process manager, bwrap and libseccomp. The two glue changes
make snapshot preparation import Torch lazily and route both observer phases
with private sibling imports. The entry also binds its lifetime to torchrun
because Torch starts it in a separate process session, and preserves bounded
worker stderr only in private failure diagnostics. A focused local parent-death
test verifies the observer is killed and reaped after its launcher dies.
The r3 private CUDA result is pinned by SHA256
`2eb01fc638db6ec01a6b0f562e1e06191b19179955410deff966a1347a0293d1`.
It passed snapshot, EMA and full 50000-by-2048 forward/VJP/enqueue/state/restore
plus a nonreference isolation case. This evidence qualifies that private
boundary only; the integrated image and full scientific route are pending.

## Preserved science and next validation

The complete public substrate, reference patch, config, beat specification,
assets, workload definitions/seeds, private evaluator and scoring evaluator are
unchanged. This includes N50000, D2048, two development and three final
workloads, 128/512 updates, 4096 held-out images, all six strict metrics and the
every-workload conjunction. Scientific command argument construction is exact.

Local tests validate nonreference snapshot admission without executing it,
fixed-context and symlink rejection, r3 source pins, both train/eval proxy import
routes, complete paired command generation, existing guards and feedback. Mock
transport and reports are explicitly structural evidence, not science.

Next: separately allocated native image/reference preflight on a new immutable
revision, reusing retained toolchain/model layers. Capture actual device/count,
runtime UID, workload reports, full memory use, elapsed time, worker closure
and train/final results. No automatic solver invocation, repeat reference trial
or required beating witness follows from this checkpoint.

## Stability finding

The wording is corrected; the measurement is not. Historical data contains one
development/train result (`2.3138400099376395`) and one final result
(`3.7359483147422274`), with `beats_reference=0` for both. These are single
samples, not repeats, and zero sample spread was removed as a repeatability
claim. In the recorded identical-code final pair, candidate `3.7359483147422274`
and reference `3.681866` differ by about `0.054082` (about `1.47%`); 15 of 18
strict per-workload metric comparisons favored the first execution. No tolerance
was added and the strict six-metric, every-workload conjunction is unchanged.

This pair demonstrates variation relevant to the strict comparisons; it does
not estimate a distribution or justify new tolerances. Future measurements must
resolve a named uncertainty, such as whether the same-code per-metric variation
can dominate a claimed improvement under the unchanged conjunction. Automatic
repeated reference trials and a passing solver are not requirements of this
source revision. No further measurements were authorized or run here. The
stability finding is **documented accurately but remains open**.

## Bounded private runtime evidence

The trusted runner captures each phase report before disposable directories close. A private 192 KiB journal (32 events; 16 KiB per report; 4 KiB failure log tail) survives into the native Verifier evidence metrics.json, including scored-zero diagnostics. It is never passed into observer/actor environments or returned as solver feedback. Reports retain process identity, device/runtime information and queue-worker closure. CUDA allocator peaks and RSS are parent-only observations, not whole-device/process-tree totals. Interrupted phases retain the last trusted partial report and actual error type. Full device resource sampling belongs to the future reference controller. Scientific scores, validity checks and thresholds are unchanged.
