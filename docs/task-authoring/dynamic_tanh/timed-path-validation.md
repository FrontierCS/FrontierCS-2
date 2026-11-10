# dynamic_tanh: numerical validation of the timed path (2026-10-12, gradient check revised)

Status: source change only. No image is built or published, no TaskVersion is
registered and no evaluation was started. **No GPU was available: everything
below that concerns an L4 is an ESTIMATE or is unverified.**

## The gap

`_validate_semantics` calls the freshly converted model in fp32, without
autocast, with an input that requires gradients. The two timed blocks call the
trained model under fp16 autocast with an input that does not require
gradients: `_timed_block` in training mode with gradients, `_inference_block`
in evaluation mode under `no_grad`. A submission may route on exactly those
properties (a guarded fast path that falls back to the eager forward when the
input requires gradients, autocast is off or not fp16, or gradients are enabled
in evaluation mode). The fp32 guards then exercise the eager path and the
timings exercise the fast path, which was checked only through the accuracy and
loss of the training script.

## A second gap: one repeated batch

The timed repeats used one batch (the first validation photographs) for all
three warmups and eleven timed repeats, and the model state does not change
between repeats. A submission that recognised the repeated values could return
results it had computed correctly once and do less work in the timed calls,
and it would match any numerical check. This is closed by giving every repeat
new values (decided by the owner after the first version of this change).

## What is validated now

For each workload `_run_workload` builds an **oracle**: a copy of the trained
reference model (reference code) loaded with the candidate model's
`state_dict()` (`_oracle_model`). Candidate and reference were trained
separately and have different weights, so the reference model itself cannot be
the comparison target.

Both paired protocols run through `_paired_protocol`:

- **New values in every repeat.** `_repeat_batches` draws, before any timing,
  one batch for each of the 14 repeats (3 warmups, 11 timed) of each protocol.
  A batch has the workload's batch size and the same shape, dtype, device and
  contiguity as before. Its values come from the workload's ten validation
  photographs: a random selection, one random circular shift, a random
  horizontal flip and added Gaussian noise (σ = 0.05), so the inputs keep the
  statistics of normalised photographs and no value repeats; targets are
  random class indices. The generator is seeded from `os.urandom` for every
  workload.
- **Same values for both sides of a pair.** Within a repeat the candidate and
  the reference each receive their own clone of that repeat's batch, in the
  unchanged alternating order (reference first in even repeats). No later call
  of any model sees those values again, in either protocol.
- **Every candidate call is a validated call.** `_timed_block` and
  `_inference_block` return what the call produced inside the timing window:
  the logits, the loss the harness computed from them and, for the training
  block, the gradient of every parameter. After both windows of the repeat have
  closed the oracle runs the same block function on that repeat's batch and the
  candidate's results are compared with it. The validated outputs are the
  outputs of the timed calls themselves.
- **No separate validation calls.** The first version interleaved untimed
  fresh-input rounds because the timed repeats could not expose a stale
  replay. With new values in every repeat they add nothing and are removed:
  there is only one kind of candidate call, so nothing distinguishes a
  validated call from a timed one.

The conditions are, for the training block: `train()` mode, gradients enabled,
`torch.autocast("cuda", float16)`, a contiguous fp32 CUDA input that does not
require gradients, `model(images)`, cross-entropy and `backward()`; for the
inference block: `eval()` mode, `no_grad`, the same autocast and input. The
fp32 guards of `_validate_semantics` are unchanged.

Unchanged: workload count, gate, score, batch shape, the numbers of warmups and
timed repeats, the pairing and its order, policy and limits. Changed in the
measurement: the timed input values are no longer the first validation
photographs with their labels but a new photograph-derived batch with random
targets in every repeat; candidate and reference are still timed on identical
values within a repeat. A mismatch is a candidate rejection at stage
`model_workload` (scored zero, `beats_reference` 0), like the other guards.
The per-workload private row gains `training_path_validation` and
`inference_path_validation` (largest tolerance-normalised deviations of
outputs, of the joint gradient and of any gradient group, the number of
validated blocks, the number of training repeats in which the gradient check
had detection power, and `replay_output_separation`); nothing solver-visible
changes.

## What it costs and what remains

- Inside the timing window the blocks keep two Python references (logits and
  loss) per batch; no GPU work is added there. Between repeats the oracle runs
  its blocks (three for a training repeat, one for an inference repeat) and
  the comparison launches a few small GPU operations per parameter tensor
  (about 177 tensors for the training block, one for the inference block). The first block of the next repeat follows that work, and
  it is the candidate's in odd repeats and the reference's in even ones (five
  and six of the eleven timed repeats). The effect on the medians is expected
  to be nil but is unmeasured.
- **What can still be reused across repeats.** Inputs differ and the
  parameters do not, because the timed blocks take no optimizer step. Anything
  that depends on the parameters alone can be prepared once and reused:
  compiled kernels and captured graphs, buffers, casts of the weights, folded
  or fused weights. That is ordinary optimisation and is not blocked; in real
  training the folded quantities would have to be refreshed after each step,
  so the training latency measured here is that of a forward-backward pass at
  fixed parameters, as it always was. Anything that depends on the input has
  to be computed for each call: the noise makes every batch distinct in every
  element, and a result carried over from another call fails the comparison.
- Recognising a batch by its content now gains nothing, since no content
  recurs. A cache keyed on the call count is a stale replay and is rejected.
- If a trained model's outputs barely depend on its input, a replayed output
  is numerically close to the correct one and no output comparison can separate
  them; it is then also within tolerance of correct. `replay_output_separation`
  records the smallest number of tolerances by which the oracle's outputs
  differed between consecutive repeats (above 1: replaying the previous
  repeat's outputs would have failed on outputs alone). The training block also
  compares gradients, which moved much more in the measurements below.

## Outputs and loss

`||got - want||_2 <= 1e-2 * ||want||_2 + 1e-4 * sqrt(numel)` for each logits
and loss tensor against the oracle's block under the same autocast. Positions
where the oracle is nonfinite must be of the same kind in the candidate (NaN
only matches NaN, infinity must have the same sign) and are left out of the
norm. On the L4 (six trained workloads, 14 repeats each) the reference matched
itself exactly and an independently written all-fp16 formulation stayed within
0.18 of this tolerance in training and 0.06 in inference. With new values in
every repeat this comparison is the primary detector of replayed or skipped
work; it is unchanged.

## Gradients

### What went wrong with the first bounds

The first version compared the candidate's gradients with the oracle's own
fp16 gradients: 3e-2 relative for all parameters jointly, 7.5e-1 per tensor.
They were calibrated on a CPU with untrained ViT-Tiny/Small and noise inputs.
On the L4 the operator's all-fp16 formulation (every operand and operation of
the mapping in half precision, result cast back), which is a numerically
honest way to run the timed block, exceeded the joint bound on three of six
workloads, by up to 15.5 times (about 46% relative L2 on `final_b`), and the
per-tensor bound once (1.56 times), while the reference matched itself with
deviation 0 everywhere.

### Mechanism

Established on CPU with states trained by the task's own recipe (authentic
`main.py`, reference patch, the six workloads' shards, seeds and batch sizes;
fp32 on CPU because AMP needs the GPU) and real fp16 autocast:

- **The trained models are saturated.** After two epochs at lr 0.004 the
  activations have grown (site inputs up to about 570) while `alpha` stays
  near 0.5 (0.48 – 0.52). 69 – 95% of the replacement-site inputs have
  `|alpha x| > 4.2` and 49 – 90% have `|alpha x| > 9`. For `|alpha x| > 4.2`
  the output rounds to exactly ±1 in half precision and the derivative
  `1 - y²` computed in fp16 is exactly zero, where fp32 still resolves it
  (down to about 1e-7).
- **The backward signal is at half-precision resolution.** The timed block
  calls `loss.backward()` without a gradient scaler. 70 – 76% of the fp32
  gradient entries are below the smallest normal fp16 number (6.1e-5), where
  fp16 has an absolute step of 6e-8 instead of a relative one; 8 – 19% of the
  entries are exactly zero in fp32 and 17 – 27% in fp16.
- **The reference keeps its sites in fp32 even under autocast**: the residual
  stream is fp32 (the position embedding and the residual additions promote
  it), so `alpha * x` and the tanh derivative are fp32 and only the linear and
  attention layers run in fp16. An all-fp16 site loses the small derivatives
  and sums its scalar `alpha` gradient over hundreds of thousands of fp16
  terms with heavy cancellation.
- Sizes measured on CPU (ViT-B/32 and ViT-B/16 states): the two fp16
  arrangements differ from each other by 0.07 – 0.5% jointly, 0.3 – 0.9% on
  the non-site trunk parameters, 0.6 – 3.8% on the `alpha` group; single
  `alpha` scalars with gradients near 1e-7 by up to 69 times their own size.
  Against fp32 the reference's own fp16 gradient is off by 0.1 – 0.7% jointly,
  up to 2.5% on the trunk and up to 30% on the `alpha` group. The joint norm is
  dominated by the classifier head (7.5 – 19.5 of 7.7 – 19.7).

**Not reproduced on CPU:** the size seen on the L4. The CPU states show a
joint disagreement of at most 0.5% where the L4 showed 0.6 – 0.85% on the
three quiet workloads and 6%, 13% and 46% on the others (worst repeat of 14).
The CPU states were trained in fp32 rather than under AMP and CPU half
arithmetic is not the L4's; which parameters carry the L4 disagreement (a
trunk gradient that outweighs the head on some repeats, or one large tensor)
is therefore not known. The calibration script prints exactly that per
workload and repeat (per group: fp32 norms, both fp16 errors against fp32,
their mutual distance, cosines, norm ratios, zero and sub-normal fractions,
nonfinite counts, the saturation of the sites, and the eight parameters that
carry most of the disagreement).

The conclusion does not depend on the missing detail: two honest fp16
backward passes are not near-equal on these states, so near-equality to one
of them is the wrong test.

### The check

For every training repeat the oracle (reference code, candidate state)
supplies four trusted gradients for the same inputs (`_oracle_block`):

- `r`: the reference under the timed conditions (fp16 autocast, fp32 sites);
- `h`: the same with every replacement site evaluated entirely in half
  precision, as separate operations (`_half_precision_sites`);
- `f`: the same in half precision as one fused multiply-add
  (`_fused_half_site_forward`); added after the L4 calibration, see below;
- `t`: an fp32 gradient without autocast (`_truth_gradients`).

For all parameters jointly and for four groups (the sites' `alpha`, `weight`
and `bias` gradients, and all other parameters), with every distance divided
by `||t||` of the group (`_gradient_errors`):

    distance = min over trusted x in {r, h, f, t} of ||c - x||
    spread   = max over trusted pairs of ||x - x'||
    accept if distance <= floor + TIMED_GRADIENT_FACTOR * spread

- The candidate has to be near *one* trusted gradient, as near as the trusted
  gradients are to each other on that very state and input. Where fp16
  resolves the gradient the bound is tight; where it does not, the bound
  widens by itself instead of by a constant chosen in advance.
- **An all-fp16 site arrangement is accepted by construction** (it is `h` or
  `f`: distance 0), as is the reference (`r`) and anything more accurate than
  these (near `t`). Other honest arrangements (fused or
  reordered fp16 or fp32 forms, hand-written backward passes) land between
  them; the factor 3 and the floors are their margin.
- Missing or misshapen gradients are rejected outright. Entries that are
  nonfinite in a trusted gradient are left out, except that where `r`, `h` and
  `f` are all nonfinite of the same kind the candidate must be of that kind; a
  nonfinite candidate entry where all trusted gradients are finite is
  rejected.
- A repeat in which the joint bound reaches 0.5 has no power against a halved
  gradient (and none against a zeroed one at 1). The number of repeats below
  0.5 is recorded per workload as `gradient_power_blocks`. Losing power on a
  repeat is the price of not rejecting honest work there; outputs and loss are
  still compared on every repeat.

Constants: `TIMED_GRADIENT_FACTOR = 3`, `TIMED_GRADIENT_FLOORS = {joint 0.02,
other 0.02, alpha 0.10, scale 0.05, shift 0.05}`. These are the values chosen
before any L4 data existed; the L4 calibration did not ask for other floors
and the factor was deliberately not refitted (decision record below).

### Separation measured on CPU

`calibrate_timed_path.py --device cpu` on the CPU-trained `dev_b` and
`final_b` states (ViT-B/32, batch 5, three training repeats each, real fp16
autocast). Values are tolerance-normalised: above 1 is rejected.

| Honest variant | outputs | gradients (worst group) |
| --- | --- | --- |
| reference | 0 | 0 |
| all-fp16 sites (independent implementation) | 0.02 – 0.03 | 0.000 |
| fp16 sites, fused `addcmul` | 0.02 – 0.03 | 0.02 – 0.06 |
| fp16 sites, hand-written backward `(1-y)(1+y)`, fp32 reductions | 0.02 – 0.03 | 0.01 – 0.03 |
| fp16 forward, fp32 analytic backward | 0.02 – 0.03 | 0.02 – 0.19 |
| fp32 sites, fused `addcmul` | 0.00 – 0.02 | 0.01 – 0.04 |

| Attack (honest forward unless noted) | outputs | gradients (worst group) | deciding group |
| --- | --- | --- | --- |
| all gradients zeroed | 0 | 39 – 44 | other |
| non-site gradients zeroed | 0 | 39 – 44 | other |
| site gradients zeroed | 0 | 16 – 18 | scale, shift |
| `alpha` gradients zeroed | 0 | 4.0 – 7.5 | alpha |
| halved | 0 | 20 – 22 | other |
| sign flipped | 0 | 79 – 88 | other |
| gradients of neighbouring sites swapped | 0 | 23 – 26 | scale, shift |
| gradients of the previous repeat's batch | 0 | 45 – 63 | other |
| everything replayed from the previous repeat | 75 – 88 | 45 – 63 | outputs |

Honest maximum 0.19, attack minimum 4.0: a factor of 21 on CPU. The joint
spread was 0.09 – 0.18% in these runs, so all repeats had power. On the L4 the
spread is larger (up to about 46% on the worst repeat), the bound follows it,
and what remains of the separation there is what the calibration has to show.

Synthetic cases in the tests cover the L4 regime directly: with a trusted
all-fp16 gradient 45% away from the others, that gradient and an independent
one at the same error level are accepted and the repeat reports no power; with
a resolved signal every attack above is rejected.

### L4 calibration, 2026-10-12

One L4, the verifier GPU image, evaluator of revision `98e40da9` (trusted set
`r`, `h`, `t`; factor 3; the floors above); six workloads trained by the
evaluator's reference training, 14 training repeats each, five honest
variants and nine attacks: 504 honest and 744 attack measurements. Raw data:
operator's `dynamic389-l4-calibration/calibration.json`. The operator's own
all-fp16 formulation matched the script's bit for bit on `dev_a` over all 14
repeats. Script verdict: **UNSOUND**, on one honest measurement.

**The fp16 noise is small on most repeats and heavy-tailed.**

| Workload | joint spread, median | joint spread, max | `alpha` spread, median | `alpha` spread, max | repeats with power |
| --- | --- | --- | --- | --- | --- |
| dev_a | 0.34% | 3.4% | 17.6% | 122% | 14/14 |
| dev_b | 0.12% | 13.3% | 0.95% | 39% | 14/14 |
| dev_c | 1.09% | 2.1% | 4.2% | 12% | 14/14 |
| final_a | 0.47% | 15.4% | 9.0% | 170% | 14/14 |
| final_b | 0.14% | 0.26% | 2.1% | 12% | 14/14 |
| final_c | 0.80% | 2.3% | 4.6% | 9% | 14/14 |

- Three of 84 repeats have a joint spread of 10 – 15% (`dev_b` 10, `final_a`
  1 and 10); the rest are below 3.5%. In the noisiest repeat of `dev_a`,
  `dev_b`, `final_a` and `final_c`, 51 – 72% of the squared disagreement
  between the two fp16 arrangements sits in `blocks.0.attn.qkv.weight` and
  12 – 40% in `patch_embed.proj.weight`: the
  gradient that has travelled back through all twelve blocks. Either
  arrangement can be the one that is off: on `dev_b` repeat 10 the
  *reference's* fp16 gradient is 13.3% from fp32 and the all-fp16 one 2.0%; on
  `final_a` repeat 10 it is 1.5% and 8.5%. This is amplification of rounding
  through depth, not a property of one formulation, and it answers what the
  CPU could not: the L4 disagreement is carried by the earliest layers.
- It moves between runs. The operator's earlier diagnostic saw 46% on
  `final_b`; in this run `final_b` is the quietest workload (at most 0.12%
  between the fp16 arrangements) and `final_a` and `dev_b` carry the large
  repeats. Run-to-run variation has not been measured beyond these two looks.
- Saturation 52 – 95% of site inputs, largest site input 568, no nonfinite
  gradient entry anywhere.
- The `alpha` group is poorly resolved on `dev_a` and `final_a`: the
  reference's own fp16 `alpha` gradient is typically 18% and 9% from fp32.
  With the trusted set of that revision every repeat still had joint power
  (largest joint bound 0.48).

**Honest variants** (tolerance-normalised, worst over 14 repeats; gradients):

| Variant | dev_a | dev_b | dev_c | final_a | final_b | final_c |
| --- | --- | --- | --- | --- | --- | --- |
| reference, all-fp16, fp32 fused | 0 | 0 | 0 | 0 | 0 | 0 |
| fp16 hand-written backward | 0.015 | 0.029 | 0.019 | 0.024 | 0.034 | 0.022 |
| fp16 forward, fp32 backward | 0.057 | 0.058 | 0.044 | 0.069 | 0.040 | 0.107 |
| fp16 fused `addcmul` | 0.043 | 0.115 | 0.026 | **0.696** | 0.041 | 0.151 |

Outputs stayed within 0.061 everywhere. Of the 504 honest measurements one is
above 0.2. It is `final_a` repeat 1, the repeat with the largest spread of the
whole run (joint 15.4%): the fused form's gradient is 2.11 spreads from the
nearest trusted gradient in all five groups alike (joint distance 32%). So it
is not a tight repeat where a fused form lands badly; it is a noisy repeat in
which a third fp16 arrangement drew a noise twice as large as the distance
between the two trusted ones. On the other two noisy repeats the same variant
was 0.07 spreads away.

**Attacks** (repeats rejected with deviation at least 1.5, of 14, or of 13 for
the two that need a previous repeat):

| Attack | dev_a | dev_b | dev_c | final_a | final_b | final_c |
| --- | --- | --- | --- | --- | --- | --- |
| all zeroed, non-site zeroed, sign flipped, sites swapped | 14 | 14 | 14 | 14 | 14 | 14 |
| site gradients zeroed | 14 | 14 | 14 | 13 | 14 | 14 |
| halved | 14 | 13 | 14 | 14 | 14 | 14 |
| `alpha` gradients zeroed | **7** | 13 | 14 | 12 | 14 | 14 |
| previous repeat's gradients | 13 | 13 | 13 | 13 | 13 | 13 |
| everything replayed | 13 | 13 | 13 | 13 | 13 | 13 |

The replay is caught by outputs on every repeat (deviation at least 35). The
weakest case is zeroed `alpha` gradients on `dev_a`, which only the poorly
resolved `alpha` group can see.

**Reading.** No honest measurement failed at run time (maximum 0.70 of the
bound) and no attack came near passing a workload, so the honest and attack
ranges do not overlap. The rule failed on margin: 0.696 against a required
0.67. That is a real finding, because the cause is structural. The spread of
two fp16 gradients is a two-sample estimate of a heavy-tailed noise, and a
third honest arrangement fell 2.1 of those estimates away.

**Alternatives computed offline from that run:**

| Change | honest maximum | zeroed `alpha` on dev_a | halved on final_a | fewest repeats with power |
| --- | --- | --- | --- | --- |
| none (factor 3) | 0.696 | 7/14 | 14/14 | 14 |
| factor 3.5 | 0.597 | 6/14 | 12/14 | 13 |
| factor 4 | 0.523 | 6/14 | 12/14 | 13 |
| factor 4.25 | 0.492 | 5/14 | 12/14 | 13 |
| factor 4.5 | 0.465 | 4/14 | 12/14 | 13 |
| factor 5 | 0.419 | 4/14 | 12/14 | 12 |
| factor 6 | 0.349 | 4/14 | 12/14 | 12 |
| fused fp16 form trusted, factor 3 | 0 for it; at most 0.107 for the others | at least 6/14 | at least 12/14 | at least 13 |

The last row is a bound, not a measurement: the stored data give each
variant's distance to the old trusted set, so the new spread was taken at its
largest possible value (old spread plus the fused form's distance) for the
attacks and for power, and the other honest variants' distances can only
shrink. Zeroed `alpha` gradients on `final_a` fall from 12 to 10, 9, 8 and 7
of 14 at factors 4, 4.5, 5 and 6 and stay at 12 with the fused form trusted;
every other attack stays at 12 or more of 14 in every row.

**Decision: the trusted set gains the fused all-fp16 form; factor and floors
stay at their pre-calibration values; the rule's honest limit becomes 0.5.**

- A larger factor is the wrong repair. Reaching a twofold honest margin needs
  4.25 or more, a number that exists only because of one measurement, and it
  spends the margin of the weakest attack: zeroed `alpha` gradients on `dev_a`
  would sit at 5 or 4 of 14, at the rule's minimum of 4 with nothing to spare.
  It also leaves the cause in place, so the next outlier could be 3 spreads.
- A third fp16 arrangement makes the spread a three-sample estimate of the
  fp16 noise, at one more fp16 oracle block per training repeat. In the repeat
  that failed, the bound becomes at least 0.02 + 3 × 0.32 ≈ 0.98, which is what
  that repeat's noise warrants; in quiet repeats nothing changes. The form
  chosen is the simplest other way to write the same half-precision mapping.
  It keeps more detection power than any factor that reaches the same honest
  margin.
- The remaining honest variants become a genuine held-out set: none of them is
  in the trusted set, and their worst value in this run was 0.107 (4.7 times
  inside the 0.5 limit). The calibration script gains two more held-out
  variants (half-precision tanh with fp32 scale and shift; fp32 product with
  half-precision tanh).

**What this decision is and is not.** It was made after seeing one calibration
run, and the form added is the one that failed in it. Three things limit the
fitting: no constant was refitted (factor and floors are the values written
before any L4 data); the change is justified by the noise structure found
(heavy tail, either arrangement can be the outlier), which would have called
for it whichever variant had exposed it; and the held-out variants, not the
added one, carry the acceptance test. What it cannot show is how it behaves on
another run, where the noisy repeats fall elsewhere.

**Confirmation required before publication.** Two further runs of the
calibration script on the L4 with this revision's judge (each run draws new
inputs; the command is below). Both must print `VERDICT: SOUND` under the rule
below. If either does not, the constants are not to be tuned: report the
cases.

    python3 task_tests/dynamic_tanh/calibrate_timed_path.py --judge /opt/frontier/task --out calibration.json

Rule, as the script applies it:

1. The factor stays 3. For each group, `needed = max over honest variants,
   workloads and repeats of (distance - 3 * spread)`; the floor becomes
   `max(current floor, 2 * needed)`. (In the 2026-10-12 run nothing was
   needed.)
2. Every honest variant must stay at or below 0.5 on every repeat of every
   workload: the trusted forms read 0, the held-out ones carry the test.
3. Every attack must be rejected with a deviation of at least 1.5 in at least
   4 of the 14 repeats of **every** workload.
4. If 2 or 3 fails the script prints `VERDICT: UNSOUND` with the cases. Then
   the constants must not be tuned to make it pass: the check has to change,
   or the owner accepts a stated loss of power for the named attack.

## What a rejection says

**What the solver sees.** The Agent-facing train result is assembled from the
verification's `state`, `metrics`, `raw_score` and, for a failed verification
only, a `failure` object. A candidate rejection is a *passed* verification
with score 0, and the evaluator's message (the result `summary`) is not part
of the train result. Until this change every rejected submission therefore
looked the same to the solver: score 0 and no metrics, whether the patch was
out of policy, the fp32 guards failed, training crashed or a timed block
disagreed.

**The channel.** The verifier already supports one declared, validated
solver-visible field for a rejected submission, used by another published
task: `metrics.candidate_diagnostic`. It is admitted only when the task
bundle declares a validator (`judge/candidate-diagnostic.json`, which pins
`judge/candidate_failure.py` by SHA-256), only as the sole metric of a scored
zero, only after `public_diagnostic` accepted it, and only in train mode. No
verifier code changes. The value is an object of strings, the form the
existing validator and that task already use:

    {"schema": "dyt.candidate-diagnostic/v1", "stage": ..., "block": ...,
     "quantity": ..., "message": ...}

- `stage`: `patch_policy`, `semantic_validation` (conversion checks and fp32
  guards), `model_workload` (training, checkpoint, a crash in a timed block)
  or `timed_path`;
- for `timed_path` only: `block` (`training` or `inference`) and `quantity`,
  the first failing comparison in a fixed order (model output, loss, gradient
  of all parameters jointly, of the parameters outside the sites, of the
  sites' `alpha`, `weight`, `bias`; a missing or wrong-shape result;
  nonfinite values that do not match); otherwise both are null;
- `message`: the fixed sentence for that category.

Everything is built from the tables in `candidate_failure.py`
(`_candidate_diagnostic`), and the validator rebuilds the object from the same
tables and compares, so no number, threshold, reference quantity, workload
identity, repeat position or text from the submission or from an exception
can enter. A rejection that is not exactly a tagged timed-path rejection gets
its stage only. The field is declared in `beat_spec.json` `solver_feedback`,
in the policy's and `config.yaml`'s `solver_visible_metrics`, in the statement
and in the task guide. It is absent from accepted submissions and from every
final response; final rejections keep the private diagnostic and the
`beats_reference: 0` verdict exactly as before.

The evaluator's message keeps the category text added earlier (`...:
timed-path validation, training block, gradient of the replacement sites'
alpha`); it stays in the result `summary` and in evidence for operators.

**Cost.** The verifier accepts `candidate_diagnostic` only as the sole metric,
so a rejected *train* submission no longer carries the private
`all_final_metrics.candidate_failure` (stage, exception kind) and runtime
receipt in its evidence; the stage is now public, the exception kind and the
GPU receipt of a rejected train call are lost. Final rejections keep them.
Keeping both would need a change to the verifier's admission rule.

**What a solver could learn.** One of 15 reachable values per rejected train
call (three stages, nine timed-path categories for the training block, three
for the inference block): at most 3.9 bits, in place of that call's
measurements. That a call was rejected was already visible as a zero score.
The run has three train calls; with a hypothetical hundred, a solver that
deliberately perturbs one quantity could bisect how large a perturbation of
*its own* result is still accepted, which the zero score alone already
allowed. It learns no reference output or gradient (the category does not
depend on their values), not which workload or repeat failed (the first
failing repeat is reported without its position, and the bound that binds is
the smallest over 42 repeats with inputs drawn anew in every verification),
and nothing that helps a wrong result pass: the rule is public in the
statement and the only way inside the bound is a gradient as close to the
trusted ones as they are to each other. The gain for honest solvers is
knowing which part of a submission to fix with the three calls they have.

## Added verifier time (ESTIMATE)

Recorded on an L4: reference training block 27 – 33 ms, inference block
9 – 12 ms; complete verifications 140 – 164 s with the previous revision of
this validation. Per training repeat the oracle now runs three blocks instead
of one (fp16, fp16 with half sites in two forms, fp32; the fp32 block is
estimated at two to three times an fp16 block) and one comparison over all
parameters (eleven sums per tensor): about 0.25 – 0.3 s, against about 0.07 s
in the first revision. With 14 training repeats and three workloads that is
**about 11 – 14 s more than no timed-path validation**, of which about 1.5 s
come from the fourth trusted gradient. The
slowest recorded train verification took 678 s from submission to feedback;
with this added it stays under 700 s against the 900 s limit, which is not
changed. GPU memory: the oracle's four gradient sets are held at once (about
1.4 GB) on a 24 GB card.

## Tests

`./python task_tests/dynamic_tanh/run.py -q`: 69 passed, 37 skipped (torch is
not in the repository environment). With torch 2.5.1+cpu and timm 1.0.15
installed: 106 passed.

Torch-free (always run): the paired schedule and its order; new values for
every repeat and every model; every candidate call validated; stale and
call-count replays rejected; a content-keyed cache never hits; the evidence
stays private; reference and candidate timing failures keep their distinct
outcomes.

Real tensors on CPU: the earlier cases (reference, guarded fast path that
recomputes, stale replay, content and call-count caches, error confined to
fp16 inference, halved or missing gradients, drawn batches, the oracle, the
wiring), and for the gradient redesign:

- grouping of parameters by replacement site;
- the bound follows the trusted spread: resolved signal (tight) and a
  trained-like one with the all-fp16 gradient 45% away (accepted, no power);
- each attack of the table on synthetic gradients, with the group that
  exposes it; a near-zero scalar carrying an fp16-sum error is accepted;
- nonfinite entries: rejected where the trusted gradients are finite, kind
  must match where the trusted fp16 gradients agree, ignored where they
  disagree;
- a saturated tiny model (most site inputs beyond `|alpha x| = 4.2`): the
  oracle supplies the four gradients and restores itself; the reference, an
  independently written fused all-fp16 formulation (distance 0) and a held-out
  arrangement outside the trusted set pass both protocols; zeroed,
  halved, sign-flipped, site-only-zeroed, `alpha`-only-zeroed and
  previous-batch gradients behind an honest forward are rejected;
- the calibration script's measurement and rule on tiny models, including an
  overlap being reported rather than hidden.
- rejection categories: the message table, each quantity and the fixed order,
  the block added by the paired schedule; through the real `verify()`
  (torch-free): `metrics.candidate_diagnostic` in a train response for
  timed-path, stage-only and patch-policy rejections, its absence from
  accepted submissions and from every final response, the unchanged final
  verdict and private diagnostic; the validator admits exactly the table.

Not exercised without a GPU: real `torch.autocast("cuda", float16)`, CUDA
events and graphs, ViT-Base on an L4, the reference patch through the whole
evaluator, the size of the trusted spread on the L4's trained states and
therefore the detection power there, the floors, the added seconds, the
timing neutrality of the oracle blocks between repeats, and the motivating
submissions.
