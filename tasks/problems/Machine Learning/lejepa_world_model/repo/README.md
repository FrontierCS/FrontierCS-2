# Latent Objective Recovery

Implement the training objective of a LeJEPA-style self-supervised encoder so that the encoder
recovers the hidden two-dimensional latent state of a synthetic world. Maximize the 0–100 score:
bidirectional affine recovery and orthogonal-map quality, plus a bonus for lower total wall time
than the withheld comparator.

## Task

The code is the LeJEPA Identifiability codebase's Gaussian two-dimensional spiral experiment. A
latent state is transformed nonlinearly into an observation, and two independently perturbed
states derived from the same latent row form a positive pair. The fixed engine trains an encoder
with your losses, evaluates it against a fixed 10,000-row latent sample, and writes a JSON and a
PyTorch result file. The bodies of the four loss interfaces in `experiments/lejepa_id/losses.py`
have been removed; the scored path uses `SIGReg`, `whitening_loss` and `alignment_loss`
(`infonce_loss` is retained but not called).

## What you may change

- `experiments/lejepa_id/losses.py` and Python helpers under
  `experiments/lejepa_id/frontier_solution/` (Python source only); at most 20 changed
  files; no file may be deleted. Everything else is fixed. Losses receive only representation
  tensors.
- Every inspected file must parse. Imports are limited to `import torch`, `import torch.nn as nn`,
  `import torch.nn.functional as F`, `import math`, and symbol imports from modules inside
  `lejepa_id.frontier_solution` (relative imports must stay inside it). The external modules
  expose exactly this API:

```text
torch:
  Tensor Size device dtype finfo iinfo pi
  float16 bfloat16 float32 float64 int32 int64 bool
  tensor as_tensor arange linspace logspace
  zeros ones empty full zeros_like ones_like empty_like full_like
  rand randn randint randperm rand_like randn_like eye
  cat concat stack vstack hstack split chunk unbind
  abs square pow sqrt rsqrt exp expm1 log log1p sin cos tan tanh
  sigmoid erf erfc softmax logsumexp sum mean var std cov amin amax
  min max minimum maximum clamp where isfinite isnan isinf all any
  sort argsort topk unique matmul mm bmm mv dot inner outer einsum
  triu tril diag diagonal trace kron cdist pdist
  no_grad enable_grad is_grad_enabled manual_seed
torch.nn or nn: Module Parameter ParameterList ModuleList Identity
torch.nn.functional or F:
  normalize mse_loss l1_loss cosine_similarity softmax log_softmax
  relu gelu silu tanh sigmoid pad
torch.linalg:
  norm vector_norm matrix_norm qr svd svdvals solve lstsq inv pinv
  cholesky eig eigh eigvals eigvalsh det slogdet matrix_rank matrix_exp
  cond cross householder_product
torch.special:
  erf erfc erfcx expit gammaln log_ndtr ndtr entr xlog1py xlogy sinc
  softmax log_softmax
torch.autograd: Function grad
math:
  pi e tau inf nan sqrt exp expm1 log log1p sin cos tan tanh erf erfc
  gamma lgamma floor ceil fabs isfinite isnan isinf prod
```

- Calls by bare name are limited to solution-defined functions and classes, imported helper symbols, and these built-ins: `abs`, `all`, `any`, `bool`, `dict`, `enumerate`, `filter`, `float`, `int`, `isinstance`, `len`, `list`, `map`, `max`, `min`, `next`, `pow`, `range`, `reversed`, `round`, `set`, `sorted`, `sum`, `super`, `tuple`, `zip`, `RuntimeError`, `ValueError`, `TypeError`, and `NotImplementedError`. Tensor and solution-object methods are permitted except that attributes beginning with `__` other than `__init__` and these capability attributes are rejected:

```text
os sys subprocess builtins importlib serialization modules module
system popen fork forkpty vfork spawn _exit
execv execve execl execlp execvp
open fdopen listdir scandir walk chdir getcwd
environ getenv putenv
remove unlink rename replace mkdir makedirs rmdir removedirs
read_text write_text read_bytes write_bytes tofile dump load save
gi_frame gi_code gi_yieldfrom cr_frame cr_code cr_await
ag_frame ag_code ag_await f_globals f_locals f_builtins
f_back f_code f_trace tb_frame tb_next mro
```

- The names `open`, `eval`, `exec`, `compile`, `__import__`, `breakpoint`, `getattr`, `setattr`, `delattr`, `vars`, `globals`, `locals`, `input`, `exit`, `quit`, `help`, `license`, `credits`, `__builtins__`, `os`, `sys`, `subprocess`, `importlib`, `builtins`, `SystemExit`, `KeyboardInterrupt`, `type`, and `object`, and any name beginning with `__` (such as `__name__`), are also rejected. No `global` or `nonlocal`. No rebinding or shadowing of imported names; no mutation of imported objects (assignment, unpacking, loop, comprehension, `with` target, `del`). Framework modules may not be aliased, passed as values or relayed through helpers; framework types only as constructors, bases or annotations. No custom metaclasses, class keywords, base-construction hooks, or binding of interpreter-managed namespace entries. Local helper symbols must be imported directly, not accessed through an imported helper module; helpers are checked under the same policy.

## Interface

The judge runs, from `/workspace/repo/experiments`:

```bash
python3 run.py --config <generated-workload.yaml> --run spiral_lejepa --seed <suite-seed>
```

with no other flags. The generated YAML is `experiments/configs/2d.yaml` with `out` set to a temporary
directory, `steps: 1000` and `log_every: 250`; its scored part:

`N: 2`, `source_dist: gaussian`, `num_eval: 10000`, `lr: 0.003`, `batch_size: 256`, `rho: 0.95`, and
run `spiral_lejepa` with `mixing: spiral`, `encoder: mlp`, `hidden: 256`, `mode: lejepa`,
`lamb: 0.001`.

- `SIGReg(knots=17, n_slices=256, t_max=3.0)` is a callable PyTorch module (training uses the
  defaults; preflight also builds it with `knots=9, n_slices=13, t_max=2.0`); `SIGReg.forward(h)`,
  `whitening_loss(h)` and `alignment_loss(h)` map a floating `(views, batch, features)` tensor to
  one scalar tensor. `infonce_loss(h, sigma)` is not called.

Loss input `h` has shape `(views=2, batch=256, features=2)`. Every checked callable must satisfy:

```python
>>> h = torch.tensor([[[0.0, 1.0], [1.0, 0.0]],
...                   [[0.2, 0.8], [0.9, 0.1]]], requires_grad=True)
>>> value = SIGReg(knots=9, n_slices=13, t_max=2.0)(h)
>>> value.shape, value.dtype, value.device.type
(torch.Size([]), torch.float32, 'cpu')
>>> value.backward(); h.grad.shape
torch.Size([2, 2, 2])
```

For seed `s` the runner writes `<out>/spiral_lejepa_seed=s.json` and a same-stem `.pt` file. The
judge reads only `r2_zh`, `r2_hz`, `orth_err_normalized`, `epsilon`, `delta` and `procrustes_mse`
from the JSON and measures wall time around the whole command.

## Correctness

Your code runs unprivileged: no process creation or execution, network sockets, tracing or
cross-process memory access; only its scratch and output directories are writable.

Preflight (30 s): each of `SIGReg(...)(h)`, `whitening_loss(h)`, `alignment_loss(h)` on fresh
`(2, 32, 2)` `float32` inputs that require grad must return a finite rank-zero tensor on the input
dtype and device whose backward gives a finite, nonzero input gradient.

Purity (150 s): apart from random numbers drawn from Torch's global generator during a call, no
state may carry between loss calls (values fixed at import or in `SIGReg.__init__` are fine). On
the same `(2, 256, 2)` input and generator seed, a freshly imported loss and one already called for
up to 1,001 updates must return the same value and input gradient to a relative `1e-4`.

The evaluation timeout in `/frontier/task/README.md` covers preflight, purity, the comparator's
three trainings and yours, so your three runs have much less than 3 × 150 s.

Each of these scores 0: a broken rule above (rejected; a stateful loss is rejected as
`content_policy`, as is a syntax error), a failed preflight, a crash or early exit, a missing or
malformed JSON or `.pt` result (the `.pt` must be a PyTorch zip archive of at least 1 KiB
containing `data.pkl`), a missing or non-finite scored metric, a negative `epsilon`, `delta` or
`procrustes_mse`, and a training subprocess over 150 seconds.

## Workload

One workload is one fixed-seed run: every optimizer update draws a fresh batch of 256 rows
`z ~ N(0, I_2)`, and each row gives two views

```text
z_view[v] = 0.95 z + sqrt(1 - 0.95^2) eta[v]
eta[v] ~ N(0, I_2), independently for v in {0, 1}.
```

so each view has correlation 0.95 with `z`. The observation is

```text
x = g(z_view) = R(pi ||z_view||_2) z_view,
```

with `R(theta)` the 2-D rotation. The MLP has width 256, GELU and five `Linear` layers; AdamW
(default weight decay) at learning rate 0.003, constant for the first half, cosine decay in the
second. The loop runs steps 0–1000 inclusive (1,001 updates). Metrics are evaluated 13 times during
training and once after it on a fixed 10,000-row Gaussian sample; the run then writes 100,000
encoded rows to the `.pt` file. All of this is inside the timed command.

| suite | workloads |
| --- | --- |
| development (train) | 3 fixed seeds |
| final | 3 other fixed seeds, same process, disjoint |

Seeds are not disclosed.

## Scoring

With `y_hat` the least-squares affine fit of `y` from `x`:

```text
R^2(x -> y) = 1 - sum((y - y_hat)^2) / sum((y - mean(y, axis=0))^2).
```

Per workload `w`: `a_w = clip(r2_zh_w, 0, 1)` (fit of `h` from `z`, `h_hat = A z + b`),
`b_w = clip(r2_hz_w, 0, 1)` (reverse fit), and

```text
o_w = ||A^T A - I_2||_F / sqrt(2).
```

With `W = 3` workloads and `T_c`, `T_h` the candidate's and the withheld comparator's total command
times:

```text
Q_w        = 35 a_w + 35 b_w + 20 clip(1 - o_w, 0, 1)
efficiency = min(10, 5 T_h / max(T_c, 1e-9))
score      = round(clip((1/W) sum_w Q_w + efficiency, 0, 100), 4)
```

Metrics and the criterion below use raw, unrounded values.

Anchors: matching the comparator's time gives an efficiency of 5, and half its time gives the
maximum 10; perfect recovery (`a = b = 1`, `o = 0`) at maximum efficiency scores 100.

**Beats the reference** (the withheld comparator) when, on every workload, computed by
`experiments/lejepa_id/metrics.py` on the final
evaluation set: `r2_zh` and `r2_hz` strictly higher; `orth_err_normalized`, `epsilon`
(`||Cov(h) - I||_F`) and `procrustes_mse` (error of `h` against the best orthogonal map of `z`)
strictly lower; `delta` (`max(L_h - 2(1-rho) tr(Cov h), 0)`) strictly lower, or both exactly 0. No
other tolerance; no metric offsets another. The criterion is separate from the score.

## Running locally

```sh
cd /workspace/repo
bash frontier-build.sh                       # byte-compiles the runner and the package
bash frontier-bench.sh --config <yaml> --run spiral_lejepa --seed 7
```

Use a copy of `experiments/configs/2d.yaml` with `steps: 1000` and `log_every: 250` to match a scored run.

`frontier-bench.sh` runs `run.py` with one BLAS/OpenMP thread.

## Environment

The judge has 2 vCPUs and 2 GiB of memory; Python 3 with CPU-only PyTorch 2.7.1, NumPy 2.2.6 and
PyYAML 6.0.2; Torch and BLAS use one thread.

## Notes

A naive complete implementation scored `49.7282` in each of three repeated development
evaluations. A repeatable gain of one point or more is a genuine improvement.
