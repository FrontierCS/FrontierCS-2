# Polar-Factor Approximation

Implement the Polar Express method: optimal odd-polynomial iterations that approximate the matrix
polar factor with matrix products only. The score rewards small scalar and matrix errors on six
small workloads; beating the reference needs strictly better errors than its best variant on all
twelve workloads.

## Task

For a real matrix with reduced SVD \(G=U\Sigma V^T\), the polar factor is \(UV^T\), the closest
matrix of the same shape with orthonormal columns (tall) or rows (wide). The code is an extract
of the PolarExpress artifact ("The Polar Express", ICLR 2026): its Python module `polar_express.py` and
the notebook `plotsforpolar.ipynb`. The bodies of `optimal_quintic`, `optimal_composition` and
`PolarExpress` have been removed.

## What you may change

- `polar_express.py` (the three stubs and any other content) and Python helpers under
  `frontier_solution/`; at most 20 changed files; no file may be deleted. The notebook and
  everything else are fixed.
- Every `.py` file there must parse, imported or not. Imports only from `math`, `itertools`,
  `numpy`, `numpy.linalg`, `numpy.polynomial`, `numpy.polynomial.polynomial`, `torch`, and
  `frontier_solution` (and its modules); no relative or wildcard imports, and no imported name
  beginning with `_`. `matmul` may be imported only from `torch` or `frontier_solution`.
- Forbidden bare names: `__builtins__ __import__ __loader__ __spec__ breakpoint compile eval exec
  getattr globals help locals setattr vars`.
- Forbidden attribute names, on any receiver (torch, NumPy, tensors, your own objects) and as
  `from` import names: every name beginning with `_`, and `ag_code ag_frame attrgetter breakpoint
  builtins cholesky cr_code cr_frame ctypes currentframe dot eig eigh eigvals eigvalsh einsum eval
  exec f_back f_builtins f_globals f_locals from_dlpack getattr getattr_static gi_code gi_frame
  globals help importlib inner inspect inv item locals lstsq lu matrix_power methodcaller
  multi_dot numpy os outer pathlib pca_lowrank pinv polar qr setattr storage subprocess svd
  svd_lowrank sys tensordot tolist untyped_storage vars vdot`. `torch.compile` is allowed.
- No attribute assignment or deletion, no class patterns with attribute lookups of these names,
  and names beginning with `_frontier_` are reserved. NumPy's array `@` and `numpy.linalg.solve`
  are allowed for coefficient arithmetic on scalar parameters; scored tensors may not be exported.

## Interface

The judge imports `polar_express` from the repository root and calls the three interfaces below; their return values are the whole output.

`optimal_quintic(l, u)` receives numeric interval endpoints satisfying `0 <= l <= u` in all required calls. It must return exactly three finite real values `(a, b, c)`, in increasing power order, representing

\[
p(x)=a x+b x^3+c x^5.
\]

`optimal_composition(l, num_iters, degree=5, safety_factor_eps=0, cushion=0)` receives the initial lower endpoint, requested sequence length, polynomial degree, and two numeric safeguard parameters. In the directly scored degree-5 calls, it must return exactly `num_iters` finite three-coefficient sequences in execution order. The notebook also calls it with degree 5 and iterates over the result. For scoring, a grid value is transformed once by each returned degree-5 sequence, from first to last.

`PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor` receives a float32 tensor `G` of shape `[2,m,n]` and the positive step count listed in the workload table. It must return a `torch.Tensor` with exactly the same shape, all values finite, and one polar-factor approximation per batch member. It must not mutate any element of `G`.

The following concrete calls show the layouts. The coefficient values shown are valid but not accurate.

```python
>>> optimal_quintic(0.5, 1.0)
(1.0, 0.0, 0.0)               # a=1, b=0, c=0: p(x)=x

>>> optimal_composition(0.5, 2, degree=5,
...                     safety_factor_eps=0.0, cushion=0.0)
[(1.0, 0.0, 0.0),
 (1.0, 0.0, 0.0)]             # two steps, in application order

>>> G = torch.eye(2, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
>>> PolarExpress(G, steps=2)
tensor([[[1., 0.],
         [0., 1.]],
        [[1., 0.],
         [0., 1.]]])           # shape [batch=2, rows=2, columns=2]
```

The judge executes `plotsforpolar.ipynb` from the repository directory using `NotebookClient(timeout=60, kernel_name="python3", allow_errors=False)`. It then imports the module in the numerical runner and performs the calls above with the twelve exact records. Execution sets `MPLBACKEND=Agg`, `TORCH_COMPILE_DISABLE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `PYTHONHASHSEED=0`; PyTorch is restricted to one thread in the direct numerical checks.

## Correctness

Your code runs in one worker process (100 s for the notebook plus all twelve records, 60 s per
notebook cell) that receives only the inputs and public records, with read-only source and no
child processes. Tensor export is disabled before import. During each `PolarExpress` call only
dtype conversion, view/transpose/expand, vector norm, elementwise add/subtract/multiply/divide
and dense `bmm`, `mm`, `matmul` or `addmm` may be dispatched, at most `3 * steps` dense products
per two-member call; `matmul` forms are accepted only on real PyTorch objects. Do not print a line
beginning `FCS_RESULT=`.

Each of these scores 0: a broken rule of "What you may change" (rejected); a notebook error; an
import failure; a forbidden operation or operation-budget excess; `optimal_quintic` not three
finite coefficients; a scored composition without exactly `steps` steps of three finite
coefficients, or non-finite on the grid; a matrix result that is not a tensor of the input's shape,
not finite, or mutates the input; a crash or timeout.

## Workload

Each suite has these twelve records, in this order; the first six are the original records, the
last six the paper-regime stratum:

| `id` | `shape` | `l` | `steps` | `safety` | `cushion` |
| --- | --- | ---: | ---: | ---: | ---: |
| `square_0` | `[12, 12]` | 0.010 | 5 | 0.010 | 0.020 |
| `square_1` | `[12, 12]` | 0.020 | 5 | 0.008 | 0.015 |
| `tall_0` | `[18, 10]` | 0.030 | 4 | 0.006 | 0.012 |
| `tall_1` | `[18, 10]` | 0.050 | 4 | 0.004 | 0.010 |
| `wide_0` | `[10, 18]` | 0.080 | 3 | 0.003 | 0.008 |
| `wide_1` | `[10, 18]` | 0.120 | 3 | 0.002 | 0.006 |
| `paper_square_0` | `[256, 256]` | 0.001 | 5 | 0.010 | 0.02407327424182761 |
| `paper_square_1` | `[256, 256]` | 0.001 | 5 | 0.010 | 0.02407327424182761 |
| `paper_wide_0` | `[128, 512]` | 0.001 | 6 | 0.010 | 0.02407327424182761 |
| `paper_wide_1` | `[128, 512]` | 0.001 | 6 | 0.010 | 0.02407327424182761 |
| `paper_tall_0` | `[512, 128]` | 0.001 | 5 | 0.010 | 0.02407327424182761 |
| `paper_tall_1` | `[512, 128]` | 0.001 | 5 | 0.010 | 0.02407327424182761 |

`shape` is `[rows, columns]`; the scalar interval is `[l, 1.0]`; `steps` is both the
coefficient-sequence length and the matrix iteration count; `safety` and `cushion` are passed as
`safety_factor_eps` and `cushion`. Each record is a separately graded workload.

- **Scalar check** (every record): `optimal_quintic(l, 1.0)` and
  `optimal_composition(l, steps, degree=5, safety_factor_eps=safety, cushion=cushion)` are called
  first, on the grid `np.linspace(l, 1.0, 1000)` (float64). The notebook is also run in full; it
  calls `optimal_composition(l=1e-3, num_iters=10, degree=5, safety_factor_eps=1e-2,
  cushion=0.02)` and, via `plot_composition_arrows`, `optimal_composition(0.001, 12, ...)`.
- **Original matrices:** `G = Q_L diag(s) Q_R^T` with random orthonormal factors,
  `s = np.geomspace(r, 1.0, min(m, n))`, `r` drawn per member in `[l/2, 2l]`, times an arbitrary
  global scale, converted to float32.
- **Paper-regime matrices:** spectra drawn per member from three hidden families (log-spaced with
  `sigma_min/sigma_max` about `1e-4.5` to `1e-2.5`, power-law decay, low effective rank with a
  small tail), times an arbitrary scale; not tied to `l`, and the two members of a pair may differ.
- Records of the same shape are stacked in pairs, so `PolarExpress` receives `[2, m, n]` six times
  per suite. A float64 reduced SVD of each `G` is the target `UV^T`.

The development (train) and final suites use disjoint seeds (new factors and spectra for every
member); the records, dimensions, parameters and order are the same.

## Scoring

For workload \(i\) with returned steps \(p_t(x)=c_{t,0}x+c_{t,1}x^3+c_{t,2}x^5\), applied in order,
and grid \(D_i=\texttt{np.linspace}(l_i,1,1000)\); \(X_i\) the returned member and \(U_iV_i^T\) the
float64 SVD target:

\[
e_{s,i}=\max_{x\in D_i}\left|(p_{T_i}\circ\cdots\circ p_1)(x)-1\right|,\qquad
e_{p,i}=\frac{\lVert X_i-U_iV_i^T\rVert_F}{\lVert U_iV_i^T\rVert_F}
\]

\[
q_{s,i}=\mathrm{clip}_{[0,1]}\!\left(\frac{-\log_{10}\max(e_{s,i},10^{-15})}{12}\right),\quad
q_{p,i}=\mathrm{clip}_{[0,1]}\!\left(\frac{-\log_{10}\max(e_{p,i},10^{-15})}{8}\right),\quad
\mathrm{score}=100\cdot\frac{\frac16\sum_{i=1}^{6}q_{s,i}+\frac16\sum_{i=1}^{6}q_{p,i}}{2}
\]

over the six original workloads only. Anchors: errors of 1 or more score 0 for their component;
\(e_s=10^{-6}\) and \(e_p=10^{-4}\) everywhere scores 50; errors at the \(10^{-15}\) floor score
100. The unmodified tree scores 0; a minimal valid implementation scores `6.3334298557` on the
development suite.

**Beats the reference** when, on every one of the twelve workloads, all five quantities beat the
bar: the scalar error \(e_{s,i}\), the relative Frobenius error \(e_{p,i}\), the spectral error
\(\lVert X_i-U_iV_i^T\rVert_2\) and the top-subspace error
\(\lVert U_1U_1^TX_iV_1V_1^T-U_1V_1^T\rVert_F/\lVert U_1V_1^T\rVert_F\) (\(U_1,V_1\): oracle singular
vectors with singular value at least \(10^{-3}\) times the largest) strictly lower, and the
Frobenius cosine similarity to \(U_iV_i^T\) (0 for a zero output) strictly higher. The bar, per
workload and quantity, is the best of four reference runs on the same inputs (published bfloat16,
float32, per-record schedule, per-record schedule without the safety factor and normalisation
margin). Matrix quantities are computed in float64; there is no tolerance.

## Running locally

```sh
cd /workspace/repo
./frontier-build.sh    # py_compile and import polar_express.py
./frontier-bench.sh    # runs the notebook (120 s per cell; the judge allows 60 s)
```

## Environment

The runtime is a CPU-only Python 3.12 environment with 2 vCPUs, 4,096 MiB of memory, and 4,096 MiB of workspace storage. Installed numerical and notebook packages are NumPy 2.2.6, CPU PyTorch 2.8.0, Matplotlib 3.10.6, nbformat 5.10.4, nbclient 0.10.2, nbconvert 7.16.6, and ipykernel 6.30.1; Git is also installed. No GPU is available or required, and the task declares no additional dependencies.

Direct numerical execution fixes PyTorch and BLAS thread counts to one despite the two-vCPU allocation. The notebook uses a non-interactive Matplotlib backend and the installed `python3` IPython kernel.
