# CKKS Precision Propagation at 2,048 Slots

Build an estimator that predicts the numerical precision of CKKS computations, accurately and
cheaply. For each circuit it receives the same clear inputs as the real encrypted Lattigo
computation and must predict the mean and the spread of the per-slot precision of its result.

## Task

The supplied tree is the noise estimator of "Accurate and Composable Noise Estimates for CKKS"
(Lattigo v6.1.1, Go 1.23.12) with seven kernels removed from `estimator/operations.go` and
`estimator/noise.go`: multiplication, product accumulation, relinearization, rescaling, division
with rounding noise, and two key-switching-noise methods. Submit a complete, working estimator:
fill in the kernels, and rewrite anything else in the editable surface if you wish, including how
an element is represented.

## What you may change

- Any `.go` file below `estimator/`: change, add (also in new sub-packages) or delete, at most 32
  files per patch.
- Fixed, byte for byte: everything under `estimator/experiments/`, `estimator/stats.go`,
  `estimator/utils.go`, `estimator/go.mod`, `estimator/go.sum`, and in `estimator/dft.go` the type
  `DFTMatrix` and its method `GenMatrices` (the rest of `dft.go` is editable). Nothing outside
  `estimator/` may change; no `_test.go` files and no non-`.go` files.
- Permitted imports: `fmt`, `errors`, `math`, `math/big`, `math/bits`, `math/cmplx`, `math/rand`,
  `math/rand/v2`, `container/heap`, `sort`, `slices`, `cmp`; `time` (only `time.Now`) and `runtime`
  (only `runtime.GC`); this module (`github.com/tuneinsight/ckks-noise-estimator` and its
  sub-packages other than `experiments`); from Lattigo v6: `core/rlwe`, `schemes/ckks`, `ring` and
  its sub-packages, `utils`, `utils/bignum`, `utils/sampling`, `utils/structs`,
  `circuits/ckks/bootstrapping`, `circuits/ckks/dft`, `circuits/ckks/lintrans`,
  `circuits/ckks/mod1`, `circuits/ckks/polynomial`, `circuits/common/lintrans`,
  `circuits/common/polynomial`.
- Forbidden: `go` statements, channels and `select`; `init` functions; package-level variables
  whose initializer contains a call, a conversion or a function literal; blank or dot imports;
  the substrings `//go:`, `//export` and `+build` anywhere (comments and strings included); a
  package `main`, or a file in `estimator/` whose package is not `estimator`; file, process,
  network, reflection or unsafe access; hardcoded measurement output or inspection of judge data. Numerical tables, types and helper state are
  allowed.

## Interface

The fixed experiment programs, the test-vector code and an interface check call the estimator
through these names; keep their signatures and meanings (a missing or changed name is a build
failure):

```go
// package estimator
func NewEstimator(p ckks.Parameters) (e Estimator)
type Estimator struct { Parameters ckks.Parameters; /* ... */ }
type Element struct { Level int; Scale rlwe.Scale; /* ... */ }

func (e Estimator) NewElement(v interface{}, Degree, Level int, scale rlwe.Scale) *Element
func (e Estimator) MaxLevel() int
func (e Estimator) DefaultScale() rlwe.Scale
func (e Estimator) AddEncodingNoise(el *Element)
func (e Estimator) AddEncryptionNoisePk(el *Element)
func (e Estimator) AddEncryptionNoiseSk(el *Element)
func (e Estimator) Decrypt(el *Element) (values []*bignum.Complex)

func (e Estimator) Add(op0 *Element, op1 rlwe.Operand, op2 *Element) (err error)
func (e Estimator) Mul(op0 *Element, op1 rlwe.Operand, op2 *Element) (err error)
func (e Estimator) MulRelin(op0 *Element, op1 rlwe.Operand, op2 *Element) (err error)
func (e Estimator) Rescale(op0, op1 *Element) (err error)
func (e Estimator) ScaleUp(op0 *Element, scale rlwe.Scale) (err error)
func (e Estimator) EvaluatePolynomialNew(elIn *Element, poly interface{}, targetScale rlwe.Scale) (elOut *Element, err error)
func (e Estimator) EvaluateMod1New(elIn *Element, evm mod1.Parameters) (elOut *Element, err error)
func (e Estimator) GoldschmidtDivisionNew(el *Element, log2min float64) (a *Element, err error)
func (e Estimator) CoeffsToSlotsNew(el *Element, mat DFTMatrix) (elReal, elImag *Element, err error)
func (e Estimator) SlotsToCoeffs(elReal, elImag *Element, mat DFTMatrix, el *Element) (err error)

func NewPowerBasis(ct *Element, basis bignum.Basis) (p *PowerBasis)
type PowerBasis struct { Value map[int]*Element; /* ... */ }
func (p *PowerBasis) GenPower(n int, lazy bool, e Estimator) (err error)

// package estimator/bootstrapping
func NewEvaluator(btpParams bootstrapping.Parameters) Evaluator
type Evaluator struct { ResidualParameters estimator.Estimator; /* ... */ }
func (eval Evaluator) Bootstrap(elIn *estimator.Element) (elOut *estimator.Element, err error)
```

`NewElement` accepts `[]*bignum.Complex`, `[]complex128`, `[]*big.Float`, `[]float64` or `nil`, at
most `MaxSlots` values, zero-padded; the element represents those clear values at the given
degree, level and scale, before noise. `Decrypt` returns one value per slot (2048 in every
circuit): the clear value plus the estimator's error for that slot, divided by the scale.
`Element.Level` and `Element.Scale` must hold the remaining level and current scale after every
operation (the programs read and write `Level`, read `Scale`). Operands may be an `*Element`, a
scalar (`complex128`, `float64`, `int`, `int64`, `uint`, `uint64`, `*big.Int`, `*big.Float`,
`*bignum.Complex`) or, where the supplied code accepts it, a `[]*bignum.Complex`. Outputs may alias
inputs. `Rescale` consumes one level and errors at level zero; error-returning methods return
`nil` on valid inputs.

Example: an element holding `1+0i` at level 1, scale 2^45, multiplied by one holding `0.5+0.25i`
with the same metadata represents `0.5+0.25i`; after relinearization and rescaling it is at level
0 with scale `2^90/Q[1]`, and `Decrypt` returns about `0.5+0.25i` (randomized by the propagated
noise).

## Correctness

A run scores 0 and cannot pass when any of these happens: the patch is rejected (it does not
apply, changes the fixed set or the two fixed `dft.go` declarations, or breaks the source policy
above); the package does not build; the interface check fails; an invocation crashes, times out
(the verification time limit covers compilation, the checks and all invocations), prints more
than 1 MiB to stdout or stderr, or yields missing, duplicate, negative or non-finite
observations.

The interface check runs your estimator at `LogN=10`, `LogQ={55,45}`, `LogP={60}`, scale 2^45: after
`AddEncodingNoise`, `AddEncryptionNoisePk`/`Sk`, `Add`, `MulRelin` (level stays 1), `Rescale`
(level 0, scale divided by `Q[1]` within 1e-9), scalar `Mul` by 2, `ScaleUp` and
`NewPowerBasis(.., Chebyshev).GenPower(2, false, e)`, `Decrypt` slot 0 must be within 1e-6 of the
exact value, and a second `Rescale` at level 0 must return an error. An estimator that works only
at `LogN=12` fails.

## Workload

The manifest is `/frontier/task/repo/workload_manifest.json`. Nine circuits, all at N=4096
(`LogN=12`) with 2048 complex slots: multiply/relinearize/rescale, T4096 power basis, degree-255
polynomial evaluation, Goldschmidt inversion on [0.1,1.9], seven iterative cleaning steps,
coefficients-to-slots DFT, slots-to-coefficients DFT, modular reduction with K=12, bootstrapping.
These nine base configurations are the programs under `estimator/experiments/`; twenty more vary
only the initial scale (2^45 or 2^55), the rescaling-prime sizes after Q0, and the secret (dense
ternary with P=2/3, expected weight 2N/3, or Hamming weight 192). Modular reduction and
bootstrapping exist at 2^45 only; multiply/relinearize/rescale has no variant. Within an
invocation the estimator and the encrypted computation use matching clear messages.

| suite | configurations | pairs each |
|---|---|---|
| train (feedback, gates nothing) | 6 base: mul/relin/rescale, power basis, polynomial, Goldschmidt, both DFTs | 1 |
| final | all 29 | 3 |

A pair is one complete run of the experiment program with your estimator and one with the
reference, back to back, first position alternating, each with fresh keys and randomness. Timed:
a trusted clock covers every statement that can reach package `estimator` (construction, test
vectors, each operation, `Decrypt`, and a garbage collection charged to the estimator), on one
scheduler thread; key generation, the encrypted computation, the clear target and the statistics
are not timed.

## Scoring

In one invocation the two discrepancies are |predicted − actual| in bits of `AVGLog2Prec.L2` and of
`STDLog2Prec.L2` (slot precision `-log2(|error|)`, AVG the mean and STD the standard deviation over
slots; coefficients-to-slots keeps both outputs in one statistics object). `a_w`, `s_w` are their
means over your invocations of configuration w; W is the number of configurations:

```text
E = sum_w (a_w + s_w) / (2 W)
score = 100 / (1 + E / (1 bit))
```

Anchors: E = 0 gives 100; E = 1 bit gives 50; the reference scored about 96.0 in a train run. The
score depends only on your accuracy.

**Beats the reference** when, on every configuration: `avg_log2_prec_l2_abs_discrepancy_bits` and
`std_log2_prec_l2_abs_discrepancy_bits` each satisfy `your mean discrepancy − reference mean
discrepancy <= band` (frozen per-configuration bands, not published), and `estimator_seconds` is
strictly lower than the reference's in every pair. Nothing compensates. With one pair, train time
counts are noisy: an estimator as fast as the reference wins a single pair about half the time.

## Running locally

`sh frontier-build.sh` builds the package (`go build -p=1`); `sh frontier-bench.sh` runs the
multiply/relinearize/rescale program. Keep build products in `.frontier-cache` (excluded from the
patch).

## Environment

Go 1.23.12 and Lattigo v6.1.1 offline. The Verifier runs candidate and reference invocations
serially with 2 CPUs, 4096 MiB memory and a 3 GiB Go soft memory target; the public helpers use 1
GiB. Agent: 2 CPUs, 2048 MiB, 8192 MiB workspace.
