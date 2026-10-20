# CKKS Precision Propagation at 2,048 Slots

## Problem
Build an estimator that predicts the numerical precision of CKKS computations, and make it cheap to run. For each circuit the estimator receives the same clear inputs as a real encrypted Lattigo computation and must predict the mean and the spread of the per-slot precision of the encrypted result. A submission is compared with a private reference estimator on two conditions at once:

1. **Accuracy**: its predictions are not worse than the reference estimator's.
2. **Time**: the estimator's own running time is strictly lower than the reference estimator's.

The supplied source tree is the estimator with seven kernels removed: multiplication, product accumulation, relinearization, rescaling, division with rounding noise, and two key-switching-noise methods (`estimator/operations.go`, `estimator/noise.go`). A submission must provide a complete, working estimator. It may fill in the seven kernels, and it may rewrite any other part of the editable surface described below, including how an estimator element is represented.

## Configurations
The configuration manifest is `/frontier/task/workload_manifest.json`. There are nine circuits: multiply/relinearize/rescale, T4096 power-basis composition, degree-255 polynomial evaluation, Goldschmidt inversion on [0.1,1.9], seven iterative cleaning steps, coefficients-to-slots DFT, slots-to-coefficients DFT, modular reduction with K=12, and bootstrapping. Every circuit uses ring dimension N=4096 (`LogN=12`) and 2048 complex slots. These are numerical benchmark parameters, not a recommendation for secure deployment.

The nine base configurations are the supplied experiment programs under `estimator/experiments/`. Twenty more configurations are derived from them by a parameter overlay that changes only the initial scale (2^45 or 2^55), the sizes of the rescaling primes after Q0, and the secret distribution (dense ternary with P=2/3, or fixed Hamming weight 192). Q0, chain length, LogP, the circuit algorithm and its depth stay fixed. Modular reduction and bootstrapping exist at scale 2^45 only, and the multiply/relinearize/rescale configuration has no variant. Dense sampling has expected weight 2N/3; it does not impose an exact weight.

Development feedback is a cheaper measurement that decides nothing: it measures only the six base configurations of multiply/relinearize/rescale, power basis, polynomial evaluation, Goldschmidt and both DFTs, with one pair of invocations each (see Scoring). It does not measure any scale or secret variant, nor iterative cleaning, modular reduction or bootstrapping. Final measures all 29 configurations with three pairs each. No development measurement is reused in final.

Keys and randomness are fresh for every invocation. Within an invocation the estimator and the encrypted computation use matching clear messages.

## Editable and fixed source
Everything below `estimator/` may be changed, added or removed, **except** the fixed set:

| Fixed | Why it is fixed |
| --- | --- |
| `estimator/experiments/` (every file) | the experiment programs: clear targets, key generation, the encrypted circuit, the order of operations |
| `estimator/stats.go` | the reported statistics |
| `estimator/utils.go` | the test vectors (`NewTestVector`, `NewTestVectorFromSeed`, `TruncatePlaintext`) and shared helpers (`NewFloat`, `NewTestRand`, `Round`, `DecompRNS`, ...) |
| `estimator/go.mod`, `estimator/go.sum` | the module and its pinned dependencies |
| in `estimator/dft.go`: the type `DFTMatrix` and its method `GenMatrices` | they build the clear target of the two DFT circuits |

The fixed files must stay byte for byte as supplied, and the two fixed declarations of `dft.go` must stay as supplied; the rest of `dft.go` is editable. Nothing outside `estimator/` can be changed. New files must be ordinary `.go` sources (no `_test.go`, no other file types) and may live in `estimator/` or in new sub-packages below it, but not under `estimator/experiments/`.

Editable code must be numerical code:

- Permitted imports: `fmt`, `errors`, `math`, `math/big`, `math/bits`, `math/cmplx`, `math/rand`, `math/rand/v2`, `container/heap`, `sort`, `slices`, `cmp`; `time` (only `time.Now`) and `runtime` (only `runtime.GC`); this module (`github.com/tuneinsight/ckks-noise-estimator` and its sub-packages other than `experiments`); and from Lattigo v6: `core/rlwe`, `schemes/ckks`, `ring` and its sub-packages, `utils`, `utils/bignum`, `utils/sampling`, `utils/structs`, `circuits/ckks/bootstrapping`, `circuits/ckks/dft`, `circuits/ckks/lintrans`, `circuits/ckks/mod1`, `circuits/ckks/polynomial`, `circuits/common/lintrans`, `circuits/common/polynomial`.
- Not permitted: `go` statements, channels and `select`; `init` functions; package-level variables whose initializer calls a function; blank or dot imports; compiler directives (`//go:`, `//export`, build constraints); file, process, network, reflection or unsafe access.

Numerical tables, types, functions and helper state are allowed. Hardcoded measurement output or inspection of judge data is invalid.

## Interface the fixed programs rely on
The experiment programs, the test-vector code and a small interface check call the estimator through the following names. They must keep these signatures and meanings; everything behind them is yours.

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

`NewElement` accepts `[]*bignum.Complex`, `[]complex128`, `[]*big.Float`, `[]float64` or `nil`, at most `MaxSlots` values, zero-padded; the element represents those clear slot values at the given degree, level and scale, before any noise. `Decrypt` returns one value per slot (2048 in every measured circuit): the clear value the element represents plus the estimator's error for that slot, divided by the scale. The experiment programs read and write `Element.Level` and read `Element.Scale`, so these two fields must hold the remaining modulus level and the current scale after every operation. Operands may be an `*Element`, a scalar (`complex128`, `float64`, `int`, `int64`, `uint`, `uint64`, `*big.Int`, `*big.Float`, `*bignum.Complex`) or, where the supplied code accepts it, a slot vector `[]*bignum.Complex`. The output of an operation may alias an input. `Rescale` consumes one level and returns an error at level zero. Error-returning methods return `nil` on valid inputs.

For example, an element holding the clear slot `1+0i` at level 1 and scale 2^45, multiplied by one holding `0.5+0.25i` with the same metadata, represents `0.5+0.25i`; after relinearization and rescaling it is at level 0 with scale `2^90/Q[1]`, and `Decrypt` returns approximately `0.5+0.25i` in that slot. The exact value is randomized by the propagated noise.

The fixed programs are compiled against your package: a missing or changed name is a build failure.

## Scoring
For each configuration the final evaluation runs three pairs of invocations; development feedback runs one pair on each of its six configurations and applies the same comparisons to it. A pair is one complete run of the experiment program with your estimator and one with the reference estimator, back to back, each with fresh keys and randomness; which of the two runs first alternates from pair to pair.

**Accuracy.** In one invocation, let the two discrepancies be the absolute difference in bits between predicted and actual `AVGLog2Prec.L2` and between predicted and actual `STDLog2Prec.L2`; slot precision is `-log2(|error|)`, AVG is the arithmetic mean over slots and STD the standard deviation over slots (the coefficients-to-slots circuit keeps both of its outputs in the same statistics object, as the supplied program does). For configuration w, `a_w` and `s_w` are the means of those discrepancies over your invocations of w (three in final, one in development). With W configurations:

```text
E = sum_w(a_w + s_w) / (2 W)
score = 100 / (1 + E / (1 bit))
```

This continuous score depends only on your own accuracy. The accuracy condition requires, for every configuration and both statistics, `your mean discrepancy - reference mean discrepancy <= frozen band`. The bands are configuration-specific tolerances for run-to-run variation of the reference against itself; they are not published.

**Time.** A trusted clock in the experiment program measures the time spent in every statement that can reach package `estimator`: constructing the estimator, creating the test vectors, each estimator operation, and `Decrypt`, followed by a garbage collection that is charged to the estimator. Key generation, the encrypted computation, the clear target and the statistics of the encrypted result are not on that clock. Programs run on one scheduler thread. The time condition requires, for every configuration and in each of its pairs, `your estimator time < reference estimator time`.

`beats_reference` is 1 exactly when both conditions hold for every configuration. There is no compensation between configurations, between the two accuracy statistics, or between accuracy and time. A valid submission that fails a condition keeps its continuous score. Missing, duplicate, negative or nonfinite observations, invalid patches, changes to the fixed set, interface or policy violations, build failures, crashes and timeouts score zero and cannot pass.

Development feedback reports `candidate_avg_log2_prec_l2_abs_discrepancy_mean_bits` and `candidate_std_log2_prec_l2_abs_discrepancy_mean_bits` (your mean absolute discrepancies over the six development configurations), `candidate_estimator_seconds_mean` (your mean estimator time per invocation over those six), and `beat_spec_progress`: for each of the three comparisons (the two accuracy statistics and estimator time) the number of development configurations `passed`, `failed` and `total`, with `pass_percent` and `fail_percent`. It reports no reference value and no per-configuration decision. Because development uses a single pair, its time count is noisy: an estimator exactly as fast as the reference wins a single pair about half the time, whereas final requires all three pairs of all 29 configurations.

## Submission
Use `/frontier/task/prepare_workspace.sh` to copy the supplied base into `/workspace/repo`. Make your edits there, then use `/frontier/task/make_submission.sh` to produce `/workspace/payload/solution.patch`. Package it as the uncompressed POSIX tar described in `/frontier/task/README.md` and use the Candidate's `frontier-submit` helper.

The packaging helper stages tracked and new files, checks every changed path, then emits the patch. Empty patches are invalid. Keep local build products in `.frontier-cache`, which is excluded from the patch. Patches are limited to 262144 bytes and 32 changed files.

## Resources
Each role has 2 CPUs. The Agent hard memory limit is 2048 MiB; the Verifier hard memory limit is 4096 MiB. Workspace storage is 8192 MiB. Candidate and reference invocations execute serially within the Verifier. Public build and benchmark helpers use a 1 GiB Go soft memory target; the Verifier uses 3 GiB. A verification has a 600-second total limit including compilation, the interface and policy checks and all invocations; a submission whose invocations do not finish inside it is a timeout. Independent final verification has its own 600-second limit. Development allows at most 100 calls and 7200 seconds of cumulative verification; a development verification (compilation, the checks, and six configurations with one pair each) takes on the order of one minute, so both limits are of the same order. The inclusive answering window is 18000 seconds (five hours), including development verification. The independent 600-second final evaluation starts after Agent exit and is outside that answering window.

Go1.23.12 and Lattigo v6.1.1 are provided offline. `bash frontier-build.sh` builds with `go build -p=1`; `bash frontier-bench.sh` runs the multiply/relinearize/rescale program. Use writable task-local caches. No GPU or external dataset is required.
