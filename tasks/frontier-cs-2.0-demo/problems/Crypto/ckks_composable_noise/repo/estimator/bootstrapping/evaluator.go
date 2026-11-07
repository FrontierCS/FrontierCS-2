package estimator

import (
	"fmt"
	"math"
	"math/big"

	"github.com/tuneinsight/ckks-noise-estimator"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/mod1"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

type Evaluator struct {
	bootstrapping.Parameters

	ResidualParameters      estimator.Estimator
	BootstrappingParameters estimator.Estimator

	S2CDFTMatrix   estimator.DFTMatrix
	C2SDFTMatrix   estimator.DFTMatrix
	Mod1Parameters mod1.Parameters

	EphemeralSecret []*bignum.Complex
}

func NewEvaluator(btpParams bootstrapping.Parameters) Evaluator {

	eval := Evaluator{
		Parameters:              btpParams,
		ResidualParameters:      estimator.NewEstimator(btpParams.ResidualParameters),
		BootstrappingParameters: estimator.NewEstimator(btpParams.BootstrappingParameters),
	}

	if btpParams.EphemeralSecretWeight != 0 {
		eval.EphemeralSecret = eval.BootstrappingParameters.SampleSecretKey(btpParams.EphemeralSecretWeight)
	}

	eval.initialize(btpParams)

	return eval
}

func (eval *Evaluator) initialize(btpParams bootstrapping.Parameters) (err error) {

	params := btpParams.BootstrappingParameters

	if eval.Mod1Parameters, err = mod1.NewParametersFromLiteral(params, btpParams.Mod1ParametersLiteral); err != nil {
		return
	}

	// [-K, K]
	K := eval.Mod1Parameters.K

	// Correcting factor for approximate division by Q
	// The second correcting factor for approximate multiplication by Q is included in the coefficients of the EvalMod polynomials
	qDiff := eval.Mod1Parameters.QDiff

	// If the scale used during the EvalMod step is smaller than Q0, then we cannot increase the scale during
	// the EvalMod step to get a free division by MessageRatio, and we need to do this division (totally or partly)
	// during the CoeffstoSlots step
	qDiv := eval.Mod1Parameters.ScalingFactor().Float64() / math.Exp2(math.Round(math.Log2(float64(params.Q()[0]))))

	// Sets qDiv to 1 if there is enough room for the division to happen using scale manipulation.
	if qDiv > 1 {
		qDiv = 1
	}

	// CoeffsToSlots vectors
	// Change of variable for the evaluation of the Chebyshev polynomial + cancelling factor for the DFT and SubSum + eventual scaling factor for the double angle formula

	scale := eval.BootstrappingParameters.DefaultScale().Float64()
	offset := eval.Mod1Parameters.ScalingFactor().Float64() / eval.Mod1Parameters.MessageRatio()

	C2SScaling := new(big.Float).SetFloat64(qDiv / (K * qDiff))
	StCScaling := new(big.Float).SetFloat64(scale / offset)

	eval.C2SDFTMatrix = estimator.DFTMatrix{MatrixLiteral: btpParams.CoeffsToSlotsParameters}
	if eval.C2SDFTMatrix.Scaling == nil {
		eval.C2SDFTMatrix.Scaling = C2SScaling
	} else {
		eval.C2SDFTMatrix.Scaling = new(big.Float).Mul(btpParams.CoeffsToSlotsParameters.Scaling, C2SScaling)
	}

	eval.C2SDFTMatrix.GenMatrices(btpParams.BootstrappingParameters.LogN(), 128)

	eval.S2CDFTMatrix = estimator.DFTMatrix{MatrixLiteral: btpParams.SlotsToCoeffsParameters}
	if eval.S2CDFTMatrix.Scaling == nil {
		eval.S2CDFTMatrix.Scaling = StCScaling
	} else {
		eval.S2CDFTMatrix.Scaling = new(big.Float).Mul(btpParams.SlotsToCoeffsParameters.Scaling, StCScaling)
	}

	eval.S2CDFTMatrix.GenMatrices(btpParams.BootstrappingParameters.LogN(), 128)

	return
}

// checks if the current message ratio is greater or equal to the last prime times the target message ratio.
func checkMessageRatio(el *estimator.Element, msgRatio float64, r *ring.Ring) bool {
	level := el.Level
	currentMessageRatio := rlwe.NewScale(r.ModulusAtLevel[level])
	currentMessageRatio = currentMessageRatio.Div(el.Scale)
	return currentMessageRatio.Cmp(rlwe.NewScale(r.SubRings[level].Modulus).Mul(rlwe.NewScale(msgRatio))) > -1
}

func (eval Evaluator) Bootstrap(elIn *estimator.Element) (elOut *estimator.Element, err error) {

	if _, err = eval.ScaleDown(elIn); err != nil {
		return nil, fmt.Errorf("eval.ScaleDown: %w", err)
	}

	if err = eval.ModUp(elIn); err != nil {
		return nil, fmt.Errorf("eval.ModUp: %w", err)
	}

	elReal, elImag, err := eval.CoeffsToSlotsNew(elIn)

	if err != nil {
		return nil, fmt.Errorf("eval.CoeffsToSlotsNew: %w", err)
	}

	if elReal, err = eval.EvalModNew(elReal); err != nil {
		return nil, fmt.Errorf("eval.EvalModNew: %w", err)
	}

	if elImag != nil {
		if elImag, err = eval.EvalModNew(elImag); err != nil {
			return nil, fmt.Errorf("eval.EvalModNew: %w", err)
		}
	}

	return eval.SlotsToCoeffsNew(elReal, elImag)
}

func (eval Evaluator) ScaleDown(el *estimator.Element) (*rlwe.Scale, error) {

	est := eval.BootstrappingParameters

	params := &eval.BootstrappingParameters.Parameters

	r := params.RingQ()

	// Removes unecessary primes
	for el.Level != 0 && checkMessageRatio(el, eval.Mod1Parameters.MessageRatio(), r) {
		el.Level--
	}

	// Current Message Ratio
	currentMessageRatio := rlwe.NewScale(r.ModulusAtLevel[el.Level])
	currentMessageRatio = currentMessageRatio.Div(el.Scale)

	// Desired Message Ratio
	targetMessageRatio := rlwe.NewScale(eval.Mod1Parameters.MessageRatio())

	// (Current Message Ratio) / (Desired Message Ratio)
	scaleUp := currentMessageRatio.Div(targetMessageRatio)

	if scaleUp.Cmp(rlwe.NewScale(0.5)) == -1 {
		return nil, fmt.Errorf("initial Q/Scale = %f < 0.5*Q[0]/MessageRatio = %f", currentMessageRatio.Float64(), targetMessageRatio.Float64())
	}

	scaleUpBigint := scaleUp.BigInt()

	if err := est.Mul(el, scaleUpBigint, el); err != nil {
		return nil, fmt.Errorf("est.Mul: %w", err)
	}

	el.Scale = el.Scale.Mul(rlwe.NewScale(scaleUpBigint))

	// errScale = CtIn.Scale/(Q[0]/MessageRatio)
	targetScale := new(big.Float).SetPrec(256).SetInt(r.ModulusAtLevel[0])
	targetScale.Quo(targetScale, new(big.Float).SetFloat64(eval.Mod1Parameters.MessageRatio()))

	if el.Level != 0 {
		if err := est.Rescale(el, el); err != nil {
			return nil, fmt.Errorf("est.Rescale: %w", err)
		}
	}

	// Rescaling error (if any)
	errScale := el.Scale.Div(rlwe.NewScale(targetScale))

	return &errScale, nil
}

func (eval Evaluator) ModUp(el *estimator.Element) (err error) {

	est := eval.BootstrappingParameters

	var H int
	if eval.EphemeralSecret != nil {
		if err = est.KeySwitch(el, eval.BootstrappingParameters.Sk[0]); err != nil {
			return fmt.Errorf("est.KeySwitch: %w", err)
		}
		H = eval.EphemeralSecretWeight
	} else {
		H = eval.BootstrappingParameters.H
	}

	K := new(big.Float).SetPrec(128)

	Q := eval.BootstrappingParameters.Q[0]

	source := estimator.NewTestRand()
	irwinHall := func() *big.Float {
		var d float64
		for i := 0; i < H+1; i++ {
			d += source.Float64(0, 1)
		}

		K.SetFloat64(math.Floor(d) - float64(H>>1))

		return new(big.Float).Mul(K, &Q)
	}

	values := make([]*bignum.Complex, est.MaxSlots())

	for i := range values {
		values[i] = &bignum.Complex{irwinHall(), irwinHall()}
	}

	eval.BootstrappingParameters.Encoder.FFT(values, est.LogMaxSlots())

	coeffs := el.Value[0]
	for i := range coeffs {
		coeffs[i].Add(coeffs[i], values[i])
	}

	el.Level = est.MaxLevel()

	if eval.EphemeralSecret != nil {
		if err = est.KeySwitch(el, eval.EphemeralSecret); err != nil {
			return fmt.Errorf("est.KeySwitch: %w", err)
		}
	}

	// Scale the message from Q0/|m| to QL/|m|, where QL is the largest modulus used during the bootstrapping.
	if scale := (eval.Mod1Parameters.ScalingFactor().Float64() / eval.Mod1Parameters.MessageRatio()) / el.Scale.Float64(); scale > 1 {
		if err = est.ScaleUp(el, rlwe.NewScale(scale)); err != nil {
			return fmt.Errorf("est.ScaleUp: %w", err)
		}
	}

	return
}

func (eval Evaluator) CoeffsToSlotsNew(el *estimator.Element) (elReal, elImag *estimator.Element, err error) {
	return eval.BootstrappingParameters.CoeffsToSlotsNew(el, eval.C2SDFTMatrix)
}

func (eval Evaluator) SlotsToCoeffsNew(elReal, elImag *estimator.Element) (el *estimator.Element, err error) {
	return eval.BootstrappingParameters.SlotsToCoeffsNew(elReal, elImag, eval.S2CDFTMatrix)
}

func (eval Evaluator) EvalModNew(elIn *estimator.Element) (elOut *estimator.Element, err error) {
	if elOut, err = eval.BootstrappingParameters.EvaluateMod1New(elIn, eval.Mod1Parameters); err != nil {
		return nil, fmt.Errorf("eval.EvaluateMod1New: %w", err)
	}
	elOut.Scale = eval.BootstrappingParameters.DefaultScale()
	return
}
