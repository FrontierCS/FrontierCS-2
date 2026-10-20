package estimator

import (
	"fmt"
	"math/big"
	"math/cmplx"
	"runtime"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/mod1"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// EvaluteMod1 applies a homomorphic mod Q on a vector scaled by Delta, scaled down to mod 1 :
//
//  1. Delta * (Q/Delta * I(X) + m(X)) (Delta = scaling factor, I(X) integer poly, m(X) message)
//  2. Delta * (I(X) + Delta/Q * m(X)) (divide by Q/Delta)
//  3. Delta * (Delta/Q * m(X)) (x mod 1)
//  4. Delta * (m(X)) (multiply back by Q/Delta)
//
// Since Q is not a power of two, but Delta is, then does an approximate division by the closest
// power of two to Q instead. Hence, it assumes that the input plaintext is already scaled by
// the correcting factor Q/2^{round(log(Q))}.
//
// !! Assumes that the input is normalized by 1/K for K the range of the approximation.
//
// Scaling back error correction by 2^{round(log(Q))}/Q afterward is included in the polynomial
func (e Estimator) EvaluateMod1New(elIn *Element, evm mod1.Parameters) (elOut *Element, err error) {
	return e.EvaluateMod1AndScaleNew(elIn, evm, 1)
}

func (e Estimator) EvaluateMod1AndScaleNew(elIn *Element, evm mod1.Parameters, scaling complex128) (elOut *Element, err error) {

	if elIn.Level < evm.LevelQ {
		return nil, fmt.Errorf("cannot Evaluate: ct.Level() < Mod1Parameters.LevelQ")
	}

	elOut = elIn.CopyNew()

	elOut.Level = evm.LevelQ

	// Normalize the modular reduction to mod by 1 (division by Q)
	elOut.Scale = evm.ScalingFactor()

	// Compute the scales that the ciphertext should have before the double angle
	// formula such that after it it has the scale it had before the polynomial
	// evaluation

	params := e.Parameters.Parameters

	Qi := params.Q()

	targetScale := elOut.Scale
	for i := 0; i < evm.DoubleAngle; i++ {
		targetScale = targetScale.Mul(rlwe.NewScale(Qi[elOut.Level-evm.Mod1Poly.Depth()-evm.DoubleAngle+i+1]))
		targetScale.Value.Sqrt(&targetScale.Value)
	}

	// Division by 1/2^r and change of variable for the Chebyshev evaluation
	if evm.Mod1Type == mod1.CosDiscrete || evm.Mod1Type == mod1.CosContinuous {
		offset := new(big.Float).Sub(&evm.Mod1Poly.B, &evm.Mod1Poly.A)
		offset.Mul(offset, new(big.Float).SetFloat64(evm.IntervalShrinkFactor()))
		offset.Quo(new(big.Float).SetFloat64(-0.5), offset)
		if err = e.Add(elOut, offset, elOut); err != nil {
			return nil, fmt.Errorf("e.Add: %w", err)
		}
	}

	// Double angle
	sqrt2pi := complex(evm.Sqrt2Pi, 0)

	var mod1Poly bignum.Polynomial
	if evm.Mod1InvPoly == nil {

		scaling := cmplx.Pow(scaling, complex(1/evm.IntervalShrinkFactor(), 0))

		mul := bignum.NewComplexMultiplier().Mul

		mod1Poly = evm.Mod1Poly.Clone()

		scalingPowBig := bignum.NewComplex().SetComplex128(scaling)

		for i := range mod1Poly.Coeffs {
			if mod1Poly.Coeffs[i] != nil {
				mul(mod1Poly.Coeffs[i], scalingPowBig, mod1Poly.Coeffs[i])
			}
		}

		sqrt2pi *= scaling

	} else {
		mod1Poly = evm.Mod1Poly
	}

	// Chebyshev evaluation
	if elOut, err = e.EvaluatePolynomialNew(elOut, mod1Poly, rlwe.NewScale(targetScale)); err != nil {
		return nil, fmt.Errorf("cannot Evaluate: %w", err)
	}

	for i := 0; i < evm.DoubleAngle; i++ {

		sqrt2pi *= sqrt2pi

		if err = e.MulRelin(elOut, elOut, elOut); err != nil {
			return nil, fmt.Errorf("e.MulRelin: %w", err)
		}

		if err = e.Mul(elOut, 2, elOut); err != nil {
			return nil, fmt.Errorf("e.Mul: %w", err)
		}

		if err = e.Add(elOut, -sqrt2pi, elOut); err != nil {
			return nil, fmt.Errorf("e.Add: %w", err)
		}

		if err = e.Rescale(elOut, elOut); err != nil {
			return nil, fmt.Errorf("e.Rescale: %w", err)
		}
	}

	// ArcSine
	if evm.Mod1InvPoly != nil {

		mul := bignum.NewComplexMultiplier().Mul

		mod1InvPoly := evm.Mod1InvPoly.Clone()

		scalingBig := bignum.NewComplex().SetComplex128(scaling)

		for i := range mod1InvPoly.Coeffs {
			if mod1InvPoly.Coeffs[i] != nil {
				mul(mod1InvPoly.Coeffs[i], scalingBig, mod1InvPoly.Coeffs[i])
			}
		}

		if elOut, err = e.EvaluatePolynomialNew(elOut, mod1InvPoly, elOut.Scale); err != nil {
			return nil, fmt.Errorf("cannot Evaluate: %w", err)
		}
	}

	runtime.GC()

	// Multiplies back by q
	elOut.Scale = elIn.Scale

	return
}
