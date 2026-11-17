package estimator

import (
	"fmt"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

func (e Estimator) AddNew(op0 *Element, op1 rlwe.Operand) (op2 *Element, err error) {
	op2 = op0.CopyNew()
	return op2, e.Add(op0, op1, op2)
}

// Add adds elIn to op1 and writes the result on p, also returns p.
func (e Estimator) Add(op0 *Element, op1 rlwe.Operand, op2 *Element) (err error) {

	switch op1 := op1.(type) {
	case *Element:

		var tmp0, tmp1 *Element

		switch op0.Scale.Cmp(op1.Scale) {
		case -1:

			ratio := op1.Scale.Div(op0.Scale)
			// Only scales up if int(ratio) >= 2
			if ratio.Float64() >= 2.0 {
				tmp0 = op0.CopyNew()
				if err = e.Mul(tmp0, ratio.BigInt(), tmp0); err != nil {
					return fmt.Errorf("e.Mul: %w", err)
				}
				op2.Scale = op1.Scale
			} else {
				tmp0 = op0
			}

			tmp1 = op1

		case 0:
			tmp0 = op0
			tmp1 = op1
			op2.Scale = tmp0.Scale
		case 1:
			ratio := op0.Scale.Div(op1.Scale)
			// Only scales up if int(ratio) >= 2
			if ratio.Float64() >= 2.0 {
				tmp1 = op1.CopyNew()
				if err = e.Mul(tmp1, ratio.BigInt(), tmp1); err != nil {
					return fmt.Errorf("e.Mul: %w", err)
				}
				op2.Scale = op0.Scale
			} else {
				tmp1 = op1
			}

			tmp0 = op0
		}

		op2.Level = min(op0.Level, op1.Level)
		op2.Degree = max(op0.Degree, op1.Degree)

		for i := 0; i < op2.Degree+1; i++ {
			m0 := tmp0.Value[i]
			m1 := tmp1.Value[i]
			m2 := op2.Value[i]
			for j := range m2 {
				m2[j].Add(m0[j], m1[j])
			}
		}

	case complex128, float64, int, int64, uint, uint64, *big.Int, *big.Float, *bignum.Complex:

		m0 := op0.Value[0]
		m2 := op2.Value[0]

		bComplex := bignum.ToComplex(op1, prec)

		bComplex[0].Mul(bComplex[0], &op2.Scale.Value)
		bComplex[1].Mul(bComplex[1], &op2.Scale.Value)

		for i := range m2 {
			m2[i].Add(m0[i], bComplex)
		}

	case []*bignum.Complex:

		bComplex := &bignum.Complex{new(big.Float).SetPrec(prec), new(big.Float).SetPrec(prec)}

		m2 := op2.Value[0]

		scale := &op2.Scale.Value

		m0 := op0.Value[0]

		for i := range op1 {
			bComplex[0].Mul(op1[i][0], scale)
			bComplex[1].Mul(op1[i][1], scale)
			Round(bComplex[0])
			Round(bComplex[1])

			m2[i][0].Add(m0[i][0], bComplex[0])
			m2[i][1].Add(m0[i][1], bComplex[1])
		}

	default:
		return fmt.Errorf("invalid op1.(type): must be *Element, complex128, float64, int, int64, uint, uint64, *big.Int, *big.Float, *bignum.Complex, []complex128, []float64, []*big.Float or []*bignum.Complex, but is %T", op1)
	}

	return
}

func (e Estimator) SubNew(op0 *Element, op1 rlwe.Operand) (op2 *Element, err error) {
	op2 = op0.CopyNew()
	return op2, e.Sub(op0, op1, op2)
}

// Sub subtracts op1 to op0 and writes the result on op2.
func (e Estimator) Sub(op0 *Element, op1 rlwe.Operand, op2 *Element) (err error) {

	switch op1 := op1.(type) {
	case *Element:

		var tmp0, tmp1 *Element

		switch op0.Scale.Cmp(op1.Scale) {
		case -1:

			ratio := op1.Scale.Div(op0.Scale)
			// Only scales up if int(ratio) >= 2
			if ratio.Float64() >= 2.0 {
				tmp0 = op0.CopyNew()
				if err = e.Mul(tmp0, ratio.BigInt(), tmp0); err != nil {
					return fmt.Errorf("e.Mul: %w", err)
				}
				op2.Scale = op1.Scale
			} else {
				tmp0 = op0
			}

			tmp1 = op1

		case 0:
			tmp0 = op0
			tmp1 = op1
			op2.Scale = tmp0.Scale
		case 1:
			ratio := op0.Scale.Div(op1.Scale)
			// Only scales up if int(ratio) >= 2
			if ratio.Float64() >= 2.0 {
				tmp1 = op1.CopyNew()
				if err = e.Mul(tmp1, ratio.BigInt(), tmp1); err != nil {
					return fmt.Errorf("e.Mul: %w", err)
				}
				op2.Scale = op0.Scale
			} else {
				tmp1 = op1
			}

			tmp0 = op0
		}

		op2.Level = min(op0.Level, op1.Level)
		op2.Degree = max(op0.Degree, op1.Degree)

		for i := 0; i < op2.Degree+1; i++ {
			m0 := tmp0.Value[i]
			m1 := tmp1.Value[i]
			m2 := op2.Value[i]
			for j := range m2 {
				m2[j].Sub(m0[j], m1[j])
			}
		}

	case complex128, float64, int, int64, uint, uint64, *big.Int, *big.Float, *bignum.Complex:

		m0 := op0.Value[0]
		m2 := op2.Value[0]

		bComplex := bignum.ToComplex(op1, prec)

		bComplex[0].Mul(bComplex[0], &op2.Scale.Value)
		bComplex[1].Mul(bComplex[1], &op2.Scale.Value)

		for i := range m2 {
			m2[i].Sub(m0[i], bComplex)
		}

	case []*bignum.Complex:

		bComplex := &bignum.Complex{new(big.Float).SetPrec(prec), new(big.Float).SetPrec(prec)}

		m2 := op2.Value[0]

		scale := &op2.Scale.Value

		m0 := op0.Value[0]

		for i := range op1 {
			bComplex[0].Mul(op1[i][0], scale)
			bComplex[1].Mul(op1[i][1], scale)
			Round(bComplex[0])
			Round(bComplex[1])
			m2[i][0].Sub(m0[i][0], bComplex[0])
			m2[i][1].Sub(m0[i][1], bComplex[1])
		}

	default:
		return fmt.Errorf("invalid op1.(type): must be *Element, complex128, float64, int, int64, uint, uint64, *big.Int, *big.Float, *bignum.Complex, []complex128, []float64, []*big.Float or []*bignum.Complex, but is %T", op1)
	}

	return
}

func (e Estimator) MulRelin(op0 *Element, op1 rlwe.Operand, op2 *Element) (err error) {

	if err = e.Mul(op0, op1, op2); err != nil {
		return fmt.Errorf("e.Mul: %w", err)
	}

	if err = e.Relinearize(op2, op2); err != nil {
		return fmt.Errorf("e.Relinearize: %w", err)
	}

	return
}

func (e Estimator) MulRelinNew(op0 *Element, op1 rlwe.Operand) (op2 *Element, err error) {

	if op2, err = e.MulNew(op0, op1); err != nil {
		return nil, fmt.Errorf("e.MulNew: %w", err)
	}

	if err = e.Relinearize(op2, op2); err != nil {
		return nil, fmt.Errorf("e.Relinearize: %w", err)
	}

	return
}

func (e Estimator) MulNew(op0 *Element, op1 rlwe.Operand) (op2 *Element, err error) {
	op2 = op0.CopyNew()
	return op2, e.Mul(op0, op1, op2)
}

func (e Estimator) Mul(op0 *Element, op1 rlwe.Operand, op2 *Element) (err error) {
	panic("implement Mul")
}

func (e Estimator) MulThenAdd(op0 *Element, op1 rlwe.Operand, op2 *Element) (err error) {
	panic("implement MulThenAdd")
}

func (e Estimator) ScaleUp(op0 *Element, scale rlwe.Scale) (err error) {
	if err = e.Mul(op0, scale.Uint64(), op0); err != nil {
		return
	}
	op0.Scale = op0.Scale.Mul(scale)
	return
}

func (e Estimator) SetScale(op0 *Element, scale rlwe.Scale) (err error) {
	ratioFlo := scale.Div(op0.Scale).Value
	if err = e.Mul(op0, &ratioFlo, op0); err != nil {
		return
	}

	if !ratioFlo.IsInt() {
		if err = e.Rescale(op0, op0); err != nil {
			return
		}
	}

	op0.Scale = scale
	return
}

func (e Estimator) KeySwitch(op0 *Element, sk []*bignum.Complex) (err error) {
	if op0.Degree != 1 {
		return fmt.Errorf("degree != 1")
	}

	e.AddKeySwitchingNoise(op0, sk)

	return
}

func (e Estimator) RotateNew(op0 *Element, k int) (op1 *Element, err error) {
	op1 = op0.CopyNew()
	return op1, e.Rotate(op1, k, op1)
}

func (e Estimator) Rotate(op0 *Element, k int, op1 *Element) (err error) {

	if op0.Degree != 1 {
		return fmt.Errorf("degree != 1")
	}

	if op0 == op1 {
		// p.Value[1]: noise of the second component (s term)
		// p.Sk[0]: sk^1
		e.AddAutomorphismNoise(op1)
		utils.RotateSliceInPlace(op1.Value[0], k)

	} else {

		e.SetToAutomorphismNoise(op1)

		m0 := op0.Value[0]
		m1 := op1.Value[0]

		for ix, iy := 0, k; iy < len(m0); ix, iy = ix+1, iy+1 {
			m1[ix].Add(m1[ix], m0[iy])
		}

		for ix, iy := len(m0)-k, 0; iy < k; ix, iy = ix+1, iy+1 {
			m1[ix].Add(m1[ix], m0[iy])
		}

		op1.Scale = op0.Scale
		op1.Level = op0.Level
	}

	return
}

func (e Estimator) ConjugateNew(op0 *Element) (op1 *Element, err error) {
	op1 = op0.CopyNew()
	return op1, e.Conjugate(op1, op1)
}

func (e Estimator) Conjugate(op0, op1 *Element) (err error) {

	if op0.Degree != 1 {
		return fmt.Errorf("degree != 1")
	}

	if op0 == op1 {

		// p.Value[1]: noise of the second component (s term)
		// p.Sk[0]: sk^1
		e.AddAutomorphismNoise(op1)

		m1 := op1.Value[0]
		for i := range m1 {
			m1[i][1].Neg(m1[i][1])
		}

	} else {

		for i := 0; i < 2; i++ {
			m0 := op0.Value[i]
			m1 := op1.Value[i]
			for j := range m0 {
				m1[j][0].Set(m0[j][0])
				m1[j][1].Set(m0[j][1])
			}
		}

		op1.Degree = op0.Degree
		op1.Scale = op0.Scale
		op1.Level = op0.Level

		e.AddAutomorphismNoise(op1)

		m1 := op1.Value[0]
		for i := range m1 {
			m1[i][1].Neg(m1[i][1])
		}
	}

	return
}

func (e Estimator) RelinearizeNew(op0 *Element) (op1 *Element, err error) {
	op1 = op0.CopyNew()
	return op1, e.Relinearize(op1, op1)
}

func (e Estimator) Relinearize(op0, op1 *Element) (err error) {
	panic("implement Relinearize")
}

// RotateHoisted applies a rotation without ModDown.
// Returned element is scaled by P.
func (e Estimator) RotateHoistedNew(op0 *Element, k int) (value []*bignum.Complex, err error) {

	if op0.Degree != 1 {
		return nil, fmt.Errorf("degree != 1")
	}

	// Scales first term by P
	value = make([]*bignum.Complex, len(op0.Value[0]))

	pV := op0.Value[0]
	P := e.P

	for i := range value {
		value[i] = bignum.NewComplex()
		value[i][0].Mul(pV[i][0], P)
		value[i][1].Mul(pV[i][1], P)
	}

	// p.Value[1]: noise of the second component (s term)
	// p.Sk[0]: sk^1
	// Added noise is scaled by P
	e0 := e.KeySwitchingNoiseRaw(op0.Level, op0.Value[1], e.Sk[0])
	for i := range value {
		value[i].Add(value[i], e0[i])
	}

	utils.RotateSliceInPlace(value, k)

	return value, nil
}

// ModDown divides by P and adds rounding noise.
func (e Estimator) ModDown(op0, op1 *Element) {
	e.DivideAndAddRoundingNoise(op0, e.P, op1)
}

func (e Estimator) Rescale(op0, op1 *Element) (err error) {
	panic("implement Rescale")
}

func (e Estimator) DivideAndAddRoundingNoise(op0 *Element, P *big.Float, op1 *Element) {
	panic("implement DivideAndAddRoundingNoise")
}
