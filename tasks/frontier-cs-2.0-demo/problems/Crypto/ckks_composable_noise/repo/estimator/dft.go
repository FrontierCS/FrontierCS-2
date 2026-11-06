package estimator

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

type DFTMatrix struct {
	dft.MatrixLiteral
	Value []lintrans.Diagonals[*bignum.Complex]
}

func (d *DFTMatrix) GenMatrices(LogN int, prec uint) {
	d.Value = d.MatrixLiteral.GenMatrices(LogN, prec)
}

func (e Estimator) DFTNew(elIn *Element, mat DFTMatrix) (elOut *Element, err error) {
	elOut = e.NewElement(nil, 1, elIn.Level, elIn.Scale)
	return elOut, e.DFT(elIn, mat, elOut)
}

func (e Estimator) DFT(elIn *Element, mat DFTMatrix, elOut *Element) (err error) {

	for i := range mat.Value {

		if i == 0 {

			scale := e.Parameters.GetOptimalScalingFactor(elIn.Scale, e.DefaultScale(), elIn.Level)

			if err = e.EvaluateLinearTransformation(elIn, LinearTransformation{
				Scale:                    scale,
				LogSlots:                 mat.LogSlots,
				LogBabyStepGianStepRatio: mat.LogBSGSRatio,
				Value:                    mat.Value[i],
			}, elOut); err != nil {
				return fmt.Errorf("e.EvaluateLinearTransformation: %w", err)
			}

		} else {

			scale := e.Parameters.GetOptimalScalingFactor(elOut.Scale, e.DefaultScale(), elOut.Level)

			if err = e.EvaluateLinearTransformation(elOut, LinearTransformation{
				Scale:                    scale,
				LogSlots:                 mat.LogSlots,
				LogBabyStepGianStepRatio: mat.LogBSGSRatio,
				Value:                    mat.Value[i],
			}, elOut); err != nil {
				return fmt.Errorf("e.EvaluateLinearTransformation: %w", err)
			}
		}

		if err = e.Rescale(elOut, elOut); err != nil {
			return fmt.Errorf("e.Rescale: %w", err)
		}
	}

	return
}

func (e Estimator) SlotsToCoeffsNew(elReal, elImag *Element, mat DFTMatrix) (el *Element, err error) {
	el = e.NewElement(nil, 1, elReal.Level, elReal.Scale)
	return el, e.SlotsToCoeffs(elReal, elImag, mat, el)
}

func (e Estimator) SlotsToCoeffs(elReal, elImag *Element, mat DFTMatrix, el *Element) (err error) {

	if elImag != nil {
		if el != elReal {
			if err = e.Mul(elImag, 1i, el); err != nil {
				return fmt.Errorf("e.Mul: %w", err)
			}

			if err = e.Add(el, elReal, el); err != nil {
				return fmt.Errorf("e.Add: %w", err)
			}

		} else {
			if err = e.MulThenAdd(elImag, 1i, elReal); err != nil {
				return fmt.Errorf("e.Mul: %w", err)
			}

			el = elReal
		}
	}

	return e.DFT(el, mat, el)
}

func (e Estimator) CoeffsToSlotsNew(el *Element, mat DFTMatrix) (elReal, elImag *Element, err error) {
	elReal = e.NewElement(nil, 1, el.Level, el.Scale)

	if (mat.Format == dft.RepackImagAsReal || mat.Format == dft.SplitRealAndImag) &&
		!(mat.Format == dft.RepackImagAsReal && mat.LogSlots < e.LogMaxSlots()) {
		elImag = e.NewElement(nil, 1, el.Level, el.Scale)
	}

	return elReal, elImag, e.CoeffsToSlots(el, mat, elReal, elImag)

}

func (e Estimator) CoeffsToSlots(el *Element, mat DFTMatrix, elReal, elImag *Element) (err error) {

	if mat.Format == dft.RepackImagAsReal || mat.Format == dft.SplitRealAndImag {

		var zV *Element
		if zV, err = e.DFTNew(el, mat); err != nil {
			return fmt.Errorf("e.DFTNew: %w", err)
		}

		if err = e.Conjugate(zV, elReal); err != nil {
			return fmt.Errorf("e.Conjugate: %w", err)
		}

		if elImag == nil {
			elImag = e.NewElement(nil, 1, elReal.Level, el.Scale)
		}

		if err = e.Sub(zV, elReal, elImag); err != nil {
			return fmt.Errorf("e.Sub: %w", err)
		}

		if err = e.Mul(elImag, -1i, elImag); err != nil {
			return fmt.Errorf("e.Mul: %w", err)
		}

		if err = e.Add(elReal, zV, elReal); err != nil {
			return fmt.Errorf("e.Add: %w", err)
		}

		// If repacking, then ct0 and ct1 right n/2 slots are zero.
		if mat.Format == dft.RepackImagAsReal && mat.LogSlots < e.LogMaxSlots() {
			if err = e.Rotate(elImag, 1<<mat.LogSlots, elImag); err != nil {
				return fmt.Errorf("e.Rotate: %w", err)
			}

			if err = e.Add(elReal, elImag, elReal); err != nil {
				return fmt.Errorf("e.Add: %w", err)
			}

			return
		}

		return
	}

	return e.DFT(el, mat, elReal)
}
