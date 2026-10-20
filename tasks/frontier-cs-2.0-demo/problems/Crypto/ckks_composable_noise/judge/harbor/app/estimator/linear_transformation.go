package estimator

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/lintrans"
	cl "github.com/tuneinsight/lattigo/v6/circuits/common/lintrans"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

type LinearTransformation struct {
	LogSlots                 int
	LogBabyStepGianStepRatio int
	Scale                    rlwe.Scale
	Value                    lintrans.Diagonals[*bignum.Complex]
}

// BSGSIndex returns the BSGSIndex of the target linear transformation.
func (lt LinearTransformation) BSGSIndex() (index map[int][]int, n1, n2 []int) {
	cols := 1 << lt.LogSlots
	N1 := cl.FindBestBSGSRatio(lt.Value.DiagonalsIndexList(), cols, lt.LogBabyStepGianStepRatio)
	return cl.BSGSIndex(utils.GetKeys(lt.Value), cols, N1)
}

func (e Estimator) EvaluateLinearTransformationNew(elIn *Element, lt LinearTransformation) (elOut *Element, err error) {
	elOut = e.NewElement(nil, 1, elIn.Level, elIn.Scale)
	return elOut, e.EvaluateLinearTransformation(elIn, lt, elOut)
}

func (e Estimator) EvaluateLinearTransformation(elIn *Element, lt LinearTransformation, elOut *Element) (err error) {

	if elIn.Degree != 1 {
		return fmt.Errorf("elIn.Degree != 1")
	}

	index, _, rotN2 := lt.BSGSIndex()

	ctPreRot := map[int][]*bignum.Complex{}

	for _, k := range rotN2 {
		if k != 0 {
			if _, ok := ctPreRot[k]; k != 0 && !ok {
				if ctPreRot[k], err = e.RotateHoistedNew(elIn, k); err != nil {
					return fmt.Errorf("e.RotateHoistedNew: %w", err)
				}
			}
		}
	}

	// TODO: check that no scaling is applied
	ctPreRot0, err := e.MulNew(elIn, e.P)

	if err != nil {
		return fmt.Errorf("e.MulNew: %w", err)
	}

	keys := utils.GetSortedKeys(index)

	scale := &lt.Scale.Value
	slots := 1 << lt.LogSlots

	acc := e.NewElement(nil, 1, elIn.Level, elIn.Scale.Mul(lt.Scale))
	acc.Level = elIn.Level

	elOut.Degree = 1
	elOut.Scale = elIn.Scale.Mul(lt.Scale)
	elOut.Level = elIn.Level

	tmp := make([]*bignum.Complex, slots)

	bComplex := bignum.NewComplex()
	dComplex := bignum.NewComplex()

	mul := bignum.NewComplexMultiplier().Mul

	for _, j := range keys {

		rot := -j & (slots - 1)

		var cnt int
		for _, i := range index[j] {

			utils.RotateSliceAllocFree(lt.Value[i+j], rot, tmp)

			coeffsAcc0 := acc.Value[0]
			coeffsAcc1 := acc.Value[1]

			if cnt == 0 {

				if i == 0 {

					coeffsRot0 := ctPreRot0.Value[0]
					coeffsRot1 := ctPreRot0.Value[1]

					r := e.RoundingNoise()

					for k := range tmp {
						bComplex[0].Mul(tmp[k][0], scale)
						bComplex[1].Mul(tmp[k][1], scale)
						bComplex.Add(bComplex, r[k])
						mul(coeffsRot0[k], bComplex, coeffsAcc0[k])
						mul(coeffsRot1[k], bComplex, coeffsAcc1[k])
					}

				} else {

					coeffsRot0 := ctPreRot[i]

					r := e.RoundingNoise()

					for k := range tmp {
						bComplex[0].Mul(tmp[k][0], scale)
						bComplex[1].Mul(tmp[k][1], scale)
						bComplex.Add(bComplex, r[k])
						mul(coeffsRot0[k], bComplex, coeffsAcc0[k])
					}
				}

			} else {

				if i == 0 {

					coeffsRot0 := ctPreRot0.Value[0]
					coeffsRot1 := ctPreRot0.Value[1]

					r := e.RoundingNoise()

					for k := range tmp {
						bComplex[0].Mul(tmp[k][0], scale)
						bComplex[1].Mul(tmp[k][1], scale)
						bComplex.Add(bComplex, r[k])
						mul(coeffsRot0[k], bComplex, dComplex)
						coeffsAcc0[k].Add(coeffsAcc0[k], dComplex)
						mul(coeffsRot1[k], bComplex, dComplex)
						coeffsAcc1[k].Add(coeffsAcc1[k], dComplex)
					}

				} else {

					coeffsRot0 := ctPreRot[i]

					r := e.RoundingNoise()

					for k := range tmp {
						bComplex[0].Mul(tmp[k][0], scale)
						bComplex[1].Mul(tmp[k][1], scale)
						bComplex.Add(bComplex, r[k])
						mul(coeffsRot0[k], bComplex, dComplex)
						coeffsAcc0[k].Add(coeffsAcc0[k], dComplex)
					}
				}
			}

			cnt++
		}

		if j != 0 {

			m0 := acc.Value[0]
			e0 := e.KeySwitchingNoiseRaw(elOut.Level, e.RoundingNoise(), e.Sk[0])
			for k := range m0 {
				m0[k].Add(m0[k], e0[k])
			}

			utils.RotateSliceInPlace(m0, j)
		}

		for i := 0; i < 2; i++ {
			m0, m1 := elOut.Value[i], acc.Value[i]
			for j := range m0 {
				m0[j].Add(m0[j], m1[j])
			}
		}
	}

	e.ModDown(elOut, elOut)

	return
}
