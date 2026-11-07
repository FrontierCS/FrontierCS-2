package main

import (
	"fmt"
	"math/big"

	"github.com/tuneinsight/ckks-noise-estimator"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

func main() {

	LogN := 12
	LogScale := 55
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,
		LogQ:            []int{55, 55, 55, 55, 55},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: LogScale,
		//Xs: ring.Ternary{H:192},
	})

	mulCmplx := bignum.NewComplexMultiplier().Mul

	var prec uint = 128

	add := func(a, b, c []*bignum.Complex) {
		for i := range c {
			if a[i] != nil && b[i] != nil {
				c[i].Add(a[i], b[i])
			}
		}
	}

	muladd := func(a, b, c []*bignum.Complex) {
		tmp := &bignum.Complex{new(big.Float), new(big.Float)}
		for i := range c {
			if a[i] != nil && b[i] != nil {
				mulCmplx(a[i], b[i], tmp)
				c[i].Add(c[i], tmp)
			}
		}
	}

	newVec := func(size int) (vec []*bignum.Complex) {
		vec = make([]*bignum.Complex, size)
		for i := range vec {
			vec[i] = &bignum.Complex{new(big.Float).SetPrec(prec), new(big.Float).SetPrec(prec)}
		}
		return
	}

	ecd := ckks.NewEncoder(params)

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	dec := ckks.NewDecryptor(params, sk)

	est := estimator.NewEstimator(params)

	statsHave := estimator.NewStats()
	statsWant := estimator.NewStats()

	DFTMatrixLiteral := dft.MatrixLiteral{
		LogSlots: params.LogMaxSlots(),
		Type:     dft.HomomorphicEncode,
		Format:   dft.RepackImagAsReal,
		LevelQ:   params.MaxLevelQ(),
		LevelP:   params.MaxLevelP(),
		Levels:   []int{1, 1, 1, 1},
	}

	DFTMatrix := estimator.DFTMatrix{
		MatrixLiteral: DFTMatrixLiteral,
	}

	DFTMatrix.GenMatrices(params.LogN(), prec)

	DFTMatrixHeFloat, err := dft.NewMatrixFromLiteral(params, DFTMatrixLiteral, ecd)

	if err != nil {
		panic(err)
	}

	evk := rlwe.NewMemEvaluationKeySet(nil, kgen.GenGaloisKeysNew(append(DFTMatrixHeFloat.GaloisElements(params), params.GaloisElementForComplexConjugation()), sk)...)

	eval := ckks.NewEvaluator(params, evk)
	hdftEval := dft.NewEvaluator(params, eval)

	// hoistingbuffer := hdftEval.NewHoistingBuffer(params.MaxLevelQ(), params.MaxLevelP())

	for i := 0; i < 1; i++ {

		fmt.Println(i)

		values, el, _, ct := est.NewTestVector(ecd, pk, -1-1i, 1+1i)

		elReal, elImag, err := est.CoeffsToSlotsNew(el, DFTMatrix)

		if err != nil {
			panic(err)
		}

		ctReal, ctImag, err := hdftEval.CoeffsToSlotsNew(ct, DFTMatrixHeFloat)

		if err != nil {
			panic(err)
		}

		for i := range DFTMatrix.Value {
			values = DFTMatrix.Value[i].Evaluate(values, newVec, add, muladd)
		}

		if elImag != nil {
			two := new(big.Float).SetInt64(2)
			for i := range values {
				values[i][0].Mul(values[i][0], two)
				values[i][1].Mul(values[i][1], two)
			}

			valuesReal := make([]*bignum.Complex, len(values))
			for i := range valuesReal {
				valuesReal[i] = &bignum.Complex{
					values[i][0],
					new(big.Float),
				}
			}

			pWantReal := ckks.GetPrecisionStats(params, ecd, nil, valuesReal, est.Decrypt(elReal), 0, false)
			pHaveReal := ckks.GetPrecisionStats(params, ecd, dec, valuesReal, ctReal, 0, false)

			statsWant.Add(pWantReal)
			statsHave.Add(pHaveReal)

			valuesImag := make([]*bignum.Complex, len(values))
			for i := range valuesImag {
				valuesImag[i] = &bignum.Complex{
					values[i][1],
					new(big.Float),
				}
			}

			pWantImag := ckks.GetPrecisionStats(params, ecd, nil, valuesImag, est.Decrypt(elImag), 0, false)
			pHaveImag := ckks.GetPrecisionStats(params, ecd, dec, valuesImag, ctImag, 0, false)
			statsWant.Add(pWantImag)
			statsHave.Add(pHaveImag)
		} else {

			pWantReal := ckks.GetPrecisionStats(params, ecd, nil, values, est.Decrypt(elReal), 0, false)
			pHaveReal := ckks.GetPrecisionStats(params, ecd, dec, values, ctReal, 0, false)

			statsWant.Add(pWantReal)
			statsHave.Add(pHaveReal)
		}
	}

	statsWant.Finalize()
	statsHave.Finalize()

	fmt.Println(statsWant.String())
	fmt.Println(statsHave.String())
}
