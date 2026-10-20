package main

import (
	"fmt"
	"math/big"

	"github.com/tuneinsight/ckks-noise-estimator"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {

	LogN := 12
	LogScale := 45
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,
		LogQ:            []int{60, 45, 45, 45, 45},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: LogScale,
		//Xs: ring.Ternary{H:192},
	})

	if err != nil {
		panic(err)
	}

	fmt.Println(params.LogQ(), params.LogP())

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
	sk, _ := kgen.GenKeyPairNew()
	dec := ckks.NewDecryptor(params, sk)

	est := estimator.NewEstimator(params)

	statsHave := estimator.NewStats()
	statsWant := estimator.NewStats()

	DFTMatrixLiteral := dft.MatrixLiteral{
		LogSlots: params.LogMaxSlots(),
		Type:     dft.HomomorphicDecode,
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

	evk := rlwe.NewMemEvaluationKeySet(nil, kgen.GenGaloisKeysNew(append(DFTMatrixHeFloat.GaloisElements(params)), sk)...)

	eval := ckks.NewEvaluator(params, evk)
	hdftEval := dft.NewEvaluator(params, eval)

	for i := 0; i < 1; i++ {

		fmt.Println(i)

		valuesReal, elReal, _, ctReal := est.NewTestVector(ecd, sk, -1, 1)
		valuesImag, elImag, _, ctImag := est.NewTestVector(ecd, sk, -1, 1)

		ctReal, err := hdftEval.SlotsToCoeffsNew(ctReal, ctImag, DFTMatrixHeFloat)

		if err != nil {
			panic(err)
		}

		for i := range valuesReal {
			valuesReal[i][1].Set(valuesImag[i][0])
		}

		for i := range DFTMatrix.Value {
			valuesReal = DFTMatrix.Value[i].Evaluate(valuesReal, newVec, add, muladd)
		}

		if err = est.SlotsToCoeffs(elReal, elImag, DFTMatrix, elReal); err != nil {
			panic(err)
		}

		pWantReal := ckks.GetPrecisionStats(params, ecd, nil, valuesReal, est.Decrypt(elReal), 0, false)
		pHaveReal := ckks.GetPrecisionStats(params, ecd, dec, valuesReal, ctReal, 0, false)

		statsWant.Add(pWantReal)
		statsHave.Add(pHaveReal)
	}

	statsWant.Finalize()
	statsHave.Finalize()

	fmt.Println(statsWant.String())
	fmt.Println(statsHave.String())
}
