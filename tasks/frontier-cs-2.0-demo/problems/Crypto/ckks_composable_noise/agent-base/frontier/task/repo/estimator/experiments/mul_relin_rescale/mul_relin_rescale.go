package main

import (
	"fmt"

	"github.com/tuneinsight/ckks-noise-estimator"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

func main() {

	LogN := 12
	LogScale := 45

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,
		LogQ:            []int{55, 45},
		LogP:            []int{60},
		LogDefaultScale: LogScale,
	})

	if err != nil {
		panic(err)
	}

	ecd := ckks.NewEncoder(params)

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	dec := ckks.NewDecryptor(params, sk)

	rlk := kgen.GenRelinearizationKeyNew(sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk)

	eval := ckks.NewEvaluator(params, evk)

	est := estimator.NewEstimator(params)

	statsHave := estimator.NewStats()
	statsWant := estimator.NewStats()

	mul := bignum.NewComplexMultiplier().Mul

	for i := 0; i < 1; i++ {

		fmt.Println(i)

		values0, el0, _, ct0 := est.NewTestVector(ecd, pk, -1-1i, 1+1i)
		values1, el1, _, ct1 := est.NewTestVector(ecd, pk, -1-1i, 1+1i)

		for j := range values0 {
			mul(values0[j], values1[j], values0[j])
		}

		if err := eval.MulRelin(ct0, ct1, ct0); err != nil {
			panic(err)
		}

		if err := eval.Rescale(ct0, ct0); err != nil {
			panic(err)
		}

		if err := est.MulRelin(el0, el1, el0); err != nil {
			panic(err)
		}

		if err := est.Rescale(el0, el0); err != nil {
			panic(err)
		}

		pWant := ckks.GetPrecisionStats(params, ecd, dec, values0, est.Decrypt(el0), 0, false)
		pHave := ckks.GetPrecisionStats(params, ecd, dec, values0, ct0, 0, false)

		statsWant.Add(pWant)
		statsHave.Add(pHave)
	}

	statsWant.Finalize()
	statsHave.Finalize()

	fmt.Println(statsWant.String())
	fmt.Println(statsHave.String())

	fmt.Println(estimator.ToLaTeXTable(LogN, LogScale, statsWant, statsHave))
}
