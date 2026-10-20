package main

import (
	"fmt"

	"github.com/tuneinsight/ckks-noise-estimator"

	"github.com/tuneinsight/lattigo/v6/circuits/common/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	hefloat "github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"

	"time"
)

func main() {

	LogN := 12
	LogScale := 55

	params, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{
		LogN:            LogN,
		LogQ:            []int{60, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: LogScale,
		//Xs:              ring.Ternary{H: 192},
	})

	if err != nil {
		panic(err)
	}

	ecd := hefloat.NewEncoder(params)

	fmt.Println(params.LogQ(), params.LogP())

	kgen := hefloat.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	dec := hefloat.NewDecryptor(params, sk)

	evk := rlwe.NewMemEvaluationKeySet(kgen.GenRelinearizationKeyNew(sk))

	eval := hefloat.NewEvaluator(params, evk)

	est := estimator.NewEstimator(params)

	statsHave := estimator.NewStats()
	statsWant := estimator.NewStats()

	mul := bignum.NewComplexMultiplier().Mul

	n := 12

	for i := 0; i < 1; i++ {

		fmt.Println(i)

		values, el, _, ct := est.NewTestVector(ecd, pk, -1, 1)

		for k := 0; k < n; k++ {
			for l := range values {
				mul(values[l], values[l], values[l])
				values[l].Add(values[l], values[l])
				values[l].Add(values[l], &bignum.Complex{estimator.NewFloat(-1), estimator.NewFloat(0)})
			}
		}

		pbCt := polynomial.NewPowerBasis(ct, bignum.Chebyshev)
		runTimed(func() {
			if err := pbCt.GenPower(1<<n, false, eval); err != nil {
				panic(err)
			}
		})

		ct = pbCt.Value[1<<n]

		pbEl := estimator.NewPowerBasis(el, bignum.Chebyshev)

		runTimed(func() {
			pbEl.GenPower(1<<n, false, est)
		})
		el = pbEl.Value[1<<n]

		pWant := hefloat.GetPrecisionStats(params, ecd, dec, values, est.Decrypt(el), 0, false)
		pHave := hefloat.GetPrecisionStats(params, ecd, dec, values, ct, 0, false)

		statsWant.Add(pWant)
		statsHave.Add(pHave)
	}

	statsWant.Finalize()
	statsHave.Finalize()

	fmt.Println(statsWant.String())
	fmt.Println(statsHave.String())

	fmt.Println(estimator.ToLaTeXTable(LogN, LogScale, statsWant, statsHave))
}

func runTimed(f func()) {
	now := time.Now()
	f()
	fmt.Println(time.Since(now))
}
