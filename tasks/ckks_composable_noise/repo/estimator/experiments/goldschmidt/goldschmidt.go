package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/ckks-noise-estimator"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/inverse"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/minimax"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

func main() {

	LogN := 12
	LogScale := 55

	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,
		LogQ:            []int{60, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: LogScale,
		Xs:              ring.Ternary{H: 192},
	})

	if err != nil {
		panic(err)
	}

	ecd := ckks.NewEncoder(params)

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	dec := ckks.NewDecryptor(params, sk)

	evk := rlwe.NewMemEvaluationKeySet(kgen.GenRelinearizationKeyNew(sk))

	eval := ckks.NewEvaluator(params, evk)

	minEvl := minimax.NewEvaluator(params, eval, nil)
	inverseEval := inverse.NewEvaluator(params, minEvl)

	est := estimator.NewEstimator(params)

	statsHave := estimator.NewStats()
	statsWant := estimator.NewStats()

	log2min := -4.0

	// 2^{-(prec - LogN + 1)}
	prec := float64(params.N()/2) / params.DefaultScale().Float64()

	// Estimates the number of iterations required to achieve the desired precision, given the interval [min, 2-min]
	start := 1 - math.Exp2(log2min)
	var iters = 1
	for start >= prec {
		start *= start // Doubles the bit-precision at each iteration
		iters++
	}

	// Minimum of 3 iterations
	// This minimum is set in the case where log2min is close to 0.
	iters = max(iters, 3)

	fmt.Println(iters)

	for i := 0; i < 1; i++ {

		fmt.Println(i)

		values, el, _, ct := est.NewTestVector(ecd, pk, 0.1, 1.9)

		GoldschmidtDivision(values, iters)

		if el, err = est.GoldschmidtDivisionNew(el, log2min); err != nil {
			panic(err)
		}

		if ct, err = inverseEval.GoldschmidtDivisionNew(ct, log2min); err != nil {
			panic(err)
		}

		pWant := ckks.GetPrecisionStats(params, ecd, dec, values, est.Decrypt(el), 0, false)
		pHave := ckks.GetPrecisionStats(params, ecd, dec, values, ct, 0, false)

		statsWant.Add(pWant)
		statsHave.Add(pHave)
	}

	statsWant.Finalize()
	statsHave.Finalize()

	fmt.Println(statsWant.String())
	fmt.Println(statsHave.String())

	fmt.Println(estimator.ToLaTeXTable(LogN, LogScale, statsWant, statsHave))
}

func GoldschmidtDivision(values []*bignum.Complex, iters int) {

	a := bignum.NewComplex()
	b := bignum.NewComplex()
	tmp := bignum.NewComplex()

	prec := values[0].Prec()

	one := bignum.NewFloat(1, prec)
	two := bignum.NewFloat(2, prec)

	mul := bignum.NewComplexMultiplier().Mul

	for i := range values {

		a[0].Neg(values[i][0])
		a[1].Neg(values[i][1])
		b[0].Add(a[0], one)
		b[1].Set(a[1])
		a[0].Add(a[0], two)

		for j := 1; j < iters; j++ {
			mul(b, b, b)
			mul(a, b, tmp)
			a.Add(a, tmp)
		}

		values[i].Set(a)
	}
}
