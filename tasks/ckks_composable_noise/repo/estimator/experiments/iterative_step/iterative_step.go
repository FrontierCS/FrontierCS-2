package main

import (
	"fmt"
	"math"
	"math/big"

	"github.com/tuneinsight/ckks-noise-estimator"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
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
		LogQ:            []int{60, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55},
		LogP:            []int{61, 61, 61, 61, 61},
		LogDefaultScale: LogScale,
		Xs:              ring.Ternary{H: 192},
	})

	if err != nil {
		panic(err)
	}

	fmt.Println(params.LogQ(), params.LogP())

	ecd := ckks.NewEncoder(params)

	kgen := ckks.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	dec := ckks.NewDecryptor(params, sk)

	evk := rlwe.NewMemEvaluationKeySet(kgen.GenRelinearizationKeyNew(sk))

	eval := ckks.NewEvaluator(params, evk)

	est := estimator.NewEstimator(params)

	statsHave := estimator.NewStats()
	statsWant := estimator.NewStats()

	coeffs := []*big.Float{
		bignum.NewFloat(0, 64),
		bignum.NewFloat(0, 64),
		bignum.NewFloat(3, 64),
		bignum.NewFloat(-2, 64),
	}

	poly := bignum.NewPolynomial(bignum.Monomial, coeffs, nil)

	polyEval := polynomial.NewEvaluator(params, eval)

	for i := 0; i < 1; i++ {
		fmt.Println(i)

		values, el, _, ct := NewTestVector(est, ecd, pk, -0.2, 0.2)

		for k := 0; k < 7; k++ {

			for i := range values {
				values[i] = poly.Evaluate(values[i])

				if values[i][0].Cmp(new(big.Float).SetFloat64(1e-40)) == -1 {
					values[i][0].SetFloat64(0)
				}
			}

			if el, err = est.EvaluatePolynomialNew(el, poly, el.Scale); err != nil {
				panic(err)
			}

			el.Level += 2

			if ct, err = polyEval.Evaluate(ct, poly, ct.Scale); err != nil {
				panic(err)
			}
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

func NewTestVector(est estimator.Estimator, ecd *ckks.Encoder, key rlwe.EncryptionKey, a, b float64) (values []*bignum.Complex, el *estimator.Element, pt *rlwe.Plaintext, ct *rlwe.Ciphertext) {

	params := est.Parameters

	prec := ecd.Prec()

	values = make([]*bignum.Complex, params.MaxSlots())

	source := estimator.NewTestRand()

	for i := range values {
		values[i] = &bignum.Complex{
			bignum.NewFloat(math.Round(source.Float64(0, 1))+source.Float64(a, b), prec),
			bignum.NewFloat(0, prec),
		}
	}

	values[0][0].SetFloat64(1)
	values[0][1].SetFloat64(0)

	el = est.NewElement(values, 1, est.MaxLevel(), est.DefaultScale())
	est.AddEncodingNoise(el)

	pt = ckks.NewPlaintext(params, params.MaxLevel())
	if err := ecd.Encode(values, pt); err != nil {
		panic(err)
	}

	enc := rlwe.NewEncryptor(params, key)
	if enc != nil {
		ct, _ = enc.EncryptNew(pt)
		switch key.(type) {
		case *rlwe.SecretKey:
			est.AddEncryptionNoiseSk(el)
		case *rlwe.PublicKey:
			est.AddEncryptionNoisePk(el)
		default:
			panic("INVALID ENCRYPION KEY")
		}
	}

	return
}
