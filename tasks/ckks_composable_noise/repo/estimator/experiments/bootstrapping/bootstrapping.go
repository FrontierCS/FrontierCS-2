package main

import (
	"fmt"

	"github.com/tuneinsight/ckks-noise-estimator"
	bootEst "github.com/tuneinsight/ckks-noise-estimator/bootstrapping"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils"
)

func main() {
	LogN := 12
	LogScale := 45
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,
		LogQ:            []int{55, 45},
		LogP:            []int{61, 61, 61},
		LogDefaultScale: LogScale,
		//Xs:              ring.Ternary{H: 192},
	})

	if err != nil {
		panic(err)
	}

	btpParametersLit := bootstrapping.ParametersLiteral{}
	btpParametersLit.LogN = utils.Pointy(params.LogN())
	// btpParametersLit.LogSlots = utils.Pointy(params.LogMaxSlots())
	btpParametersLit.Xs = params.Xs()
	// btpParametersLit.CoeffsToSlotsFactorizationDepthAndLogScales = [][]int{{58}, {58}, {58}, {58}}
	btpParametersLit.LogP = []int{61, 61, 61, 61}

	btpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParametersLit)
	if err != nil {
		panic(err)
	}

	fmt.Println(btpParams.BootstrappingParameters.LogQ(), btpParams.BootstrappingParameters.LogP())
	fmt.Println(btpParams.BootstrappingParameters.MaxLevel())

	kgen := rlwe.NewKeyGenerator(params)

	sk, pk := kgen.GenKeyPairNew()

	ecd := ckks.NewEncoder(params)
	dec := rlwe.NewDecryptor(params, sk)

	evk, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		panic(err)
	}

	var eval *bootstrapping.Evaluator
	if eval, err = bootstrapping.NewEvaluator(btpParams, evk); err != nil {
		panic(err)
	}

	evalEst := bootEst.NewEvaluator(btpParams)

	estParamsResidual := evalEst.ResidualParameters

	statsHave := estimator.NewStats()
	statsWant := estimator.NewStats()

	for i := 0; i < 1; i++ {

		fmt.Println(i)

		values, el, _, ct := estParamsResidual.NewTestVector(ecd, pk, -1-1i, 1+1i)

		// Encrypted
		if ct, err = eval.Evaluate(ct); err != nil {
			panic(err)
		}

		// Simulated
		if el, err = evalEst.Bootstrap(el); err != nil {
			panic(err)
		}

		valuesWant := evalEst.ResidualParameters.Decrypt(el)
		fmt.Println(valuesWant[0])
		fmt.Println(values[0])

		pWant := ckks.GetPrecisionStats(params, ecd, dec, values, valuesWant, 0, false)
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
