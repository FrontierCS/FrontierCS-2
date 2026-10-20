package estimator

import (
	"fmt"
	"github.com/tuneinsight/lattigo/v6/utils"
	"math"
)

func (e Estimator) GoldschmidtDivisionNew(el *Element, log2min float64) (a *Element, err error) {

	// 2^{-(prec - LogN + 1)}
	prec := float64(e.N()/2) / e.DefaultScale().Float64()

	// Estimates the number of iterations required to achieve the desired precision, given the interval [min, 2-min]
	start := 1 - math.Exp2(log2min)
	var iters = 1
	for start >= prec {
		start *= start // Doubles the bit-precision at each iteration
		iters++
	}

	iters = utils.Max(iters, 3)

	if a, err = e.MulNew(el, -1); err != nil {
		return nil, fmt.Errorf("e.MulNew: %w", err)
	}

	var b *Element
	if b, err = e.AddNew(a, 1); err != nil {
		return nil, fmt.Errorf("e.AddNew: %w", err)
	}

	if err = e.Add(a, 2, a); err != nil {
		return nil, fmt.Errorf("e.Add: %w", err)
	}

	tmp := e.NewElement(nil, 1, a.Level, a.Scale)

	for j := 1; j < iters; j++ {

		if err = e.MulRelin(b, b, b); err != nil {
			return nil, fmt.Errorf("e.MulRelin: %w", err)
		}

		if err = e.Rescale(b, b); err != nil {
			return nil, fmt.Errorf("e.Rescale: %w", err)
		}

		if err = e.MulRelin(a, b, tmp); err != nil {
			return nil, fmt.Errorf("e.MulRelin: %w", err)
		}

		if err = e.Rescale(tmp, tmp); err != nil {
			return nil, fmt.Errorf("e.Rescale: %w", err)
		}

		if err = e.SetScale(a, tmp.Scale); err != nil {
			return nil, fmt.Errorf("e.SetScale: %w", err)
		}

		if err = e.Add(a, tmp, a); err != nil {
			return nil, fmt.Errorf("e.Add: %w", err)
		}
	}

	return
}
