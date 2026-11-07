package estimator

import (
	"fmt"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

type Element struct {
	Degree int
	Level  int
	Scale  rlwe.Scale
	Value  [3][]*bignum.Complex //(m + e0, e1, e2)
}

func (p Element) CopyNew() *Element {

	Value := [3][]*bignum.Complex{}

	for i := range Value {

		v0 := p.Value[i]
		v1 := make([]*bignum.Complex, len(v0))

		for j := range v0 {
			v1[j] = bignum.NewComplex()
			v1[j].Set(v0[j])
		}

		Value[i] = v1
	}

	return &Element{
		Degree: p.Degree,
		Level:  p.Level,
		Scale:  p.Scale,
		Value:  Value,
	}
}

func (e Estimator) NewElement(v interface{}, Degree, Level int, scale rlwe.Scale) *Element {

	e0 := make([]*bignum.Complex, e.MaxSlots())
	e1 := make([]*bignum.Complex, e.MaxSlots())
	e2 := make([]*bignum.Complex, e.MaxSlots())

	var slots int

	switch v := v.(type) {
	case []*bignum.Complex:
		if len(v) > e.MaxSlots() {
			panic("len(v) > p.MaxSlots()")
		}

		slots = len(v)

		for i := range v {
			e0[i] = bignum.ToComplex(v[i], prec)
		}

	case []complex128:

		if len(v) > e.MaxSlots() {
			panic("len(v) > p.MaxSlots()")
		}

		slots = len(v)

		for i := range v {
			e0[i] = bignum.ToComplex(v[i], prec)
		}

	case []*big.Float:

		if len(v) > e.MaxSlots() {
			panic("len(v) > p.MaxSlots()")
		}

		slots = len(v)

		for i := range v {
			e0[i] = bignum.ToComplex(v[i], prec)
		}

	case []float64:

		if len(v) > e.MaxSlots() {
			panic("len(v) > p.MaxSlots()")
		}

		slots = len(v)

		for i := range v {
			e0[i] = bignum.ToComplex(v[i], prec)
		}

	case nil:
		slots = 0
	default:
		panic(fmt.Errorf("invalid v.(type): must be []*bignum.Complex, []complex128, []*big.Float or []float64"))
	}

	for i := 0; i < slots; i++ {
		e0[i][0].Mul(e0[i][0], &scale.Value)
		e0[i][1].Mul(e0[i][1], &scale.Value)
	}

	for i := slots; i < e.MaxSlots(); i++ {
		e0[i] = bignum.ToComplex(0, prec)
	}

	for i := range e1 {
		e1[i] = bignum.ToComplex(0, prec)
		e2[i] = bignum.ToComplex(0, prec)
	}

	return &Element{
		Degree: Degree,
		Level:  Level,
		Scale:  scale,
		Value:  [3][]*bignum.Complex{e0, e1, e2},
	}
}
