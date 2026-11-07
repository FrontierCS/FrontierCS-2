package estimator

import (
	"fmt"
	"math/big"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

const (
	GaloisGen = 5
)

type Encoder struct {
	prec         uint
	bigintCoeffs []big.Int
	m            int
	rotGroup     []int
	roots        interface{}
	buffCmplx    interface{}
}

func NewEncoder(LogN int, prec uint) (ecd *Encoder) {

	m := 2 << LogN

	rotGroup := make([]int, m>>2)
	fivePows := 1
	for i := 0; i < m>>2; i++ {
		rotGroup[i] = fivePows
		fivePows *= int(GaloisGen)
		fivePows &= (m - 1)
	}

	ecd = &Encoder{
		prec:         prec,
		bigintCoeffs: make([]big.Int, m>>1),
		m:            m,
		rotGroup:     rotGroup,
	}

	if prec <= 53 {

		ecd.roots = ckks.GetRootsComplex128(ecd.m)
		ecd.buffCmplx = make([]complex128, ecd.m>>2)

	} else {

		tmp := make([]*bignum.Complex, ecd.m>>2)

		for i := 0; i < ecd.m>>2; i++ {
			tmp[i] = &bignum.Complex{bignum.NewFloat(0, prec), bignum.NewFloat(0, prec)}
		}

		ecd.roots = ckks.GetRootsBigComplex(ecd.m, prec)
		ecd.buffCmplx = tmp
	}

	return
}

func (ecd Encoder) IFFT(values interface{}, logN int) (err error) {
	switch values := values.(type) {
	case []complex128:
		switch roots := ecd.roots.(type) {
		case []complex128:
			if logN < 4 {
				ckks.SpecialIFFTDouble(values, 1<<logN, ecd.m, ecd.rotGroup, ecd.roots.([]complex128))
			} else {
				ckks.SpecialiFFTDoubleUnrolled8(values, 1<<logN, ecd.m, ecd.rotGroup, ecd.roots.([]complex128))
			}
		default:
			return fmt.Errorf("cannot IFFT: values.(type)=%T doesn't roots.(type) = %T", values, roots)
		}

	case []*bignum.Complex:
		switch roots := ecd.roots.(type) {
		case []*bignum.Complex:
			ckks.SpecialIFFTArbitrary(values, 1<<logN, ecd.m, ecd.rotGroup, ecd.roots.([]*bignum.Complex))
		default:
			return fmt.Errorf("cannot IFFT: values.(type)=%T doesn't roots.(type) = %T", values, roots)
		}
	default:
		return fmt.Errorf("cannot IFFT: invalid values.(type), accepted types are []complex128 and []*bignum.Complex but is %T", values)
	}
	return

}

func (ecd Encoder) FFT(values interface{}, logN int) (err error) {
	switch values := values.(type) {
	case []complex128:
		switch roots := ecd.roots.(type) {
		case []complex128:
			if logN < 4 {
				ckks.SpecialFFTDouble(values, 1<<logN, ecd.m, ecd.rotGroup, roots)
			} else {
				ckks.SpecialFFTDoubleUL8(values, 1<<logN, ecd.m, ecd.rotGroup, roots)
			}
		default:
			return fmt.Errorf("cannot IFFT: values.(type)=%T doesn't roots.(type) = %T", values, roots)
		}

	case []*bignum.Complex:
		switch roots := ecd.roots.(type) {
		case []*bignum.Complex:
			ckks.SpecialFFTArbitrary(values, 1<<logN, ecd.m, ecd.rotGroup, roots)
		default:
			return fmt.Errorf("cannot IFFT: values.(type)=%T doesn't roots.(type) = %T", values, roots)
		}

	default:
		return fmt.Errorf("cannot IFFT: invalid values.(type), accepted types are []complex128 and []*bignum.Complex but is %T", values)
	}
	return
}
