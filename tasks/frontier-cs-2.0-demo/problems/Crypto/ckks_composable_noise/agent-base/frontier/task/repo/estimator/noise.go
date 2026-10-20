package estimator

import (
	"math"
	"math/big"
	"math/rand"
	"time"

	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// Noise samples noise in R[X]/(X^N+1) according to f(), scales it by 2^-logScale and
// returns it in C^N/2.
func (e Estimator) Noise(f func() *big.Float) (noise []*bignum.Complex) {
	noise = make([]*bignum.Complex, e.MaxSlots())
	for i := range noise {
		noise[i] = &bignum.Complex{f(), f()}
	}

	// R[X]/(X^N+1) -> C^N/2
	if err := e.Encoder.FFT(noise, e.LogMaxSlots()); err != nil {
		panic(err)
	}

	return
}

// NoiseRingToCanonical samples a noisy vector with standard deviation
// sigma * sqrt(N/2), which emulates the sampling in the ring followed
// by the encoding of the noisy vector with the canonical embeding.
func (e Estimator) NoiseRingToCanonical(sigma float64) (noise []*bignum.Complex) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// R[X]/(X^N+1) -> C^N/2 increases variance by sqrt(N/2)
	sigma *= math.Sqrt(float64(e.N() / 2))

	f := func() *big.Float {

		s := r.NormFloat64() * sigma

		if r.Int()&1 == 0 {
			s *= -1
		}

		return NewFloat(s)
	}

	noise = make([]*bignum.Complex, e.MaxSlots())
	for i := range noise {
		noise[i] = &bignum.Complex{f(), f()}
	}

	return
}

func (e Estimator) AddNoiseRingToCanonical(sigma float64, noise []*bignum.Complex) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// R[X]/(X^N+1) -> C^N/2 increases variance by sqrt(N/2)
	sigma *= math.Sqrt(float64(e.N() / 2))

	f := func() *big.Float {

		s := r.NormFloat64() * sigma

		if r.Int()&1 == 0 {
			s *= -1
		}

		return NewFloat(s)
	}

	for i := range noise {
		noise[i].Add(noise[i], &bignum.Complex{f(), f()})
	}
}

// RoundingNoise samples a rounding error in the ring and decode it into the canonical embeding
// Standard deviation: sqrt(1/12)
func (e Estimator) RoundingNoise() (noise []*bignum.Complex) {

	if e.Heuristic {
		return e.NoiseRingToCanonical(math.Sqrt(1 / 12.0))
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return e.Noise(func() *big.Float { return NewFloat(r.Float64() - 0.5) })
}

func (e Estimator) NormalNoise(sigma float64) (noise []*bignum.Complex) {

	if e.Heuristic {
		return e.NoiseRingToCanonical(sigma)
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	f := func() *big.Float {

		s := r.NormFloat64() * sigma

		if r.Int()&1 == 0 {
			s *= -1
		}

		return NewFloat(s)
	}

	return e.Noise(f)
}

func (e Estimator) KeySwitchingNoise(levelQ int, eCt, sk []*bignum.Complex) (e0, e1 []*bignum.Complex) {
	panic("implement KeySwitchingNoise")
}

func (e Estimator) KeySwitchingNoiseRaw(levelQ int, eCt, sk []*bignum.Complex) (noise []*bignum.Complex) {
	panic("implement KeySwitchingNoiseRaw")
}
