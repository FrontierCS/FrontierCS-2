package estimator

import (
	"math/big"
	"math/rand"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

type Estimator struct {
	Parameters ckks.Parameters
	Encoder
	LogN      int
	Sigma     float64
	Scale     rlwe.Scale
	H         int
	Q         []big.Float
	P         *big.Float
	LevelP    int
	Sk        [][]*bignum.Complex
	Heuristic bool
}

func NewEstimator(p ckks.Parameters) (e Estimator) {

	e = Estimator{}
	e.Parameters = p
	e.LogN = p.LogN()
	e.Sigma = p.NoiseFreshSK()
	e.Scale = p.DefaultScale()
	e.Heuristic = true

	Q := p.Q()
	P := p.P()

	Qi := make([]big.Float, len(Q))
	for i := range Q {
		Qi[i] = *NewFloat(Q[i])
	}
	e.Q = Qi

	Pi := NewFloat(1)
	for i := range P {
		Pi.Mul(Pi, NewFloat(P[i]))
	}
	e.P = Pi
	e.LevelP = len(P) - 1

	e.Encoder = *NewEncoder(p.LogN(), prec)

	e.H = min(p.N(), p.XsHammingWeight())

	// Samples a secret-key
	sk := e.SampleSecretKey(e.H)

	mul := bignum.NewComplexMultiplier().Mul
	sk2 := make([]*bignum.Complex, p.MaxSlots())
	for i := range sk2 {
		sk2[i] = &bignum.Complex{NewFloat(0), NewFloat(0)}
		mul(sk[i], sk[i], sk2[i])
	}

	e.Sk = [][]*bignum.Complex{sk, sk2}

	return
}

func (e Estimator) SampleSecretKey(H int) (sk []*bignum.Complex) {

	N := e.N()

	H = min(N, H)

	skF := make([]*big.Float, N)
	for i := 0; i < H; i++ {
		if i&1 == 0 {
			skF[i] = NewFloat(1)
		} else {
			skF[i] = NewFloat(-1)
		}

	}
	for i := H; i < N; i++ {
		skF[i] = NewFloat(0)
	}

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	r.Shuffle(len(skF), func(i, j int) { skF[i], skF[j] = skF[j], skF[i] })

	sk = make([]*bignum.Complex, e.MaxSlots())
	for i := range sk {
		sk[i] = &bignum.Complex{skF[2*i], skF[2*i+1]}
	}

	// R[X]/(X^N+1) -> C^N/2
	if err := e.Encoder.FFT(sk, e.LogMaxSlots()); err != nil {
		panic(err)
	}

	return
}

func (e Estimator) N() int {
	return 1 << e.LogN
}

func (e Estimator) MaxSlots() int {
	return 1 << e.LogMaxSlots()
}

func (e Estimator) LogMaxSlots() int {
	return e.LogN - 1
}

func (e Estimator) MaxLevel() int {
	return len(e.Q) - 1
}

func (e Estimator) DefaultScale() rlwe.Scale {
	return e.Scale
}

// Decrypt decrypts the element by evaluating <(el0, el1, el2), (1, sk, sk^2)>.
func (e Estimator) Decrypt(el *Element) (values []*bignum.Complex) {

	v := el.Value[0]

	values = make([]*bignum.Complex, e.MaxSlots())

	for i := range values {
		values[i] = bignum.NewComplex().Set(v[i])
	}

	mul := bignum.NewComplexMultiplier().Mul
	tmp := bignum.NewComplex()

	for i := 1; i < el.Degree+1; i++ {
		sk := e.Sk[i-1]
		ei := el.Value[i]
		for j := range ei {
			mul(ei[j], sk[j], tmp)
			values[j].Add(values[j], tmp)
		}
	}

	scale := &el.Scale.Value

	for i := range values {
		values[i][0].Quo(values[i][0], scale)
		values[i][1].Quo(values[i][1], scale)
	}

	return
}

// AddEncodingNoise adds the encoding noise, which is
// {round(1/2), 0}.
func (e Estimator) AddEncodingNoise(el *Element) {
	noise := e.RoundingNoise()
	value := el.Value[0]
	for j := range value {
		value[j].Add(value[j], noise[j])
	}
}

// AddRoundingNoise adds the rounding noise,
// which is {round(1/2), round(1/2)}.
func (e Estimator) AddRoundingNoise(el *Element) {
	for i := 0; i < el.Degree+1; i++ {
		noise := e.RoundingNoise()
		ei := el.Value[i]
		for j := range ei {
			ei[j].Add(ei[j], noise[j])
		}
	}
}

// AddEncryptionNoiseSk adds the encryption noise
// from SK encryption, which is {sigma, 0}.
func (e Estimator) AddEncryptionNoiseSk(el *Element) {
	noise := e.NormalNoise(e.Sigma)
	value := el.Value[0]
	for i := range value {
		value[i].Add(value[i], noise[i])
	}
}

// AddEncryptionNoisePk adds the encryption noise
// from PK encryption, which is {round(1/2), round(1/2)}
// This assumes that P != 0 and that |N * e * sk| < P.
func (e Estimator) AddEncryptionNoisePk(el *Element) {
	el.Degree = max(1, el.Degree)
	e.AddRoundingNoise(el)
}

func (e Estimator) AddKeySwitchingNoise(el *Element, sk []*bignum.Complex) {
	e0, e1 := e.KeySwitchingNoise(el.Level, el.Value[1], sk)
	m0, m1 := el.Value[0], el.Value[1]
	for i := range m0 {
		m0[i].Add(m0[i], e0[i])
		m1[i].Set(e1[i])
	}
}

func (e Estimator) AddAutomorphismNoise(el *Element) {
	e0, e1 := e.KeySwitchingNoise(el.Level, el.Value[1], e.Sk[0])
	m0, m1 := el.Value[0], el.Value[1]
	for i := range m0 {
		m0[i].Add(m0[i], e0[i])
		m1[i].Set(e1[i])
	}
}

func (e Estimator) SetToAutomorphismNoise(el *Element) {
	el.Value[0], el.Value[1] = e.KeySwitchingNoise(el.Level, el.Value[1], e.Sk[0])
}

func (e Estimator) AddRelinearizationNoise(el *Element) {
	e0, e1 := e.KeySwitchingNoise(el.Level, el.Value[2], e.Sk[1])
	m0 := el.Value[0]
	m1 := el.Value[1]
	for i := range m0 {
		m0[i].Add(m0[i], e0[i])
		m1[i].Add(m1[i], e1[i])
	}
}
