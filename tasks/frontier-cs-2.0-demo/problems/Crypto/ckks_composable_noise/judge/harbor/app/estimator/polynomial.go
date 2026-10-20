package estimator

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

// PowerBasis is a struct storing powers of a ciphertext.
type PowerBasis struct {
	bignum.Basis
	Value map[int]*Element
}

// NewPowerBasis creates a new PowerBasis. It takes as input a ciphertext
// and a basistype. The struct treats the input ciphertext as a monomial X and
// can be used to generates power of this monomial X^{n} in the given BasisType.
func NewPowerBasis(ct *Element, basis bignum.Basis) (p *PowerBasis) {
	p = new(PowerBasis)
	p.Value = make(map[int]*Element)
	p.Value[1] = ct.CopyNew()
	p.Basis = basis
	return
}

// GenPower recursively computes X^{n}.
// If lazy = true, the final X^{n} will not be relinearized.
// Previous non-relinearized X^{n} that are required to compute the target X^{n} are automatically relinearized.
// Scale sets the threshold for rescaling (ciphertext won't be rescaled if the rescaling operation would make the scale go under this threshold).
func (p *PowerBasis) GenPower(n int, lazy bool, e Estimator) (err error) {

	if _, ok := p.Value[n]; !ok {
		var rescale bool
		if rescale, err = p.genPower(n, lazy, e); err != nil {
			return fmt.Errorf("p.genPower: %w", err)
		}

		if rescale {
			if err = e.Rescale(p.Value[n], p.Value[n]); err != nil {
				return fmt.Errorf("e.Rescale: %w", err)
			}
		}
	}

	return nil
}

func (p *PowerBasis) genPower(n int, lazy bool, e Estimator) (rescale bool, err error) {

	if _, ok := p.Value[n]; !ok {

		isPow2 := n&(n-1) == 0

		// Computes the index required to compute the asked ring evaluation
		var a, b, c int
		if isPow2 {
			a, b = n/2, n/2 //Necessary for optimal depth
		} else {
			// [Lee et al. 2020] : High-Precision and Low-Complexity Approximate Homomorphic Encryption by Error Variance Minimization
			// Maximize the number of odd terms of Chebyshev basis
			k := int(math.Ceil(math.Log2(float64(n)))) - 1
			a = (1 << k) - 1
			b = n + 1 - (1 << k)

			if p.Basis == bignum.Chebyshev {
				c = int(math.Abs(float64(a) - float64(b))) // Cn = 2*Ca*Cb - Cc, n = a+b and c = abs(a-b)
			}
		}

		// Recurses on the given indexes
		var rescaleA, rescaleB bool
		if rescaleA, err = p.genPower(a, lazy && !isPow2, e); err != nil {
			return false, fmt.Errorf("e.genPower: %w", err)
		}

		if rescaleB, err = p.genPower(b, lazy && !isPow2, e); err != nil {
			return false, fmt.Errorf("e.genPower: %w", err)
		}

		// Computes C[n] = C[a]*C[b]
		if lazy {

			if p.Value[a].Degree == 2 {
				if err = e.Relinearize(p.Value[a], p.Value[a]); err != nil {
					return false, fmt.Errorf("e.Relinearize: %w", err)
				}
			}

			if p.Value[b].Degree == 2 {
				if err = e.Relinearize(p.Value[b], p.Value[b]); err != nil {
					return false, fmt.Errorf("e.Relinearize: %w", err)
				}
			}

			if rescaleA {
				if err = e.Rescale(p.Value[a], p.Value[a]); err != nil {
					return false, fmt.Errorf("e.Relinearize: %w", err)
				}
			}

			if rescaleB {
				if err = e.Rescale(p.Value[b], p.Value[b]); err != nil {
					return false, fmt.Errorf("e.Relinearize: %w", err)
				}
			}

			if p.Value[n], err = e.MulNew(p.Value[a], p.Value[b]); err != nil {
				return false, fmt.Errorf("e.MulNew: %w", err)
			}

		} else {

			if rescaleA {
				if err = e.Rescale(p.Value[a], p.Value[a]); err != nil {
					return false, fmt.Errorf("e.Relinearize: %w", err)
				}
			}

			if rescaleB {
				if err = e.Rescale(p.Value[b], p.Value[b]); err != nil {
					return false, fmt.Errorf("e.Relinearize: %w", err)
				}
			}

			if p.Value[n], err = e.MulRelinNew(p.Value[a], p.Value[b]); err != nil {
				return false, fmt.Errorf("e.MulRelinNew: %w", err)
			}

		}

		if p.Basis == bignum.Chebyshev {

			// Computes C[n] = 2*C[a]*C[b]
			if err = e.Mul(p.Value[n], 2, p.Value[n]); err != nil {
				return false, fmt.Errorf("e.Mul: %w", err)
			}

			// Computes C[n] = 2*C[a]*C[b] - C[c]
			if c == 0 {
				if err = e.Add(p.Value[n], -1, p.Value[n]); err != nil {
					return false, fmt.Errorf("e.Mul: %w", err)
				}
			} else {
				// Since C[0] is not stored (but rather seen as the constant 1), only recurses on c if c!= 0
				if err = p.GenPower(c, lazy, e); err != nil {
					return false, fmt.Errorf("e.GenPower: %w", err)
				}
				if err = e.Sub(p.Value[n], p.Value[c], p.Value[n]); err != nil {
					return false, fmt.Errorf("e.GenPower: %w", err)
				}
			}
		}

		return true, nil
	}

	return false, nil
}
