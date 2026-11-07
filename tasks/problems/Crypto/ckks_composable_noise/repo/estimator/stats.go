package estimator

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type Stats struct {
	ckks.PrecisionStats
	N float64
}

func NewStats() Stats {
	s := Stats{}
	s.MINLog2Prec.Real = 1e10
	s.MINLog2Prec.Imag = 1e10
	s.MINLog2Prec.L2 = 1e10
	s.MINLog2Err.Real = 1e10
	s.MINLog2Err.Imag = 1e10
	s.MINLog2Err.L2 = 1e10
	return s
}

func ToLaTeXTable(LogN, LogScale int, predicted, actual Stats) string {
	return fmt.Sprintf(`
\begin{table}[]
    \centering
    \begin{tabular}{|c||c|c|c||c|c|c|}
    \hline
        & \multicolumn{3}{c||}{Predicted} & \multicolumn{3}{c|}{Actual}  \\
        \hline
        $\log_{2}$ & real & imag & l2 & real & imag & l2\\
        \hline
        Min & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f \\
        AVG & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f \\
        STD & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f & %5.2f \\
        \hline
    \end{tabular}
    \caption{$N=2^{%d}$ and $\Delta = 2^{%d}$}
    \label{tab:my_label}
\end{table}
`,
		predicted.MINLog2Prec.Real, predicted.MINLog2Prec.Imag, predicted.MINLog2Prec.L2,
		actual.MINLog2Prec.Real, actual.MINLog2Prec.Imag, actual.MINLog2Prec.L2,
		predicted.AVGLog2Prec.Real, predicted.AVGLog2Prec.Imag, predicted.AVGLog2Prec.L2,
		actual.AVGLog2Prec.Real, actual.AVGLog2Prec.Imag, actual.AVGLog2Prec.L2,
		predicted.STDLog2Prec.Real, predicted.STDLog2Prec.Imag, predicted.STDLog2Prec.L2,
		actual.STDLog2Prec.Real, actual.STDLog2Prec.Imag, actual.STDLog2Prec.L2,
		LogN, LogScale)
}

func (s *Stats) Add(a ckks.PrecisionStats) {

	s.MINLog2Prec.Real = min(s.MINLog2Prec.Real, a.MINLog2Prec.Real)
	s.MINLog2Prec.Imag = min(s.MINLog2Prec.Imag, a.MINLog2Prec.Imag)
	s.MINLog2Prec.L2 = min(s.MINLog2Prec.L2, a.MINLog2Prec.L2)

	s.MAXLog2Prec.Real = max(s.MAXLog2Prec.Real, a.MAXLog2Prec.Real)
	s.MAXLog2Prec.Imag = max(s.MAXLog2Prec.Imag, a.MAXLog2Prec.Imag)
	s.MAXLog2Prec.L2 = max(s.MAXLog2Prec.L2, a.MAXLog2Prec.L2)

	s.AVGLog2Prec.Real += a.AVGLog2Prec.Real
	s.AVGLog2Prec.Imag += a.AVGLog2Prec.Imag
	s.AVGLog2Prec.L2 += a.AVGLog2Prec.L2

	s.MEDLog2Prec.Real += a.MEDLog2Prec.Real
	s.MEDLog2Prec.Imag += a.MEDLog2Prec.Imag
	s.MEDLog2Prec.L2 += a.MEDLog2Prec.L2

	s.STDLog2Prec.Real += a.STDLog2Prec.Real * a.STDLog2Prec.Real
	s.STDLog2Prec.Imag += a.STDLog2Prec.Imag * a.STDLog2Prec.Imag
	s.STDLog2Prec.L2 += a.STDLog2Prec.L2 * a.STDLog2Prec.L2

	s.MINLog2Err.Real = min(s.MINLog2Err.Real, a.MINLog2Err.Real)
	s.MINLog2Err.Imag = min(s.MINLog2Err.Imag, a.MINLog2Err.Imag)
	s.MINLog2Err.L2 = min(s.MINLog2Err.L2, a.MINLog2Err.L2)

	s.MAXLog2Err.Real = max(s.MAXLog2Err.Real, a.MAXLog2Err.Real)
	s.MAXLog2Err.Imag = max(s.MAXLog2Err.Imag, a.MAXLog2Err.Imag)
	s.MAXLog2Err.L2 = max(s.MAXLog2Err.L2, a.MAXLog2Err.L2)

	s.AVGLog2Err.Real += a.AVGLog2Err.Real
	s.AVGLog2Err.Imag += a.AVGLog2Err.Imag
	s.AVGLog2Err.L2 += a.AVGLog2Err.L2

	s.MEDLog2Err.Real += a.MEDLog2Err.Real
	s.MEDLog2Err.Imag += a.MEDLog2Err.Imag
	s.MEDLog2Err.L2 += a.MEDLog2Err.L2

	s.STDLog2Err.Real += a.STDLog2Err.Real * a.STDLog2Err.Real
	s.STDLog2Err.Imag += a.STDLog2Err.Imag * a.STDLog2Err.Imag
	s.STDLog2Err.L2 += a.STDLog2Err.L2 * a.STDLog2Err.L2

	s.N++
}

func (s *Stats) Finalize() {

	s.AVGLog2Prec.Real /= s.N
	s.AVGLog2Prec.Imag /= s.N
	s.AVGLog2Prec.L2 /= s.N

	s.MEDLog2Prec.Real /= s.N
	s.MEDLog2Prec.Imag /= s.N
	s.MEDLog2Prec.L2 /= s.N

	s.STDLog2Prec.Real = math.Sqrt(s.STDLog2Prec.Real / s.N)
	s.STDLog2Prec.Imag = math.Sqrt(s.STDLog2Prec.Imag / s.N)
	s.STDLog2Prec.L2 = math.Sqrt(s.STDLog2Prec.L2 / s.N)

	s.AVGLog2Err.Real /= s.N
	s.AVGLog2Err.Imag /= s.N
	s.AVGLog2Err.L2 /= s.N

	s.MEDLog2Err.Real /= s.N
	s.MEDLog2Err.Imag /= s.N
	s.MEDLog2Err.L2 /= s.N

	s.STDLog2Err.Real = math.Sqrt(s.STDLog2Err.Real / s.N)
	s.STDLog2Err.Imag = math.Sqrt(s.STDLog2Err.Imag / s.N)
	s.STDLog2Err.L2 = math.Sqrt(s.STDLog2Err.L2 / s.N)
}
