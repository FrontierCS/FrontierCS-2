"""detok_latent_denoising: reference-relative continuous score (PSNR, LPIPS, time)."""

import deployment_test_guard  # noqa: F401  # isort: skip

import itertools
import math
import unittest

from test_detok_warmup import load_evaluator

EV = load_evaluator()
PO, PE, LO, LE, T = EV.PSNR_O, EV.PSNR_E, EV.LPIPS_O, EV.LPIPS_E, EV.TIME
REF = {PO: 24.46, PE: 24.44, LO: 0.138, LE: 0.138, T: 43.0}
SPEED_2X = 15 * math.tanh(2.0)


def cand(d_o=0.0, d_e=0.0, lo=1.0, le=1.0, ratio=1.0, ref=REF):
    """PSNR deltas (dB), LPIPS ratios L_cand / L_ref, speed ratio T_ref / T_cand."""
    return {PO: ref[PO] + d_o, PE: ref[PE] + d_e, LO: ref[LO] * lo, LE: ref[LE] * le, T: ref[T] / ratio}


def q_psnr(d):
    return 8.75 * math.tanh(d / 1.25)


def q_lpips(r):
    return 8.75 * math.tanh(math.log2(1 / r) / 0.25)


class DetokScoring(unittest.TestCase):
    def test_metric_set(self):
        self.assertEqual(EV.SPEC_METRICS, (PO, PE, LO, LE, T))
        self.assertEqual([EV.HIGHER_IS_BETTER[k] for k in EV.SPEC_METRICS], [True, True, False, False, False])

    def test_reference_scores_fifty(self):
        self.assertEqual(EV.workload_score(dict(REF), REF), 50.0)
        self.assertEqual(EV.continuous_score([dict(REF)] * 3, [REF] * 3), 50.0)
        self.assertEqual(EV.REFERENCE_SCORE, 50.0)

    def test_expected_values(self):
        self.assertAlmostEqual(EV.workload_score(cand(1.0, 1.0), REF), 50 + 2 * q_psnr(1.0))
        self.assertAlmostEqual(EV.workload_score(cand(lo=0.8, le=0.8), REF), 50 + 2 * q_lpips(0.8))
        self.assertAlmostEqual(EV.workload_score(cand(ratio=2.0), REF), 50 + SPEED_2X)
        self.assertAlmostEqual(EV.workload_score(cand(ratio=0.5), REF), 50 - SPEED_2X)

    def test_psnr_gate(self):
        # Full speed credit down to -0.02 dB, half at -0.07 dB, none at -0.12 dB.
        self.assertAlmostEqual(EV.workload_score(cand(-0.02, 0.5, ratio=2.0), REF),
                               50 + q_psnr(-0.02) + q_psnr(0.5) + SPEED_2X)
        self.assertAlmostEqual(EV.workload_score(cand(-0.07, 0.5, ratio=2.0), REF),
                               50 + q_psnr(-0.07) + q_psnr(0.5) + SPEED_2X / 2)
        self.assertAlmostEqual(EV.workload_score(cand(0.5, -0.12, ratio=2.0), REF), 50 + q_psnr(0.5) + q_psnr(-0.12))

    def test_lpips_gate(self):
        # Full speed credit while log2(L_c / L_r) <= 0.004, none from 0.024.
        full, half, none = 2 ** 0.004, 2 ** 0.014, 2 ** 0.024
        self.assertAlmostEqual(EV.workload_score(cand(lo=full, ratio=2.0), REF), 50 + q_lpips(full) + SPEED_2X)
        self.assertAlmostEqual(EV.workload_score(cand(lo=half, ratio=2.0), REF), 50 + q_lpips(half) + SPEED_2X / 2)
        self.assertAlmostEqual(EV.workload_score(cand(le=none, ratio=2.0), REF), 50 + q_lpips(none))

    def test_slowdown_is_never_gated(self):
        self.assertAlmostEqual(EV.workload_score(cand(-0.5, -0.5, 1.1, 1.1, 0.5), REF),
                               50 + 2 * q_psnr(-0.5) + 2 * q_lpips(1.1) - SPEED_2X)

    def test_bounded(self):
        for d, r, t in itertools.product((-50.0, 50.0), (1e-9, 1e9), (1e-6, 1e6)):
            s = EV.workload_score(cand(d, d, r, r, t), REF)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 100.0)
        self.assertTrue(0.0 <= EV.workload_score(cand(lo=0.0, le=0.0), REF) <= 100.0)

    def test_monotone(self):
        grid = {
            "d_o": [-3, -0.3, -0.12, -0.07, -0.02, 0, 0.02, 0.1, 0.5, 2],
            "d_e": [-3, -0.3, -0.12, -0.07, -0.02, 0, 0.02, 0.1, 0.5, 2],
            "lo": [2.0, 1.1, 1.02, 1.01, 1.003, 1.0, 0.99, 0.9, 0.5],  # falling LPIPS = better
            "le": [2.0, 1.1, 1.02, 1.01, 1.003, 1.0, 0.99, 0.9, 0.5],
            "ratio": [0.25, 0.9, 1.0, 1.02, 1.5, 4],
        }
        for axis, values in grid.items():
            others = [k for k in grid if k != axis]
            for point in itertools.product(*(grid[k][::3] for k in others)):
                fixed = dict(zip(others, point))
                row = [EV.workload_score(cand(**{**fixed, axis: v}), REF) for v in values]
                self.assertEqual(row, sorted(row), (axis, fixed))

    def test_equal_code_noise_bound(self):
        # Per-workload timing within +/-2 %, PSNR within +/-0.02 dB, LPIPS within +/-0.2 %.
        for d_o, d_e, lo, le, r in itertools.product((-0.02, 0.02), (-0.02, 0.02), (1 / 1.002, 1.002),
                                                     (1 / 1.002, 1.002), (1 / 1.02, 1.02)):
            self.assertLessEqual(abs(EV.workload_score(cand(d_o, d_e, lo, le, r), REF) - 50.0), 2.0)

    def test_quality_loss_not_bought_by_speed(self):
        for r in (1.5, 2.0, 10.0, 1e6):
            self.assertLess(EV.workload_score(cand(-0.12, -0.12, ratio=r), REF), 50.0)
            self.assertLess(EV.workload_score(cand(lo=1.02, le=1.02, ratio=r), REF), 50.0)

    def test_smoothing_trade_is_not_rewarded(self):
        # Measured eval-time bilateral filter: about +0.15 dB PSNR and +3 % LPIPS on both legs.
        self.assertLess(EV.workload_score(cand(0.15, 0.15, 1.03, 1.03), REF), 50.0)

    def test_rejects_invalid_measurements(self):
        for bad in (cand(ratio=math.inf), {**REF, PO: math.nan}, {**REF, T: 0.0}, {**REF, LO: -0.1},
                    {**REF, LE: math.inf}):
            with self.assertRaises(RuntimeError):
                EV.workload_score(bad, REF)
        with self.assertRaises(RuntimeError):
            EV.continuous_score([dict(REF)] * 2, [REF] * 3)


if __name__ == "__main__":
    unittest.main()
