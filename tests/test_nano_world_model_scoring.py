"""nano_world_model: reference-relative continuous score over every clip of the suite."""

from __future__ import annotations

import deployment_test_guard  # noqa: F401 # isort: skip

import importlib.util
import itertools
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/nano_world_model"
REF = {"psnr": 36.6, "ssim": 0.9837, "lpips": 0.0198}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def cand(dp=0.0, dssim=1.0, rlpips=1.0, ref=REF):
    """PSNR delta (dB); ratios (1 - SSIM_cand) / (1 - SSIM_ref) and LPIPS_cand / LPIPS_ref."""
    return {"psnr": ref["psnr"] + dp, "ssim": 1 - (1 - ref["ssim"]) * dssim, "lpips": ref["lpips"] * rlpips}


class NanoWorldModelScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old_modules, old_path = dict(sys.modules), list(sys.path)
        sys.path.insert(0, str(TASK / "verifier"))
        sys.modules.pop("beat_spec_feedback", None)
        cls.verifier = module(TASK / "verifier/fcs2_verifier.py", "nano_scoring_test_verifier")
        sys.modules.pop("submission_patch", None)
        with patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=lambda _: {})}):
            cls.ev = module(TASK / "judge/evaluator.py", "nano_scoring_test_evaluator")
        sys.path[:] = old_path
        for key in set(sys.modules) - set(old_modules) - {"nano_scoring_test_verifier", "nano_scoring_test_evaluator"}:
            del sys.modules[key]

    def score(self, candidate, reference=None, n=6):
        reference = reference or [dict(REF)] * n
        if isinstance(candidate, dict):
            candidate = [candidate] * n
        return self.ev.continuous_score(candidate, reference)

    def test_reference_scores_fifty(self):
        self.assertEqual(self.ev.REFERENCE_SCORE, 50.0)
        for n in (4, 6):
            self.assertEqual(self.score(dict(REF), n=n), 50.0)
        rows = [cand(0.3 * i, 1 + 0.01 * i, 1 - 0.02 * i) for i in range(6)]
        self.assertEqual(self.ev.continuous_score(rows, [dict(r) for r in rows]), 50.0)

    def test_statement_calibration_values(self):
        self.assertAlmostEqual(self.score(cand(dp=1.0)), 56.33, places=2)
        self.assertAlmostEqual(self.score(cand(dssim=0.9)), 57.79, places=2)
        self.assertAlmostEqual(self.score(cand(rlpips=0.9)), 56.05, places=2)
        self.assertAlmostEqual(self.score(cand(1.0, 0.9, 0.9)), 70.17, places=2)
        self.assertAlmostEqual(self.score(cand(dp=1.0, rlpips=1.02)), 48.81, places=2)

    def test_every_clip_counts_through_suite_means(self):
        base = [dict(REF)] * 6
        for clip in range(6):
            rows = list(base)
            rows[clip] = cand(dp=0.6)
            self.assertAlmostEqual(self.ev.continuous_score(rows, base), self.score(cand(dp=0.1)), places=9)
        # Only the suite means matter: redistributing a metric across clips keeps the score.
        spread = [cand(dp=d) for d in (1.2, -0.6, 0.0, 0.3, 0.9, -0.6)]
        self.assertAlmostEqual(self.ev.continuous_score(spread, base), self.score(cand(dp=0.2)), places=9)
        with self.assertRaises(ValueError):
            self.ev.continuous_score([dict(REF)] * 5, base)

    def test_bounded_and_finite(self):
        for row in (
            {"psnr": 99.0, "ssim": 1.0, "lpips": 0.0},
            {"psnr": 0.0, "ssim": -1.0, "lpips": 5.0},
            {"psnr": 99.0, "ssim": 0.0, "lpips": 0.0},
        ):
            value = self.score(row)
            self.assertTrue(math.isfinite(value) and 0.0 <= value <= 100.0, (row, value))
        self.assertGreater(self.score({"psnr": 99.0, "ssim": 1.0, "lpips": 0.0}), 99.9)
        self.assertLess(self.score({"psnr": 0.0, "ssim": -1.0, "lpips": 5.0}), 0.1)

    def test_monotone_in_each_metric(self):
        dps = (-0.5, -0.1, -0.02, 0.0, 0.02, 0.1, 0.5, 2.0)
        ratios = (1.2, 1.02, 1.002, 1.0, 0.998, 0.98, 0.8, 0.5)
        grid = list(itertools.product(dps, ratios, ratios))
        for dp, rs, rl in grid:
            here = self.score(cand(dp, rs, rl))
            self.assertGreaterEqual(self.score(cand(dp + 0.05, rs, rl)), here - 1e-12)
            self.assertGreaterEqual(self.score(cand(dp, rs * 0.99, rl)), here - 1e-12)
            self.assertGreaterEqual(self.score(cand(dp, rs, rl * 0.99)), here - 1e-12)

    def test_no_saturation_over_observed_range(self):
        # Observed A100 range: about -0.05..+1.9 dB, LPIPS -19 %..+1.4 %, (1 - SSIM) -14 %..+1 %.
        best = self.score(cand(1.9, 0.86, 0.81))
        self.assertLess(best, 85.0)
        self.assertGreater(self.score(cand(2.5, 0.75, 0.7)), best + 4.0)
        self.assertGreater(self.score(cand(1.0)) - 50.0, 6.0)  # +1 dB alone

    def test_one_metric_cannot_pay_for_a_clear_loss(self):
        # Large PSNR/SSIM gains with a 2 % worse LPIPS (e.g. blurring) score below the reference.
        self.assertLess(self.score(cand(1.85, 0.86, 1.02)), 50.0)
        # Large LPIPS gain with 0.2 dB lower PSNR also scores below the reference.
        self.assertLess(self.score(cand(-0.2, 1.0, 0.81)), 50.0)
        # Gains count fully inside the dead zone (every normalised gain >= -0.01).
        full = self.score(cand(1.0, 1.0, 1.0))
        self.assertAlmostEqual(self.score(cand(1.0, 1.0, 2 ** (0.4 * 0.01))), full - 50 / 3 * math.tanh(0.01), places=9)

    def test_pass_implies_above_reference(self):
        rng = random.Random(3407)
        for _ in range(500):
            reference = [cand(rng.uniform(-3, 3), rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)) for _ in range(6)]
            candidate = [{"psnr": r["psnr"] + rng.uniform(1e-6, 1.0), "ssim": r["ssim"] + (1 - r["ssim"]) * rng.uniform(1e-6, 0.3),
                          "lpips": r["lpips"] * rng.uniform(0.7, 1 - 1e-6)} for r in reference]
            self.assertGreater(self.ev.continuous_score(candidate, reference), 50.0)

    def test_equal_quality_noise_is_small(self):
        # Magnitudes of A100 reference-vs-reference repeats: |dPSNR| < 1e-4 dB, relative SSIM/LPIPS < 1e-4.
        for dp, rs, rl in itertools.product((-1e-4, 1e-4), (1 - 1e-4, 1 + 1e-4), (1 - 1e-4, 1 + 1e-4)):
            self.assertLess(abs(self.score(cand(dp, rs, rl)) - 50.0), 0.02)
        # Much larger equal-code differences (0.01 dB, 0.1 %) stay within a quarter point.
        for dp, rs, rl in itertools.product((-0.01, 0.01), (0.999, 1.001), (0.999, 1.001)):
            self.assertLess(abs(self.score(cand(dp, rs, rl)) - 50.0), 0.25)

    def evaluate(self, candidate, reference, role="final"):
        with (
            patch.object(self.ev, "_apply", return_value=SimpleNamespace(ok=True, workdir="/nonexistent/repo")),
            patch.object(self.ev, "_policy_check"),
            patch.object(self.ev, "_reference_rows", return_value=reference),
            patch.object(self.ev, "_run_entrypoint", return_value=candidate),
            patch.dict(os.environ, {"FRONTIER_SUBMISSION_ROLE": role}),
        ):
            return self.ev.evaluate("solution.patch")

    def test_evaluate_exports_relative_score_and_operator_rows(self):
        reference = [cand(0.1 * i) for i in range(6)]
        candidate = [cand(0.1 * i + 0.5, 0.9, 0.9) for i in range(6)]
        score, unbounded, _, metrics = self.evaluate(candidate, reference)
        self.assertAlmostEqual(score, self.ev.continuous_score(candidate, reference), places=12)
        self.assertEqual(score, unbounded)
        self.assertEqual(metrics["reference_score"], 50.0)
        self.assertAlmostEqual(metrics["margin"], score - 50.0, places=12)
        self.assertEqual(metrics["beats_reference"], 1)
        rows = metrics["per_workload_measurements"]
        self.assertEqual([row["workload"] for row in rows], [f"clip-{i:02d}" for i in range(1, 7)])
        self.assertEqual([row["candidate"] for row in rows], candidate)
        self.assertEqual([row["reference"] for row in rows], reference)

    def verify(self, value, mode):
        v = self.verifier
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            root = Path(tmp)
            payload, dataset = root / "payload", root / "dataset"
            payload.write_bytes(b"payload")
            dataset.write_bytes(b"dataset")
            with (
                patch.object(v, "load_candidate_diagnostic", return_value=None),
                patch.object(v, "extract_payload"),
                patch.object(v, "entrypoint_path", return_value=payload),
                patch.object(v, "stage_task", return_value=TASK / "judge"),
                patch.object(v, "load_evaluator", return_value=lambda _: value),
            ):
                return v.verify(bundle=TASK / "judge", entrypoint="solution.patch", payload=payload, dataset=dataset,
                                mode=mode, result_path=root / "result.json", evidence_path=root / "evidence.tar.zst")

    def test_operator_rows_never_reach_solver_feedback(self):
        reference = [dict(REF)] * 4
        candidate = [cand(0.4, 0.95, 0.95)] * 4
        value = self.evaluate(candidate, reference, role="train")
        self.assertIn("per_workload_measurements", value[3])
        train = self.verify(value, "train")
        self.assertEqual(set(train["metrics"]), {"score_unbounded", "psnr", "ssim", "lpips", "beat_spec_progress"})
        self.assertNotIn("private_metrics", train)
        final = self.verify(self.evaluate([cand(0.4, 0.95, 0.95)] * 6, [dict(REF)] * 6), "eval")
        self.assertEqual(final["metrics"], {"score_unbounded": final["raw_score"]})
        self.assertNotIn("per_workload_measurements", final["private_metrics"])
        self.assertEqual(final["private_metrics"]["reference_score"], 50.0)


if __name__ == "__main__":
    unittest.main()
