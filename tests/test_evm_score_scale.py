"""EVM State Checkpoints: the continuous score no longer saturates.

The score was 100 * log10(Q_C / Q_B) clipped to [0, 100], so a twelvefold composite
quality and a tenfold one both scored 100. It is now 100 * tanh(decades / 1.36) of
the same unbounded log ratio: zero at the shipped backend, strictly increasing,
below 100 for every finite quality. The beat-reference rule is unchanged. Every
workload's candidate, reference and shipped-backend values are kept in evidence.
"""
import deployment_test_guard  # noqa: F401 # isort: skip

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
EVM = ROOT / "tasks/frontier-cs-2.0-demo/problems/Database/evm_state_checkpoints"
FIELDS = ("throughput", "create_ns", "restore_ns", "retained_bytes")


def load(path: Path, name: str, extra_path: Path):
    sys.path.insert(0, str(extra_path))
    try:
        for helper in ("submission_patch", "beat_spec_feedback"):
            sys.modules.pop(helper, None)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(extra_path))


def observation(tps, create, restore, retained):
    return {"throughput": tps, "create_ns": create, "restore_ns": restore, "retained_bytes": retained}


class EvmScoreScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = load(EVM / "judge/evaluator.py", "evm_score_scale_evaluator", EVM / "judge")
        cls.feedback = load(EVM / "verifier/beat_spec_feedback.py", "evm_score_scale_feedback",
                            EVM / "verifier")

    def setUp(self):
        self.e._REFERENCE_ANCHOR.clear()
        self.addCleanup(self.e._REFERENCE_ANCHOR.clear)

    def test_mapping_is_zero_at_the_shipped_backend_and_strictly_increasing_below_100(self):
        bounded = self.e._bounded_score
        for value in (0.0, -0.1, -40.0, float("nan"), float("-inf")):
            self.assertEqual(bounded(value), 0.0)
        previous = 0.0
        for unbounded in [x * 2.5 for x in range(1, 200)]:  # up to 10^4.975x
            score = bounded(unbounded)
            self.assertGreater(score, previous)
            self.assertLess(score, 100.0)
            previous = score
        self.assertLess(bounded(1e6), 100.0 + 1e-12)

    def test_recorded_finals_no_longer_clip(self):
        bounded = self.e._bounded_score
        # Unbounded final scores recorded on the uniform-metrics version: three were clipped to 100.
        recorded = {"a": 107.74601365893348, "b": 106.44073372448088,
                    "c": 102.54355766242458, "d": 83.70437128299251,
                    "e": 63.860724954667035}
        scores = {name: bounded(value) for name, value in recorded.items()}
        self.assertAlmostEqual(scores["a"], 65.97, places=2)
        self.assertAlmostEqual(scores["b"], 65.42, places=2)
        self.assertAlmostEqual(scores["c"], 63.75, places=2)
        self.assertEqual(sorted(scores, key=scores.get), sorted(recorded, key=recorded.get))
        self.assertTrue(all(score < 70 for score in scores.values()))
        # The platform's reference anchor (median 74.76 unbounded) sits at 50.
        self.assertAlmostEqual(bounded(74.76154275002324), 50.0, delta=0.05)
        # A further doubling of the best recorded quality still earns about 11 points.
        self.assertGreater(bounded(107.746 + 100 * math.log10(2)) - scores["a"], 10)

    def test_gate_failure_reports_the_mapped_reference_score(self):
        self.e._REFERENCE_ANCHOR[True] = 74.76
        metrics = self.e._gate_metrics(True)
        self.assertAlmostEqual(metrics["reference_score"], self.e._bounded_score(74.76))
        self.assertEqual(metrics["beats_reference"], 0)

    def synthetic(self, suite):
        samples, profiles = {}, {}
        for index, name in enumerate(("candidate", "naive", "reference")):
            scale = (3.0, 1.0, 2.0)[index]
            samples[name] = {case: [observation(1000 * scale + r, 100 / scale + r, 50 / scale, 4096 / scale)
                                    for r in range(5)] for case, _ in suite}
            profiles[name] = {"per_seed": {seed: [observation(10 * scale + r, 40 / scale, 30 / scale + r,
                                                              2048 / scale) for r in range(5)]
                                           for _, seed in suite}}
        return samples, profiles

    def test_workload_values_reproduce_the_composite_for_every_variant(self):
        e = self.e
        suite = e.FINAL_SUITE
        samples, profiles = self.synthetic(suite)
        values = e._workload_values(suite, samples, profiles)
        self.assertEqual(set(values), set(e._workload_ids(suite)))
        for workload, row in values.items():
            self.assertEqual(set(row), {"candidate", "naive", "reference"})
            for variant in row.values():
                self.assertEqual(set(variant), set(FIELDS) | {"quality", "observations"})
                self.assertEqual(len(variant["observations"]["throughput"]), 5)
        binaries = {name: Path("/bin/true") for name in samples}
        # Recompute each variant's composite from the exported values alone.
        for name in samples:
            logs = [math.log(values[w][name]["quality"]) for w in values]
            exported = math.exp(sum(logs) / len(logs))
            fake = iter([run for case, _ in suite for run in samples[name][case]])
            with mock.patch.object(e, "_one_benchmark", side_effect=lambda *a: next(fake)):
                measured = e._qualities({name: binaries[name]}, Path("/"), suite,
                                        {name: profiles[name]}, 5)[0][name]
            self.assertAlmostEqual(exported, measured, places=9)

    def test_evaluate_scores_with_the_mapping_and_keeps_workload_values_out_of_feedback(self):
        e = self.e
        suite = e.FINAL_SUITE
        samples, profiles = self.synthetic(suite)
        qualities = {"candidate": 11.95, "naive": 1.0, "reference": 10 ** 0.7476}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            work = tmp / "submission" / "work"
            (work / e.EDITABLE).parent.mkdir(parents=True)
            (work / e.EDITABLE).write_bytes(b"// candidate\n")
            patch = tmp / "solution.patch"
            patch.write_text("+++ b/lib/evmcow/execution_state.hpp\n+// fine\n")
            applied = SimpleNamespace(ok=True, message="", metrics={"valid_patch": 1}, workdir=work)
            cache = tmp / "cache"
            (cache / "source" / e.EDITABLE).parent.mkdir(parents=True)

            def profile(tree, output, seeds):
                name = "candidate" if tree == work else ("naive" if tree == e.APP_DIR else "reference")
                return {"create_ns": 1.0, "restore_ns": 1.0, "retained_bytes": 1.0,
                        "per_seed": {seed: profiles[name]["per_seed"][seed][:1] for seed in seeds}}

            def repeat(profs, binaries, seeds):
                for name in ("candidate", "reference"):
                    for seed in seeds:
                        profs[name]["per_seed"][seed] = profiles[name]["per_seed"][seed]

            with mock.patch.dict("os.environ", {"FRONTIER_SUBMISSION_ROLE": "final",
                                                "FRONTIER_SPECTRUM_CACHE": str(cache)}), \
                    mock.patch.object(e, "APP_DIR", EVM / "agent-base/frontier/task/repo"), \
                    mock.patch.object(e, "apply_submission_patch", return_value=applied), \
                    mock.patch.object(e, "_ensure_cache", return_value=(cache, cache / "source")), \
                    mock.patch.object(e, "_reference_header", return_value=b"// reference\n"), \
                    mock.patch.object(e, "_checkpoint_profile", side_effect=profile), \
                    mock.patch.object(e, "_build_variant"), \
                    mock.patch.object(e, "_qualities", return_value=(qualities, samples)), \
                    mock.patch.object(e, "_repeat_checkpoint_profiles", side_effect=repeat):
                score, unbounded, message, metrics = e.evaluate(str(patch))
        self.assertIn("valid Spectrum checkpoint backend", message)
        self.assertAlmostEqual(unbounded, 100 * math.log10(11.95))
        self.assertAlmostEqual(score, e._bounded_score(unbounded))
        self.assertAlmostEqual(metrics["reference_score"], 50.0, delta=0.05)
        self.assertAlmostEqual(metrics["margin"], score - metrics["reference_score"])
        self.assertEqual(set(metrics["workload_values"]), set(e._workload_ids(suite)))
        self.assertAlmostEqual(metrics["reference_score_unbounded"], 74.76, places=6)
        json.dumps(metrics, allow_nan=False)
        spec = EVM / "judge/beat_spec.json"
        visible = self.feedback.public_metrics(metrics, beat_spec_path=spec)
        private = self.feedback.private_metrics(metrics, beat_spec_path=spec)
        for projected in (visible, private):
            self.assertNotIn("workload_values", projected)
            self.assertNotIn("reference_score_unbounded", projected)
            self.assertNotIn("reference_quality", projected)


if __name__ == "__main__":
    unittest.main()
