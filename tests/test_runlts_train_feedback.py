"""RUNLTS train feedback: a named rejection category and a joint win count.

A rejected development submission returns `candidate_diagnostic`, which the
shipped verifier re-derives from the pinned judge/candidate_failure.py table. A
measured development submission returns per-metric `beat_spec_progress` and
`all_metrics_progress` for traces clearing both prediction comparisons. Final
exposes neither. The evaluator results below are invented comparison inputs; no
predictor is built or run.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
import copy
from unittest import mock
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_task_definition as renderer  # noqa: E402

TASK = renderer.problem_directory(
    ROOT / "tasks/frontier-cs-2.0-demo/problems", "runlts_predictor"
)
BUNDLE_FILES = ("beat_spec.json", "candidate-diagnostic.json", "candidate_failure.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their defining module
    spec.loader.exec_module(module)
    return module


class RunltsTrainFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="runlts-feedback-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.old_env = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.old_env)))
        modules = dict(sys.modules)
        old_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))

        def restore():
            for key in set(sys.modules) - set(modules):
                del sys.modules[key]
            sys.modules.update(modules)

        self.addCleanup(restore)
        for name in ("beat_spec_feedback", "evaluator", "candidate_failure",
                     "source_policy", "submission_patch"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(TASK / "verifier"))
        sys.path.insert(0, str(TASK / "judge"))
        self.adapter = load("runlts_feedback_adapter", TASK / "verifier/fcs2_verifier.py")
        self.evaluator = load("runlts_feedback_evaluator", TASK / "judge/evaluator.py")
        self.table = load("runlts_feedback_table", TASK / "judge/candidate_failure.py")
        os.environ["FRONTIER_VERIFIER_SCRATCH"] = str(self.tmp)
        self.bundle = self.tmp / "bundle"
        self.bundle.mkdir()
        (self.bundle / "evaluator.py").write_text("# supplied by the test\n")
        for name in BUNDLE_FILES:
            shutil.copy2(TASK / "judge" / name, self.bundle / name)
        self.payload = self.tmp / "payload.tar"
        with tarfile.open(self.payload, "w", format=tarfile.USTAR_FORMAT) as archive:
            member = tarfile.TarInfo("payload/solution.patch")
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))

    def run_mode(self, mode, result):
        self.adapter.load_evaluator = lambda _task_dir: (lambda _path: result)
        try:
            return self.adapter.verify(
                bundle=self.bundle, entrypoint="payload/solution.patch",
                payload=self.payload, dataset=TASK / f"datasets/{mode}.json", mode=mode,
                result_path=self.tmp / f"{mode}-result.json",
                evidence_path=self.tmp / f"{mode}-evidence.tar.zst")
        except self.adapter.VerifierFailure as failure:
            return {"status": "failed", "failure": {"code": failure.code}}

    def rejected(self, category):
        visible = self.table.diagnostic(category)
        return 0.0, 0.0, visible["message"], {"candidate_diagnostic": visible}

    def graded(self, final=False):
        """24 (or 105) rows: 9 win both comparisons, 5 only one, 4 only the other."""
        e = self.evaluator
        rows = [{"id": f"fixture-{i}"} for i in range(105 if final else 24)]
        reference = {r["id"]: e.Measurement(1000, 20, 200, 2000, Decimal("20"), Decimal("200"))
                     for r in rows}

        def cell(i):
            b, w = (10, 100) if i < 9 else (10, 200) if i < 14 else (20, 100) if i < 18 else (20, 300)
            return e.Measurement(1000, b, w, 2000, Decimal(b), Decimal(w))

        candidate = {r["id"]: cell(i) for i, r in enumerate(rows)}
        return e.grade(rows, candidate, reference, final, logical_storage_bits=1572864)

    @staticmethod
    def counts(passed, total=24):
        return {"passed": passed, "failed": total - passed, "total": total,
                "pass_percent": round(100.0 * passed / total, 2),
                "fail_percent": round(100.0 * (total - passed) / total, 2)}

    def declared(self):
        spec = json.loads((TASK / "judge/beat_spec.json").read_text())
        policy = json.loads((TASK / "policy.json").read_text())
        names = [entry["name"] for entry in spec["solver_feedback"]]
        self.assertEqual(names, policy["metadata"]["evaluation"]["solver_visible_metrics"])
        self.assertEqual(names, self.evaluator.CONFIG["evaluation"]["solver_visible_metrics"])
        self.assertFalse(set(names) & set(spec["withheld"]))
        return names

    def test_every_category_reaches_train_feedback_as_the_table_entry(self):
        self.assertIn("candidate_diagnostic", self.declared())
        for category, sentence in self.table.CATEGORIES.items():
            with self.subTest(category=category):
                res = self.run_mode("train", self.rejected(category))
                self.assertEqual(res["status"], "passed")
                self.assertEqual(res["raw_score"], 0.0)
                self.assertEqual(res["metrics"], {
                    "score_unbounded": 0.0,
                    "candidate_diagnostic": {
                        "schema": "runlts.candidate-diagnostic/v1", "category": category,
                        "message": f"Rejected: {sentence}."}})
                self.assertNotIn("private_metrics", res)

    def test_final_never_shows_a_diagnostic(self):
        res = self.run_mode("eval", self.rejected("build_failed"))
        self.assertEqual(res["metrics"], {"score_unbounded": 0.0})
        # What the evaluator itself returns for a final rejection is unchanged.
        res = self.run_mode("eval", (0.0, 0.0, "build or simulator execution failed", {}))
        self.assertEqual((res["status"], res["metrics"]), ("passed", {"score_unbounded": 0.0}))

    def test_verifier_rejects_anything_but_the_table_entry(self):
        good = self.table.diagnostic("forbidden_construct")
        for metrics in (
            {"candidate_diagnostic": dict(good, message=good["message"] + " at line 12")},
            {"candidate_diagnostic": dict(good, category="not_a_category")},
            {"candidate_diagnostic": dict(good, symbol="hist_to_use")},
            {"candidate_diagnostic": dict(good, schema="other/v1")},
            {"candidate_diagnostic": "Rejected: free text"},
            {"candidate_diagnostic": good, "train_br_mis_pki_mean": 1.0},
        ):
            with self.subTest(metrics=metrics):
                res = self.run_mode("train", (0.0, 0.0, "x", metrics))
                self.assertEqual(res["status"], "failed")
                self.assertEqual(res["failure"]["code"], "invalid_evaluator_result")
        res = self.run_mode("train", (1.0, 1.0, "x", {"candidate_diagnostic": good}))
        self.assertEqual(res["failure"]["code"], "invalid_evaluator_result")

    def test_bundle_without_the_declaration_cannot_emit_a_diagnostic(self):
        (self.bundle / "candidate-diagnostic.json").unlink()
        res = self.run_mode("train", self.rejected("patch_policy"))
        self.assertEqual(res["failure"]["code"], "invalid_evaluator_result")

    def test_changed_helper_is_a_configuration_error(self):
        with (self.bundle / "candidate_failure.py").open("a") as stream:
            stream.write("CATEGORIES['x'] = 'free text'\n")
        res = self.run_mode("train", self.rejected("patch_policy"))
        self.assertEqual(res["failure"]["code"], "verifier_configuration_error")

    def test_measured_train_result_carries_the_joint_count(self):
        result = self.graded()
        res = self.run_mode("train", result)
        self.assertEqual(res["status"], "passed")
        self.assertEqual(
            set(res["metrics"]),
            {"score_unbounded", "beat_spec_progress", *self.declared()} - {"candidate_diagnostic"})
        self.assertEqual(res["metrics"]["all_metrics_progress"], self.counts(9))
        self.assertEqual(res["metrics"]["beat_spec_progress"], {
            "br_mis_pki": self.counts(14), "cyc_wp_pki": self.counts(13),
            "logical_storage_bits": self.counts(24)})
        self.assertEqual(res["metrics"]["category_means"], {"uncategorized": {
            "traces": 24, "br_mis_pki_mean": 340 / 24, "cyc_wp_pki_mean": 4100 / 24}})
        encoded = json.dumps(res)
        for leaked in ("fixture-", "reference", "margin", "beats_reference",
                       "beat_spec_workload_passes", "paired_measurements", "logical_storage_accounting"):
            self.assertNotIn(leaked, encoded)

    def test_final_result_exposes_no_progress(self):
        result = self.graded(final=True)
        self.assertNotIn("all_metrics_progress", result[3])
        self.assertNotIn("category_means", result[3])
        res = self.run_mode("eval", result)
        self.assertEqual(res["metrics"], {"score_unbounded": result[1]})
        self.assertNotIn("all_metrics_progress", json.dumps(res["private_metrics"]))

    def evidence(self, mode):
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(
            (self.tmp / f"{mode}-evidence.tar.zst").read_bytes())
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            return json.loads(archive.extractfile("metrics.json").read())

    def telemetry(self):
        return self.evaluator.runtime_snapshot({"started": self.evaluator.time.monotonic(),
            "submission_role": "development", "traces": []})

    def test_private_telemetry_preserves_real_adapter_public_projection(self):
        for mode, original in (("train", self.graded()), ("eval", self.graded(final=True)),
                               ("train", self.rejected("build_failed")),
                               ("eval", (0.0, 0.0, "candidate rejected", {}))):
            with self.subTest(mode=mode, message=original[2]):
                baseline = self.run_mode(mode, copy.deepcopy(original))
                result = copy.deepcopy(original)
                telemetry = self.telemetry()
                result[3]["private_runtime_telemetry"] = telemetry
                actual = self.run_mode(mode, result)
                self.assertEqual(actual, baseline)
                self.assertEqual(self.evidence(mode)["metrics"]["private_runtime_telemetry"], telemetry)
                self.assertNotIn("private_runtime_telemetry", json.dumps(actual))

    def test_telemetry_is_validated_before_private_projection(self):
        for field, value in (("schema", "other"), ("traces", "private"),
                             ("elapsed_seconds", -1), ("longest_trace", {})):
            telemetry = self.telemetry()
            telemetry[field] = value
            result = self.rejected("build_failed")
            result[3]["private_runtime_telemetry"] = telemetry
            actual = self.run_mode("train", result)
            self.assertEqual(actual["failure"]["code"], "invalid_evaluator_result")

    def test_exception_telemetry_reaches_private_adapter_evidence(self):
        e = self.evaluator
        error = e.EvaluationError("private reference failure")
        with mock.patch.object(e, "_evaluate", side_effect=error):
            self.adapter.load_evaluator = lambda _task: e.evaluate
            result = self.adapter.verify(bundle=self.bundle, entrypoint="payload/solution.patch",
                payload=self.payload, dataset=TASK / "datasets/train.json", mode="train",
                result_path=self.tmp / "train-result.json",
                evidence_path=self.tmp / "train-evidence.tar.zst")
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("private_runtime_telemetry", json.dumps(result))
        evidence = self.evidence("train")
        telemetry = evidence["metrics"]["private_execution"]["private_runtime_telemetry"]
        self.assertEqual(telemetry["schema"], "runlts.private-runtime-telemetry/v1")

    def test_failed_trace_is_recorded_without_changing_error(self):
        e = self.evaluator
        ledger = {"started": e.time.monotonic(), "submission_role": "final", "traces": []}
        token = e._RUNTIME_TELEMETRY.set(ledger)
        try:
            with mock.patch.object(e, "run_trace", side_effect=e.CandidateError("execution failed")):
                with self.assertRaises(e.CandidateError):
                    e.timed_trace("candidate", Path("."), {"id": "private-trace"}, 0)
        finally:
            e._RUNTIME_TELEMETRY.reset(token)
        row = ledger["traces"][0]
        self.assertEqual((row["role"], row["workload"], row["completed"]),
                         ("candidate", "private-trace", False))
        self.assertGreaterEqual(row["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
