"""RUNLTS train feedback through the shared verifier.

Train shows the Agent only the task.toml [feedback] metrics and the aggregate
`beat_spec_progress`. A train candidate failure (storage, build, simulation, output) is a
scored zero whose metrics are `beat_spec_progress` with every train trace failing and the
`candidate_failure` code of the failing stage; its fixed sentence (evaluator SENTENCES) is the
private message. A scope or source-policy rejection is the shared verifier's, with one shared
category. A measured development submission returns the [feedback] values (candidate_failure
0), per-metric `beat_spec_progress` and `all_metrics_progress` for traces clearing both
prediction comparisons. Final exposes none of them. The evaluator results below are invented comparison
inputs; no predictor is built or run.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402

TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Arch/runlts_predictor"
# The judge as the Verifier image holds it: judge/, the generated config.yaml and pristine/.
BUNDLE = task_manifest.judge_bundle(TASK)
SHARED_VERIFIER = task_manifest.SHARED / "verifier"
# The beat spec's design record; train-visible metrics are task.toml [feedback].
DESIGN_SPEC = ROOT / "docs/task-authoring/runlts_predictor/provenance/beat_spec.json"
STAGES = ("semantic_audit", "storage_cap", "source_audit_incomplete", "build_failed",
          "simulator_run_failed", "output_format", "instruction_count_mismatch")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their defining module
    spec.loader.exec_module(module)
    return module


def never(*_args):
    raise AssertionError("evaluate() must not run for a rejected patch")


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
        for name in ("beat_spec_feedback", "evaluator", "source_policy", "edit_scope"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(SHARED_VERIFIER))
        sys.path.insert(0, str(task_manifest.SHARED / "lib"))  # isolate, beside the judge in the Verifier base
        sys.path.insert(0, str(BUNDLE))
        self.adapter = load("runlts_feedback_adapter", SHARED_VERIFIER / "fcs2_verifier.py")
        self.evaluator = load("runlts_feedback_evaluator", BUNDLE / "evaluator.py")
        self.edit_scope, _ = self.adapter.load_edit_scope(BUNDLE)
        os.environ["FRONTIER_VERIFIER_SCRATCH"] = str(self.tmp)
        self.payload = self.payload_of((BUNDLE / "baseline_naive.patch").read_bytes())

    def payload_of(self, patch, name="payload.tar"):
        path = self.tmp / name
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
            member = tarfile.TarInfo("payload/solution.patch")
            member.size = len(patch)
            archive.addfile(member, io.BytesIO(patch))
        return path

    def run_mode(self, mode, evaluate, check_tree=None, payload=None):
        """The shared verifier's verify() on the real bundle, payload and edit scope, with
        `evaluate` (and the judge's check_tree hook) standing in for the staged judge."""
        self.adapter.load_evaluator = lambda _task_dir: (evaluate, check_tree)
        try:
            return self.adapter.verify(
                bundle=BUNDLE, entrypoint="payload/solution.patch",
                payload=payload or self.payload, dataset=TASK / f"datasets/{mode}.json", mode=mode,
                result_path=self.tmp / f"{mode}-result.json",
                evidence_path=self.tmp / f"{mode}-evidence.tar.zst")
        except self.adapter.VerifierFailure as failure:
            return {"status": "failed", "failure": {"code": failure.code}}

    def returning(self, result):
        return lambda _tree: copy.deepcopy(result)

    def failing(self, category, message="private detail at /private/path line 12"):
        """The real evaluate() whose candidate stage fails with `category`."""
        e = self.evaluator

        def evaluate(tree):
            with mock.patch.object(e, "_evaluate", side_effect=e.CandidateError(message, category)):
                return e.evaluate(tree)
        return evaluate

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
        spec = json.loads(DESIGN_SPEC.read_text())
        names = list(task_manifest.read_task(TASK)["feedback"])
        self.assertEqual(names, [entry["name"] for entry in spec["solver_feedback"]] + ["candidate_failure"])
        self.assertEqual(names, self.evaluator.CONFIG["evaluation"]["solver_visible_metrics"])
        self.assertEqual(names, list(self.evaluator.FEEDBACK))
        self.assertFalse(set(names) & set(spec["withheld"]))
        self.assertNotIn("candidate_diagnostic", names)
        return names

    def failed_train(self, category):
        """The visible train metrics of a candidate failure: all 24 train traces fail all three
        comparisons, and candidate_failure is the stage's code (1..7, 8 for any other)."""
        return {"score_unbounded": 0.0,
                "beat_spec_progress": {name: self.counts(0) for name in
                                       ("br_mis_pki", "cyc_wp_pki", "logical_storage_bits")},
                "candidate_failure": STAGES.index(category) + 1 if category in STAGES else 8}

    def evidence(self, mode):
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(
            (self.tmp / f"{mode}-evidence.tar.zst").read_bytes())
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            return json.loads(archive.extractfile("metrics.json").read())

    def real_patch(self, name, edit):
        tree = self.tmp / "repo"
        shutil.copytree(BUNDLE / "pristine", tree)

        def git(*args):
            return subprocess.check_output(["git", "-C", str(tree), "-c", "user.name=Fixture",
                                            "-c", "user.email=fixture@localhost", *args])
        git("init", "-q")
        git("add", "-A")
        git("commit", "-qm", "base")
        path = tree / name
        path.write_text(edit(path.read_text()))
        patch = git("diff")
        shutil.rmtree(tree)
        return patch

    # ------------------------------------------------------------ candidate failures

    def test_train_candidate_failure_is_score_zero_and_its_stage_code_only(self):
        self.declared()
        self.assertEqual(tuple(self.evaluator.SENTENCES), STAGES)
        for category in (*STAGES, None):
            with self.subTest(category=category):
                res = self.run_mode("train", self.failing(category), self.evaluator.check_tree)
                self.assertEqual(res.get("status"), "passed", res)
                self.assertEqual(res["raw_score"], 0.0)
                self.assertEqual(res["metrics"], self.failed_train(category))
                self.assertNotIn("private_metrics", res)
                self.assertNotIn("t000", json.dumps(res["metrics"]))
                # Operator-side: the message is the fixed sentence (the private reason for a
                # failure with no stage), and the telemetry stays in the evidence archive.
                evidence = self.evidence("train")
                self.assertEqual(evidence["message"],
                                 f"Rejected: {self.evaluator.SENTENCES[category]}." if category
                                 else "private detail at /private/path line 12")
                self.assertEqual(evidence["metrics"]["private_runtime_telemetry"]["schema"],
                                 "runlts.private-runtime-telemetry/v1")

    def test_final_candidate_failure_keeps_its_private_reason_and_no_metric(self):
        res = self.run_mode("eval", self.failing("build_failed", "build or simulator execution failed"))
        self.assertEqual((res["status"], res["raw_score"], res["summary"], res["metrics"]),
                         ("passed", 0.0, "build or simulator execution failed", {"score_unbounded": 0.0}))
        # The operator-only record of a final zero: it did not beat the reference.
        self.assertEqual(res["private_metrics"], {"beats_reference": 0})
        self.assertNotIn("candidate_failure", json.dumps(res))

    def test_scope_and_source_policy_rejections_are_shared_categories(self):
        cases = (
            # The shared edit scope: a retained file.
            ("lib/cbp.cc", lambda s: s + "\n// edit\n", "out_of_scope"),
            # check_tree's source audit: whitespace between the regions, a forbidden word in one.
            ("my_cond_branch_predictor.cc",
             lambda s: s.replace("namespace nRUNLTS {\n", "namespace nRUNLTS { \n", 1), "out_of_scope"),
            ("my_cond_branch_predictor.cc",
             lambda s: s.replace("namespace nRUNLTS {\n\n", "namespace nRUNLTS {\n\nstatic int read = 0;\n", 1),
             "content_policy"),
        )
        for name, edit, category in cases:
            with self.subTest(name=name, category=category):
                payload = self.payload_of(self.real_patch(name, edit), "rejected.tar")
                for mode in ("train", "eval"):
                    res = self.run_mode(mode, never, self.evaluator.check_tree, payload)
                    self.assertEqual(res["status"], "passed")
                    self.assertEqual(res["raw_score"], 0.0)
                    self.assertEqual(res["metrics"], {"score_unbounded": 0.0, "rejected": category})
                    self.assertEqual(res["summary"], f"{self.edit_scope.CATEGORIES[category]} ({name})")

    def test_verifier_admits_no_task_diagnostic_or_task_category(self):
        sentence = self.evaluator.SENTENCES["build_failed"]
        for metrics in (
            {"candidate_diagnostic": {"schema": "runlts.candidate-diagnostic/v1",
                                      "category": "build_failed", "message": f"Rejected: {sentence}."}},
            {"rejected": "build_failed"},
            {"rejected": "out_of_scope", "train_br_mis_pki_mean": 1.0},
        ):
            with self.subTest(metrics=metrics):
                res = self.run_mode("train", self.returning((0.0, 0.0, "x", metrics)))
                self.assertEqual(res["status"], "failed")
                self.assertEqual(res["failure"]["code"], "invalid_evaluator_result")

    # ------------------------------------------------------------ measured results

    def test_measured_train_result_carries_the_joint_count(self):
        res = self.run_mode("train", self.returning(self.graded()), self.evaluator.check_tree)
        self.assertEqual(res["status"], "passed")
        self.assertEqual(set(res["metrics"]), {"score_unbounded", "beat_spec_progress", *self.declared()})
        self.assertEqual(res["metrics"]["candidate_failure"], 0)
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
        res = self.run_mode("eval", self.returning(result))
        self.assertEqual(res["metrics"], {"score_unbounded": result[1]})
        self.assertNotIn("all_metrics_progress", json.dumps(res["private_metrics"]))

    # ------------------------------------------------------------ private telemetry

    def telemetry(self):
        return self.evaluator.runtime_snapshot({"started": self.evaluator.time.monotonic(),
            "submission_role": "development", "traces": []})

    def test_private_telemetry_preserves_the_public_projection(self):
        for mode, original in (("train", self.graded()), ("eval", self.graded(final=True)),
                               ("eval", (0.0, 0.0, "candidate rejected", {}))):
            with self.subTest(mode=mode, message=original[2]):
                baseline = self.run_mode(mode, self.returning(original))
                result = copy.deepcopy(original)
                telemetry = self.telemetry()
                result[3]["private_runtime_telemetry"] = telemetry
                actual = self.run_mode(mode, self.returning(result))
                self.assertEqual(actual, baseline)
                self.assertEqual(self.evidence(mode)["metrics"]["private_runtime_telemetry"], telemetry)
                self.assertNotIn("private_runtime_telemetry", json.dumps(actual))

    def test_exception_telemetry_reaches_private_adapter_evidence(self):
        e = self.evaluator
        error = e.EvaluationError("private reference failure")
        with mock.patch.object(e, "_evaluate", side_effect=error):
            result = self.run_mode("train", e.evaluate, e.check_tree)
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
