"""JoinTree train feedback is aggregate; final stays private.

The evaluator emits a native per-query `queries` dict. In train mode the task
verifier turns it into aggregate counts through the shipped beat-spec helper:
per-metric `beat_spec_progress`, `all_metrics_progress` for queries clearing
both comparisons, and the candidate's own suite-median latency. A train result
carries exactly the fields declared in judge/beat_spec.json and policy.json and
nothing withheld; an eval result exposes no solver-visible metric.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_task_definition as renderer  # noqa: E402

TASK = renderer.problem_directory(
    ROOT / "tasks/frontier-cs-2.0-demo/problems", "join_tree_rewriting"
)
EXPECTED = [f"dev/{i}" for i in range(1, 21)] + [f"final/{i}" for i in range(1, 20)]


def load_comparison():
    old = list(sys.path)
    sys.path.insert(0, str(TASK / "judge"))
    try:
        for name in ("query_measurements",):
            sys.modules.pop(name, None)
        return importlib.import_module("query_measurements")
    finally:
        sys.path[:] = old


def probe(qm, identity, *, candidate=10.0, correct=True, identical=False):
    cm = {m: [1.0] for m in qm.CORRECTNESS_METHODS}
    cmm = {m: [correct if m == "Candidate" else True] for m in qm.CORRECTNESS_METHODS}
    tm = {
        m: [
            100.0 if m.startswith("Binary") else 20.0 if m.startswith("Reference") else candidate
        ] * 5
        for m in qm.TIMING_METHODS
    }
    tmm = {m: [correct if m.startswith("Candidate") else True] * 5 for m in qm.TIMING_METHODS}
    return qm.compare_query(
        identity=identity,
        correctness={"durations_us": cm, "matches_original": cmm},
        timing={"durations_us": tm, "matches_original": tmm},
        identical=identical,
        repetitions=5,
    )


class JoinTreeTrainFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((TASK / "judge/beat_spec.json").is_file())
        self.qm = load_comparison()
        self.tmp = Path(tempfile.mkdtemp(prefix="jt-routing-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.old_env = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.old_env)))
        self.old_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.old_path))
        sys.path.insert(0, str(TASK / "verifier"))
        # Restore any cached beat_spec_feedback so this task's verifier-local
        # copy never leaks into a later test that loads another task's
        # verifier. Pop before use and restore the prior module after.
        saved = sys.modules.get("beat_spec_feedback")
        self.addCleanup(
            lambda s=saved: sys.modules.__setitem__("beat_spec_feedback", s)
            if s is not None else sys.modules.pop("beat_spec_feedback", None))
        for name in ("beat_spec_feedback",):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location(
            "jt_feedback_adapter", TASK / "verifier/fcs2_verifier.py")
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        os.environ["FRONTIER_VERIFIER_SCRATCH"] = str(self.tmp)
        self.bundle = self.tmp / "bundle"
        self.bundle.mkdir()
        (self.bundle / "evaluator.py").write_text("# supplied by the test\n")
        shutil.copy2(TASK / "judge/beat_spec.json", self.bundle / "beat_spec.json")
        self.payload = self.tmp / "payload.tar"
        with tarfile.open(self.payload, "w", format=tarfile.USTAR_FORMAT) as archive:
            member = tarfile.TarInfo("payload/solution.patch")
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))

    def native_result(self, selected, correct=True):
        queries = {q: probe(self.qm, q, correct=correct) for q in selected}
        score, metrics = self.qm.summarize_queries(selected, queries)
        return score, score, "probe", metrics

    def mixed_result(self, selected):
        """13 queries faster than the reference, 7 slower, one fast one incorrect."""
        queries = {
            q: probe(self.qm, q, candidate=10.0 + i if i < 13 else 25.0 + i,
                     correct=i != 2)
            for i, q in enumerate(selected)}
        score, metrics = self.qm.summarize_queries(selected, queries)
        return score, score, "probe", metrics

    def run_mode(self, mode, result):
        os.environ["FRONTIER_SUBMISSION_ROLE"] = (
            "final" if mode == "eval" else "development")
        self.adapter.load_evaluator = lambda _task_dir: (lambda _path: result)
        return self.adapter.verify(
            bundle=self.bundle, entrypoint="payload/solution.patch",
            payload=self.payload, dataset=TASK / f"datasets/{mode}.json", mode=mode,
            result_path=self.tmp / f"{mode}-result.json",
            evidence_path=self.tmp / f"{mode}-evidence.tar.zst")

    @staticmethod
    def counts(passed, total=20):
        return {"passed": passed, "failed": total - passed, "total": total,
                "pass_percent": round(100.0 * passed / total, 2),
                "fail_percent": round(100.0 * (total - passed) / total, 2)}

    def declared_keys(self):
        spec = json.loads((TASK / "judge/beat_spec.json").read_text())
        policy = json.loads((TASK / "policy.json").read_text())
        declared = [entry["name"] for entry in spec["solver_feedback"]]
        self.assertEqual(
            sorted(declared),
            sorted(policy["metadata"]["evaluation"]["solver_visible_metrics"]))
        self.assertFalse(set(declared) & set(spec["withheld"]))
        return {"score_unbounded", "beat_spec_progress", *declared}

    def test_train_carries_exactly_the_declared_aggregates(self):
        result = self.mixed_result(EXPECTED[:20])
        res = self.run_mode("train", result)
        self.assertEqual(res["status"], "passed")
        self.assertEqual(set(res["metrics"]), self.declared_keys())
        self.assertEqual(res["metrics"], {
            "score_unbounded": result[1],
            "beat_spec_progress": {
                "exact_result": self.counts(19),
                "median_latency": self.counts(10),
            },
            "all_metrics_progress": self.counts(9),
            # Candidate per-query medians are 10..22 and 38..44; median of 20.
            "candidate_median_latency_us": 19.5,
        })
        self.assertNotIn("private_metrics", res)

    def test_train_exposes_nothing_withheld(self):
        result = self.mixed_result(EXPECTED[:20])
        res = self.run_mode("train", result)
        encoded = json.dumps(res)
        for leaked in ("reference", "margin", "beats_reference", "binary",
                       "beat_spec_workload_passes", "queries", "durations_us",
                       "correctness_pass", "latency_pass", "dev/", "final/"):
            self.assertNotIn(leaked, encoded)
        # No reference latency (20.0), baseline latency (100.0) or per-query
        # candidate value other than the suite median appears as a number.
        def numbers(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from numbers(item)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                yield value
        visible = set(numbers(res["metrics"])) - {res["metrics"]["score_unbounded"]}
        self.assertEqual(
            visible, {19.0, 1.0, 20.0, 95.0, 5.0, 10.0, 50.0, 9.0, 11.0, 45.0, 55.0, 19.5})

    def test_counts_do_not_depend_on_which_queries_pass(self):
        selected = EXPECTED[:20]
        first = self.run_mode("train", self.mixed_result(selected))
        second = self.run_mode("train", self.mixed_result(selected[::-1]))
        self.assertEqual(first["metrics"], second["metrics"])

    def test_unmeasured_query_withholds_the_latency_median(self):
        selected = EXPECTED[:20]
        queries = {q: probe(self.qm, q) for q in selected}
        queries[selected[3]] = self.qm.compare_query(
            identity=selected[3], correctness=None, timing=None, identical=False,
            repetitions=5, failure="candidate_lowering_failed")
        score, metrics = self.qm.summarize_queries(selected, queries)
        res = self.run_mode("train", (score, score, "probe", metrics))
        self.assertEqual(set(res["metrics"]), self.declared_keys())
        self.assertIsNone(res["metrics"]["candidate_median_latency_us"])
        self.assertEqual(res["metrics"]["all_metrics_progress"], self.counts(19))
        self.assertEqual(
            res["metrics"]["beat_spec_progress"]["exact_result"], self.counts(19))

    def test_rejected_patch_reports_zero_progress(self):
        selected = EXPECTED[:20]
        queries = {
            q: self.qm.compare_query(
                identity=q, correctness=None, timing=None, identical=False,
                repetitions=5, failure="candidate_not_measured")
            for q in selected}
        score, metrics = self.qm.summarize_queries(selected, queries)
        res = self.run_mode("train", (score, score, "candidate patch rejected", metrics))
        self.assertEqual(res["metrics"], {
            "score_unbounded": 0.0,
            "beat_spec_progress": {
                "exact_result": self.counts(0), "median_latency": self.counts(0)},
            "all_metrics_progress": self.counts(0),
            "candidate_median_latency_us": None,
        })

    def test_inconsistent_private_decisions_fail_closed(self):
        _s, _u, _m, metrics = self.native_result(EXPECTED[:20])
        forged = json.loads(json.dumps(metrics))
        forged["beats_reference"] = 0
        with self.assertRaises(self.adapter.VerifierFailure):
            self.run_mode("train", (50.0, 50.0, "probe", forged))
        partial = json.loads(json.dumps(metrics))
        partial["queries"].pop(EXPECTED[0])
        with self.assertRaises(self.adapter.VerifierFailure):
            self.run_mode("train", (50.0, 50.0, "probe", partial))

    def test_train_requires_the_bundled_beat_spec(self):
        (self.bundle / "beat_spec.json").unlink()
        with self.assertRaises(self.adapter.VerifierFailure):
            self.run_mode("train", self.native_result(EXPECTED[:20]))

    def test_eval_exposes_no_solver_metric_and_keeps_private_verdict(self):
        res = self.run_mode("eval", self.native_result(EXPECTED))
        self.assertEqual(res["status"], "passed")
        self.assertEqual(res["metrics"], {})
        self.assertEqual(
            set(res), {"status", "raw_score", "summary", "metrics", "private_metrics"})
        private = res["private_metrics"]
        self.assertEqual(
            set(private),
            {"beats_reference", "aggregation", "beat_spec_progress",
             "reference_score", "margin"})
        self.assertEqual(private["beats_reference"], 1)
        self.assertEqual(len(private["beat_spec_progress"]), 78)
        self.assertEqual(private["aggregation"]["kind"], "all_workloads_all_metrics")
        self.assertEqual(
            private["beat_spec_progress"]["final/19/median_latency"],
            {"passed": 1, "failed": 0, "total": 1, "pass_percent": 100,
             "fail_percent": 0})

    def test_eval_fails_closed_on_one_bad_query(self):
        queries = {q: probe(self.qm, q) for q in EXPECTED}
        queries[EXPECTED[-1]] = probe(self.qm, EXPECTED[-1], correct=False)
        score, metrics = self.qm.summarize_queries(EXPECTED, queries)
        res = self.run_mode("eval", (score, score, "probe", metrics))
        self.assertEqual(res["metrics"], {})
        self.assertEqual(res["private_metrics"]["beats_reference"], 0)


if __name__ == "__main__":
    unittest.main()
