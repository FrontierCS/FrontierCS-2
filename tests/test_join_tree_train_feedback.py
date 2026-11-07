"""JoinTree train feedback is aggregate; final stays private; the source policy is check_tree.

The evaluator's summarize_queries emits the per-query decisions (beat_spec_workload_passes)
and the two solver_feedback aggregates; the shared verifier turns the decisions into train
`beat_spec_progress` and the private final verdict. A train result carries exactly the fields
declared in task.toml [feedback] and nothing withheld; an eval result exposes no solver
metric beyond score_unbounded.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_task_definition as renderer  # noqa: E402
import task_manifest  # noqa: E402

TASK = renderer.problem_directory(
    ROOT / "tasks/frontier-cs-2.0-demo/problems", "join_tree_rewriting"
)
EDITABLE = "2phase_nsa/binary_plan/rewrite_policy.py"
EXPECTED = [f"dev/{i}" for i in range(1, 21)] + [f"final/{i}" for i in range(1, 20)]


def load_judge_module(name):
    old = list(sys.path)
    sys.path.insert(0, str(TASK / "judge"))
    try:
        for module in ("query_measurements", name):
            sys.modules.pop(module, None)
        return importlib.import_module(name)
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


def patch_of(source: str) -> bytes:
    """A git diff of the editable module replaced by `source`."""
    with tempfile.TemporaryDirectory() as scratch:
        repo = Path(scratch) / "repo"
        shutil.copytree(TASK / "repo", repo)
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_CEILING_DIRECTORIES": scratch}
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
        subprocess.run([*git, "init", "-q"], check=True, env=env)
        subprocess.run([*git, "add", "-A"], check=True, env=env)
        subprocess.run([*git, "commit", "-qm", "base"], check=True, env=env)
        (repo / EDITABLE).write_text(source)
        return subprocess.run([*git, "diff"], check=True, env=env, capture_output=True).stdout


class JoinTreeTrainFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.qm = load_judge_module("query_measurements")
        self.tmp = Path(tempfile.mkdtemp(prefix="jt-routing-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.old_env = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.old_env)))
        self.old_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.old_path))
        shared = task_manifest.SHARED / "verifier"
        sys.path.insert(0, str(shared))
        saved = sys.modules.get("beat_spec_feedback")
        self.addCleanup(
            lambda s=saved: sys.modules.__setitem__("beat_spec_feedback", s)
            if s is not None else sys.modules.pop("beat_spec_feedback", None))
        sys.modules.pop("beat_spec_feedback", None)
        spec = importlib.util.spec_from_file_location("jt_shared_verifier", shared / "fcs2_verifier.py")
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        os.environ["FRONTIER_VERIFIER_SCRATCH"] = str(self.tmp)
        # A light bundle: the shared verifier needs the evaluator file, the beat spec, the
        # generated edit-scope config and the pristine starter.
        self.bundle = self.tmp / "bundle"
        self.bundle.mkdir()
        (self.bundle / "evaluator.py").write_text("# supplied by the test\n")
        shutil.copytree(TASK / "repo", self.bundle / "pristine")
        generated = task_manifest.generated_files(TASK)
        (self.bundle / "config.yaml").write_bytes(
            generated[f"{task_manifest.GENERATED}/judge/config.yaml"])
        source = (TASK / "repo" / EDITABLE).read_text()
        self.payload = self.make_payload(patch_of(source + "\n# a candidate\n"))

    def make_payload(self, patch: bytes) -> Path:
        path = self.tmp / f"payload-{len(list(self.tmp.glob('payload-*')))}.tar"
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
            member = tarfile.TarInfo("payload/solution.patch")
            member.size = len(patch)
            archive.addfile(member, io.BytesIO(patch))
        return path

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

    def run_mode(self, mode, result, payload=None, check_tree=None):
        self.adapter.load_evaluator = lambda _task_dir: ((lambda _tree: result), check_tree)
        return self.adapter.verify(
            bundle=self.bundle, entrypoint="payload/solution.patch",
            payload=payload or self.payload, dataset=TASK / f"datasets/{mode}.json", mode=mode,
            result_path=self.tmp / f"{mode}-result.json",
            evidence_path=self.tmp / f"{mode}-evidence.tar.zst")

    @staticmethod
    def counts(passed, total=20):
        return {"passed": passed, "failed": total - passed, "total": total,
                "pass_percent": round(100.0 * passed / total, 2),
                "fail_percent": round(100.0 * (total - passed) / total, 2)}

    def declared_keys(self):
        policy = task_manifest.load_policy(TASK)
        declared = policy["metadata"]["evaluation"]["solver_visible_metrics"]
        self.assertEqual(sorted(declared), ["all_metrics_progress", "candidate_median_latency_us"])
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
        self.assertFalse(metrics["reference_score_complete"])
        res = self.run_mode("train", (score, score, "probe", metrics))
        self.assertEqual(set(res["metrics"]), self.declared_keys())
        self.assertIsNone(res["metrics"]["candidate_median_latency_us"])
        self.assertEqual(res["metrics"]["all_metrics_progress"], self.counts(19))
        self.assertEqual(
            res["metrics"]["beat_spec_progress"]["exact_result"], self.counts(19))

    def test_every_query_failing_reports_zero_progress(self):
        selected = EXPECTED[:20]
        queries = {
            q: self.qm.compare_query(
                identity=q, correctness=None, timing=None, identical=False,
                repetitions=5, failure="candidate_lowering_failed")
            for q in selected}
        score, metrics = self.qm.summarize_queries(selected, queries)
        res = self.run_mode("train", (score, score, "probe", metrics))
        self.assertEqual(res["metrics"], {
            "score_unbounded": 0.0,
            "beat_spec_progress": {
                "exact_result": self.counts(0), "median_latency": self.counts(0)},
            "all_metrics_progress": self.counts(0),
            "candidate_median_latency_us": None,
        })

    def test_inconsistent_private_decisions_fail_closed(self):
        _s, _u, _m, metrics = self.native_result(EXPECTED[:20])
        partial = json.loads(json.dumps(metrics))
        partial["beat_spec_workload_passes"][0]["metrics"].pop("exact_result")
        with self.assertRaises(self.adapter.VerifierFailure):
            self.run_mode("train", (50.0, 50.0, "probe", partial))
        queries = {q: probe(self.qm, q) for q in EXPECTED[:20]}
        queries[EXPECTED[0]]["passed"] = False
        with self.assertRaises(self.qm.MeasurementError):
            self.qm.summarize_queries(EXPECTED[:20], queries)

    def test_eval_exposes_no_solver_metric_and_keeps_private_verdict(self):
        res = self.run_mode("eval", self.native_result(EXPECTED))
        self.assertEqual(res["status"], "passed")
        self.assertEqual(set(res["metrics"]), {"score_unbounded"})
        private = res["private_metrics"]
        self.assertEqual(private["beats_reference"], 1)
        for name in ("exact_result", "median_latency"):
            self.assertEqual(private["beat_spec_progress"][name]["passed"], 39)
        self.assertIn("reference_score", private)
        self.assertIn("margin", private)

    def test_eval_fails_closed_on_one_bad_query(self):
        queries = {q: probe(self.qm, q) for q in EXPECTED}
        queries[EXPECTED[-1]] = probe(self.qm, EXPECTED[-1], correct=False)
        score, metrics = self.qm.summarize_queries(EXPECTED, queries)
        res = self.run_mode("eval", (score, score, "probe", metrics))
        self.assertEqual(set(res["metrics"]), {"score_unbounded"})
        self.assertEqual(res["private_metrics"]["beats_reference"], 0)

    def test_source_policy_is_a_content_policy_rejection(self):
        evaluator = load_judge_module("evaluator")
        source = (TASK / "repo" / EDITABLE).read_text()
        payload = self.make_payload(patch_of("import os\n" + source))
        res = self.run_mode("train", self.native_result(EXPECTED[:20]), payload=payload,
                            check_tree=evaluator.check_tree)
        self.assertEqual(res["raw_score"], 0.0)
        self.assertEqual(res["metrics"], {"score_unbounded": 0.0, "rejected": "content_policy"})
        # The starter module passes the policy.
        self.assertIsNone(evaluator.check_tree(str(TASK / "repo"), str(TASK / "repo"), []))

    def test_an_empty_patch_is_judged_and_another_file_is_out_of_scope(self):
        res = self.run_mode("train", self.native_result(EXPECTED[:20]),
                            payload=self.make_payload(b""))
        self.assertEqual(res["status"], "passed")
        self.assertNotIn("rejected", res["metrics"])
        other = "2phase_nsa/binary_plan/util.py"
        first = (TASK / "repo" / other).read_text().splitlines()[0]
        patch = (f"diff --git a/{other} b/{other}\n--- a/{other}\n+++ b/{other}\n"
                 f"@@ -1 +1,2 @@\n+# out of scope\n {first}\n").encode()
        res = self.run_mode("train", self.native_result(EXPECTED[:20]),
                            payload=self.make_payload(patch))
        self.assertEqual(res["metrics"], {"score_unbounded": 0.0, "rejected": "out_of_scope"})


if __name__ == "__main__":
    unittest.main()
