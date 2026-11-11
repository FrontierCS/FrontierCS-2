"""Protect current Polygon rules when importing the disambiguation repair."""

import deployment_test_guard  # noqa: F401

import importlib.util
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/problems/Programming Language/polygon"
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402

# The judge as the Verifier image holds it (judge/ with the starter at pristine/).
JUDGE = Path(tempfile.mkdtemp(prefix="polygon-v4-judge-")) / "task"
shutil.copytree(TASK / "judge", JUDGE)
shutil.copytree(TASK / "repo", JUDGE / "pristine")
sys.path.insert(
    0, str(task_manifest.SHARED / "lib")
)  # isolate, as on the sandbox image
sys.path.insert(
    0, str(task_manifest.SHARED / "verifier-gpu")
)  # the control image's adapter
sys.path.insert(0, str(JUDGE))
spec = importlib.util.spec_from_file_location(
    "polygon_v4_evaluator", JUDGE / "evaluator.py"
)
evaluator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)
import isolate  # noqa: E402
import search_harness  # noqa: E402

# A stand-in for the fixed adapter's search entry (frontier_runner._search_main), uploaded as the
# sandbox's frontier_runner.py: it starts a long-lived child, reports it on stderr, sends the
# start reading and then (MODE "result") a result line, or (MODE "hang") nothing.
FAKE_ADAPTER = """
import json, os, subprocess, sys, time
MODE = {mode!r}
def _search_main(start, result):
    sys.stdin.read()
    p = subprocess.Popen(["sleep", "60"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"child={{p.pid}}", file=sys.stderr, flush=True)
    os.write(start, f"{{time.monotonic()!r}}\\n".encode())
    time.sleep({seconds})
    if MODE == "result":
        os.write(result, json.dumps({{"decided": True, "eq": True, "iterations": 1, "backtracks": 0}}).encode() + b"\\n")
        os._exit(0)
    if MODE.startswith("bytes="):  # a result line of exactly that many bytes, then the newline
        line = json.dumps({{"decided": True, "eq": True, "pad": ""}}).encode()
        line = line[:-2] + b"x" * (int(MODE[6:]) - len(line)) + b'"}}'
        data = line + b"\\n"
        while data:
            data = data[os.write(result, data):]
        os._exit(0)
    time.sleep(60)
"""


def _sandbox(directory: Path, mode: str, seconds: float = 0.0) -> Path:
    """A search sandbox's workdir with the fake adapter and one tree."""
    root = directory / "sandbox"
    (root / "trees" / evaluator.CANDIDATE_SIDE).mkdir(parents=True)
    (root / "trees" / evaluator.REFERENCE_SIDE).mkdir(parents=True)
    shutil.copy(JUDGE / "search_child.py", root / "search_child.py")
    (root / evaluator.RUNNER).write_text(
        FAKE_ADAPTER.format(mode=mode, seconds=seconds)
    )
    return root


def _entry(side: str, budget: int, limit: int) -> dict:
    return {
        "index": 0,
        "side": side,
        "budget_seconds": budget,
        "setup_seconds": 30.0,
        "limit_seconds": limit,
        "result_max_bytes": 128 * 1024,
        "payload": {},
    }


class PolygonV4ReconciliationTests(unittest.TestCase):
    @unittest.skipUnless(isolate.available(), "needs an isolate backend")
    def test_search_descendants_are_stopped_after_success_and_timeout(self):
        for mode in ("result", "hang"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = _sandbox(Path(directory), mode)
                out = Path(directory) / "out"
                out.mkdir()
                search_harness.run_entry(
                    root, _entry(evaluator.CANDIDATE_SIDE, 1, 2), out, 0
                )
                record = json.loads((out / "0.json").read_text())
                self.assertEqual(record["decided"], mode == "result")
                pid = int(record["stderr_tail"].split("child=")[1].split()[0])
                for _ in range(50):
                    status = Path(f"/proc/{pid}/stat")
                    if not status.exists() or status.read_text().split()[2] == "Z":
                        break
                    time.sleep(0.01)
                else:
                    self.fail(
                        "Search descendant remained active after the instance ended"
                    )

    @unittest.skipUnless(isolate.available(), "needs an isolate backend")
    def test_the_result_line_is_capped_exactly_and_never_truncated(self):
        cap = 128 * 1024
        for size, kept in ((cap, True), (cap + 1, False)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                root = _sandbox(Path(directory), f"bytes={size}")
                out = Path(directory) / "out"
                out.mkdir()
                search_harness.run_entry(
                    root, _entry(evaluator.CANDIDATE_SIDE, 5, 14), out, 0
                )
                record = json.loads((out / "0.json").read_text())
                self.assertEqual(
                    (record["decided"], record["result_over_cap"]), (kept, not kept)
                )
                if kept:
                    self.assertEqual(len((out / "0.message").read_bytes()), cap)
                    self.assertIsNone(record["reason"])
                else:
                    self.assertFalse((out / "0.message").exists())
                    self.assertEqual(
                        record["reason"],
                        f"search result exceeds the {cap}-byte output cap",
                    )

    def test_both_roles_preserve_hard_pairs_and_add_disambiguation(self):
        suites = []
        for role in ("dev", "final"):
            with mock.patch.dict(
                "os.environ",
                {"FRONTIER_SUBMISSION_ROLE": role, "FRONTIER_FAST_TEST": "0"},
            ):
                rows = evaluator._load_suite()
            self.assertEqual(len(rows), 21)
            self.assertEqual(sum(row.kind == "disambiguation" for row in rows), 3)
            self.assertEqual(
                sum(row.stratum == evaluator.HARD_STRATUM for row in rows), 2
            )
            self.assertTrue(
                all(
                    row.timeout_seconds == 20
                    for row in rows
                    if row.stratum == evaluator.HARD_STRATUM
                )
            )
            suites.append(
                {
                    (row.payload["problem_id"], row.payload["qid"])
                    for row in rows
                    if row.kind == "pair"
                }
            )
        self.assertFalse(suites[0] & suites[1])

    def test_hard_timing_remains_strict_and_score_excludes_hard_cases(self):
        with mock.patch.dict(
            "os.environ", {"FRONTIER_SUBMISSION_ROLE": "dev", "FRONTIER_FAST_TEST": "0"}
        ):
            workloads = evaluator._load_suite()
        hard = next(row for row in workloads if row.stratum == evaluator.HARD_STRATUM)
        ref = evaluator.Observation(True, 4.0, 1, 0, hard.timeout_seconds)
        for value, expected in ((4.0, False), (2.0, False), (1.99, True)):
            candidate = evaluator.Observation(True, value, 1, 0, hard.timeout_seconds)
            result = evaluator._comparison_decisions((hard,), [candidate], [ref])
            self.assertEqual(result[hard.identifier]["wall_time_seconds"], expected)
        rows = [
            evaluator.Observation(True, 0.1, 1, 0, row.timeout_seconds)
            for row in workloads
        ]
        base = evaluator.score_observations(evaluator._original(workloads, rows))
        for index, row in enumerate(workloads):
            if row.stratum == evaluator.HARD_STRATUM:
                rows[index] = evaluator.Observation(False, 20.0, 0, 0, 20)
        self.assertEqual(
            base, evaluator.score_observations(evaluator._original(workloads, rows))
        )

    def test_reference_failure_never_becomes_candidate_zero(self):
        with (
            mock.patch.dict("os.environ", {"FRONTIER_FAST_TEST": "1"}),
            mock.patch.object(evaluator, "_build"),
            mock.patch.object(
                evaluator,
                "_measure_pair",
                return_value=([], [evaluator.Observation(False, 2.0, 0, 0, 2, False)]),
            ),
        ):
            with self.assertRaises(evaluator.ReferenceFailure):
                evaluator.evaluate(str(JUDGE / "pristine"))

    def test_reference_failure_persists_private_case_diagnostics(self):
        workloads = (
            evaluator.Workload("w", "disambiguation", 2, {"kind": "disambiguation"}),
        )
        base = Path(tempfile.mkdtemp())
        out = base / "out"
        out.mkdir()
        for number, (side, reason, tail) in enumerate(
            (
                (
                    evaluator.REFERENCE_SIDE,
                    "search process exited without a result",
                    "forced reference diagnostic",
                ),
                (evaluator.CANDIDATE_SIDE, None, "candidate noise"),
            )
        ):
            (out / f"{number}.json").write_text(
                json.dumps(
                    {
                        "index": 0,
                        "side": side,
                        "decided": reason is None,
                        "seconds": 0.1 if reason is None else None,
                        "reason": reason,
                        "returncode": 3 if reason else 0,
                        "limit": None,
                        "backend": "landlock",
                        "wall_seconds": 1.0,
                        "stderr_tail": tail,
                        "result_over_cap": False,
                    }
                )
            )
            if reason is None:
                (out / f"{number}.message").write_text(
                    json.dumps({"decided": True, "eq": True})
                )
        result = mock.Mock(steps=[mock.Mock(exit_code=0)], outputs=base, logs=[b""])
        # The fixed adapter's helpers without its validation imports (no toolchain here).
        spec = importlib.util.spec_from_file_location(
            "polygon_v4_adapter", JUDGE / "pristine" / evaluator.RUNNER
        )
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        with (
            mock.patch.dict("os.environ", {"FRONTIER_FAST_TEST": "1"}),
            mock.patch.object(evaluator, "_build"),
            mock.patch.object(
                evaluator, "_search_payload", lambda workload, budget: {}
            ),
            mock.patch.object(evaluator, "_search_run", lambda *args: None),
            mock.patch.object(evaluator, "_load_suite", lambda: workloads),
            mock.patch.object(evaluator, "_Validator", mock.MagicMock()),
            mock.patch.object(evaluator, "_adapter", lambda: adapter),
            mock.patch("fcs2_verifier_gpu.run_sandboxes", lambda runs, into: [result]),
        ):
            with self.assertRaises(evaluator.ReferenceFailure) as raised:
                evaluator.evaluate(str(JUDGE / "pristine"))
        shutil.rmtree(base, ignore_errors=True)
        evidence = raised.exception.private_execution_evidence
        reference = [
            row for row in evidence["cases"] if row["side"] == evaluator.REFERENCE_SIDE
        ]
        candidate = [
            row for row in evidence["cases"] if row["side"] == evaluator.CANDIDATE_SIDE
        ]
        self.assertEqual(len(reference), 1)
        self.assertEqual(reference[0]["returncode"], 3)
        self.assertFalse(reference[0]["valid"])
        self.assertIn("forced reference diagnostic", reference[0]["stderr_tail"])
        self.assertTrue(candidate[0]["valid"])
        self.assertNotIn("stderr_tail", candidate[0])
        self.assertNotIn("forced", str(raised.exception))

    def test_only_the_reference_gets_a_larger_budget(self):
        workload = evaluator.Workload("budget", "pair", 2, {"timeout_seconds": 2})
        self.assertEqual(evaluator._reference_budget_seconds(workload), 12)
        self.assertEqual(
            evaluator._reference_budget_seconds(
                evaluator.Workload("d100", "pair", 12, {})
            ),
            24,
        )
        self.assertEqual(
            evaluator._reference_budget_seconds(
                evaluator.Workload("hard", "pair", 20, {}, evaluator.HARD_STRATUM)
            ),
            40,
        )
        self.assertEqual(evaluator._limit_seconds(workload, 2), 14)
        self.assertEqual(evaluator._limit_seconds(workload, 12), 24)
        if not isolate.available():
            return
        with tempfile.TemporaryDirectory() as directory:
            root = _sandbox(Path(directory), "result", 2.3)
            out = Path(directory) / "out"
            out.mkdir()
            for number, (side, budget) in enumerate(
                ((evaluator.CANDIDATE_SIDE, 2), (evaluator.REFERENCE_SIDE, 12))
            ):
                search_harness.run_entry(
                    root,
                    _entry(side, budget, evaluator._limit_seconds(workload, budget)),
                    out,
                    number,
                )
            candidate, reference = (
                json.loads((out / f"{n}.json").read_text()) for n in (0, 1)
            )
        # 2.3 s exceeds the candidate's 2 s budget (undecided) but not the reference's 12 s.
        self.assertEqual((candidate["decided"], candidate["reason"]), (False, None))
        self.assertTrue(reference["decided"])
        self.assertGreaterEqual(reference["seconds"], 2.3)

    def test_current_operational_limits_and_hygiene(self):
        policy = task_manifest.load_policy(TASK)
        self.assertEqual(policy["quota"]["eval_timeout_seconds"], 1800)
        self.assertEqual(policy["quota"]["wall_time_seconds"], 18000)
        self.assertEqual(policy["network"]["audit"]["max_bundle_bytes"], 34359738368)
        self.assertEqual(policy["network"]["audit"]["max_model_sessions"], 1024)
        self.assertEqual(policy["metadata"]["runtime"]["submission_kind"], "patch")
        for name in ("DESIGN.md", "NOTES.md", "REVIEW_DOSSIER.md", "test_evaluator.py"):
            self.assertFalse((TASK / "judge" / name).exists())
