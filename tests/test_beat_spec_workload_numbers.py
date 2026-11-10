"""BBQ and Batch Local Counting keep each workload's measured numbers beside its decisions.

The numbers are the quantities the decisions compare. They change no decision and
never reach solver feedback.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import json
import statistics
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "tasks/problems"
sys.path.insert(0, str(ROOT.parents[1] / "scripts"))
import task_manifest  # noqa: E402
BBQ = ROOT / "Network/bbq_packet_scheduling"
BLC = ROOT / "Database/batch_local_counting"


def _load(path, name, extra_path=None):
    # Task-local helper names (submission_patch, state_guard) must not stay in
    # sys.modules or sys.path for the next task's evaluator.
    saved_path = list(sys.path)
    with patch.dict(sys.modules), patch.object(sys, "dont_write_bytecode", True):
        if extra_path is not None:
            sys.path.insert(0, str(extra_path))
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = saved_path
    return module


def _feedback(task):
    # A task.toml task uses the shared train feedback.
    path = (task_manifest.SHARED / "verifier/beat_spec_feedback.py" if task_manifest.is_native(task)
            else task / "verifier/beat_spec_feedback.py")
    return _load(path, "feedback_" + task.name)


def _without_numbers(metrics):
    rows = [{"workload": row["workload"], "metrics": row["metrics"]}
            for row in metrics["beat_spec_workload_passes"]]
    return dict(metrics, beat_spec_workload_passes=rows)


def _bbq_decisions(candidate, reference, cases):
    """The decision rule as it stood before the numbers were added."""
    rows = []
    for case in cases:
        c, r = candidate.get(case["id"], {}), reference[case["id"]]
        valid = c.get("valid", False) and c.get("count") == r["count"] and r["count"] > 0
        rows.append({"workload": case["id"], "metrics": {
            "throughput_ops_per_second": bool(valid and
                c["count"] * r["cycles"] * r["period_ps"] > r["count"] * c["cycles"] * c["period_ps"]),
            "latency_seconds": bool(valid and
                c["latency_cycles"] * c["period_ps"] <= r["latency_cycles"] * r["period_ps"])}})
    return rows


def _blc_decisions(naive, reference, candidate, names):
    rows = []
    for name in names:
        ref_times, cand_times = reference["case_times"][name], candidate["case_times"][name]
        correct = candidate["case_digests"][name] == reference["case_digests"][name]
        faster = all(c < r for c, r in zip(cand_times, ref_times, strict=True))
        rows.append({"workload": name, "metrics": {
            "exact_counts": bool(correct), "batch_latency_seconds": bool(correct and faster)}})
    return rows


class BBQWorkloadNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The judge as the Verifier image holds it (judge/ + generated config.yaml + pristine/).
        bundle = task_manifest.judge_bundle(BBQ)
        cls.evaluator = _load(bundle / "evaluator.py", "bbq_numbers_evaluator", bundle)
        cls.cases = cls.evaluator.DEV + cls.evaluator.HELD
        # The archived beat spec (the judge no longer ships it); train feedback is task.toml [feedback].
        cls.spec = json.loads((ROOT.parents[1] / "docs/task-authoring/bbq_packet_scheduling/provenance/beat_spec.json").read_text())
        cls.visible = list(task_manifest.read_task(BBQ)["feedback"])

    def rows(self, latency, period, span=408, count=288):
        # Built by the runner's own physical(), as the evaluator receives them.
        measured = dict(valid=True, count=count, cycles=span + latency, offered_span=span,
                        latency_cycles=latency, throughput_ops_per_cycle=count / (span + latency))
        clocked = dict(period_ps=period, lut_count=1, register_count=1, memory_bits=1,
                       memory_inventory={})
        row = self.evaluator.runner.physical(measured, clocked)
        return {case["id"]: dict(row) for case in self.cases}

    def scenarios(self):
        reference = self.rows(14, 2800)
        invalid = self.rows(13, 2800)
        invalid[self.cases[0]["id"]] = {"valid": False}
        short = self.rows(13, 2800)
        short[self.cases[1]["id"]] = dict(short[self.cases[1]["id"]], count=287)
        for candidate in (self.rows(14, 2800), self.rows(6, 4800), self.rows(13, 2800),
                          self.rows(14, 2300), self.rows(17, 2300), self.rows(18, 2300),
                          invalid, short, {}):
            yield candidate, reference

    def test_decisions_are_unchanged(self):
        for candidate, reference in self.scenarios():
            metrics = self.evaluator.summarize(candidate, reference, self.cases)[3]
            rows = metrics["beat_spec_workload_passes"]
            self.assertEqual([{"workload": row["workload"], "metrics": row["metrics"]} for row in rows],
                             _bbq_decisions(candidate, reference, self.cases))
            self.assertEqual(metrics["beats_reference"],
                             int(all(v for row in rows for v in row["metrics"].values())))


class BatchLocalCountingWorkloadNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = _load(BLC / "judge/evaluator.py", "blc_numbers_evaluator")
        cls.names = cls.evaluator.CASE_NAMES
        # A task.toml task keeps its beat spec as provenance only.
        spec = BLC / "judge/beat_spec.json"
        if not spec.is_file():
            spec = ROOT.parents[1] / "docs/task-authoring/batch_local_counting/provenance/beat_spec.json"
        cls.spec_path = spec
        cls.spec = json.loads(spec.read_text())

    def side(self, times, digest="d"):
        return {"case_times": {name: list(times) for name in self.names},
                "case_digests": {name: digest for name in self.names}}

    def scenarios(self):
        naive = self.side([9.0, 9.1, 9.2, 9.3, 9.4])
        reference = self.side([3.5, 3.6, 3.7, 3.4, 3.55])
        one_slow = self.side([1.0, 1.1, 3.8, 1.2, 1.3])
        mixed = self.side([0.2, 0.3, 0.25, 0.21, 0.22])
        mixed["case_times"][self.names[2]] = [3.5, 3.6, 3.7, 3.4, 3.55]
        wrong = self.side([0.2, 0.3, 0.25, 0.21, 0.22])
        wrong["case_digests"][self.names[4]] = "other"
        for candidate in (self.side([0.2, 0.3, 0.25, 0.21, 0.22]), one_slow, mixed, wrong, reference):
            yield naive, reference, candidate

    def test_decisions_are_unchanged(self):
        for naive, reference, candidate in self.scenarios():
            rows = self.evaluator._workload_decisions(naive, reference, candidate)
            self.assertEqual([{"workload": row["workload"], "metrics": row["metrics"]} for row in rows],
                             _blc_decisions(naive, reference, candidate, self.names))
            metrics = self.evaluator._metrics(50.0, 20.0, rows)
            self.assertIs(metrics["beat_spec_workload_passes"], rows)
            self.assertEqual(metrics["beats_reference"],
                             int(all(v for row in rows for v in row["metrics"].values())))

if __name__ == "__main__":
    unittest.main()
