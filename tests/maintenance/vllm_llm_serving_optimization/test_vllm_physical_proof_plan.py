"""Offline assessment gates only; these tests never invoke Docker or prove OOM."""

import deployment_test_guard  # noqa: F401

import importlib.util
import unittest

from vllm_task import REPO_ROOT  # noqa: E402

SCRIPT = (
    REPO_ROOT
    / "docs/task-authoring/vllm_llm_serving_optimization/tools/prove_physical_containment.py"
)
spec = importlib.util.spec_from_file_location("vllm_physical_proof", SCRIPT)
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


class PhysicalProofAssessmentTests(unittest.TestCase):
    def assess(self, case="oom", record=None, **changes):
        args = dict(
            before={"memory.events.local": {"oom_kill": 0}},
            after={"memory.events.local": {"oom_kill": 1}},
            runtime={"OOMKilled": True, "ExitCode": 137},
            elapsed=4,
            cleanup={"container_absent": True, "cgroup_empty_or_removed": True},
            tolerance=1,
        )
        args.update(changes)
        return proof.assess(
            case, record or {"status": "runtime_error", "cause": "child_exit"}, **args
        )

    def test_exit137_without_physical_local_oom_counter_never_passes(self):
        self.assertTrue(self.assess(before=None))
        self.assertTrue(self.assess(after={"memory.events.local": {"oom_kill": 0}}))
        self.assertTrue(self.assess(runtime={"OOMKilled": False, "ExitCode": 137}))
        self.assertFalse(self.assess())

    def test_watchdog_and_cleanup_require_the_actual_source_terminal(self):
        self.assertTrue(self.assess(case="watchdog"))
        self.assertFalse(
            self.assess(
                case="watchdog", record={"status": "timeout", "cause": "deadline"}
            )
        )
        self.assertTrue(self.assess(elapsed=27))
        self.assertTrue(
            self.assess(
                cleanup={"container_absent": False, "cgroup_empty_or_removed": True}
            )
        )
        self.assertTrue(
            self.assess(
                cleanup={"container_absent": True, "cgroup_empty_or_removed": False}
            )
        )

    def test_healthy_sibling_requires_complete_valid_wire_result(self):
        self.assertTrue(self.assess(case="healthy"))
        self.assertFalse(
            self.assess(
                case="healthy",
                record={"status": "complete", "result": {"healthy": True}},
            )
        )

    def test_oom_source_silent_dribble_has_real_length_prefix(self):
        self.assertNotIn("print", proof.child_program("oom"))
        self.assertNotIn("os.write", proof.child_program("oom"))
        self.assertIn('struct.pack("!I",65000)', proof.child_program("dribble"))


if __name__ == "__main__":
    unittest.main()
