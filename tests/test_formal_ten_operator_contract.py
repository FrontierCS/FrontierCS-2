"""Keep explicit operator requirements across future native task imports."""

# Install isolation before runtime imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_task_definition as renderer
import task_manifest

TASKS = (
    "batch_local_counting",
    "bounded_evm_path_prediction",
    "evm_state_checkpoints",
    "graph_mining_method_selection",
    "join_tree_rewriting",
    "sparse_video_attention_policy",
    "bbq_packet_scheduling",
    "babble_anti_unification",
    "shockwave_dynamic_gpu_allocation",
)


class FormalTenOperatorContract(unittest.TestCase):
    def test_budget_and_selected_final_contract(self):
        for name in TASKS:
            with self.subTest(task=name):
                task = renderer.problem_directory(renderer.TASK_ROOT / "problems", name)
                # Either definition format: policy.json, or task.toml over the shared defaults.
                policy = task_manifest.load_policy(task)
                self.assertEqual(policy["quota"]["wall_time_seconds"], 18000)
                self.assertEqual(policy["network"]["audit"]["max_model_sessions"], 1024)
                self.assertGreaterEqual(
                    policy["network"]["audit"]["max_bundle_bytes"], 32 * 1024**3
                )
                self.assertGreaterEqual(
                    policy["network"]["audit"]["max_model_request_bytes"], 64 * 1024**2
                )
                paths = [
                    task / "agent-base/frontier/task/README.md",
                    task / "agent-base/frontier/task/bin/README.md",
                ]
                native = task_manifest.is_native(task)
                # A task.toml task's Agent README is generated: its own opening plus the shared
                # platform text, which states the save rules in its own words.
                text = " ".join(
                    (task_manifest.agent_readme(task) if native
                     else " ".join(p.read_text() for p in paths if p.is_file())).split()
                )
                if name == "bounded_evm_path_prediction":
                    # The saved selection is the final answer, kept up to date; a task.toml
                    # task states it in the shared store-final text.
                    self.assertIn(
                        "Save your chosen answer early and update it with" if native
                        else "You must save the final answer you select",
                        text,
                    )
                    self.assertNotIn("save your first usable chosen", text.lower())
                if native:
                    self.assertIn("Save your chosen answer early", text)
                else:
                    self.assertIn("Saving has no save-count quota.", text)
                self.assertIn("store-final", text)
                self.assertNotIn("100-update checkpoint/store-final cap", text)

    def test_published_task_excludes_development_residue(self):
        for name in TASKS:
            task = renderer.problem_directory(renderer.TASK_ROOT / "problems", name)
            forbidden = [task / "tests"]
            for directory in (task, task / "judge", task / "verifier"):
                forbidden.extend(
                    directory / filename
                    for filename in (
                        "DESIGN.md",
                        "NOTES.md",
                        "SUMMARY.md",
                        "REVIEW_DOSSIER.md",
                    )
                )
                forbidden.extend(directory.glob("test_*.py"))
            for path in forbidden:
                with self.subTest(task=name, path=str(path.relative_to(task))):
                    self.assertFalse(
                        path.exists(), "Keep maintenance files outside published tasks"
                    )


if __name__ == "__main__":
    unittest.main()
