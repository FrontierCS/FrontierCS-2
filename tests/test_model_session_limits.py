"""The approved campaign model limit must survive task imports and relocation."""

#
import deployment_test_guard  # noqa: F401  # isort: skip
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The approved campaign covers these existing tasks, including both copies
# where demo and pending overlap. Unrelated synthetic importer fixtures retain
# their authored limits.
CAMPAIGN_TASKS = frozenset(
    {
        "approximate_mips_query_stopping",
        "babble_anti_unification",
        "batch_local_counting",
        "bbq_packet_scheduling",
        "bounded_evm_path_prediction",
        "dbscan_gpu_kernel_optimization",
        "detok_latent_denoising",
        "diffusion_policy_cpu",
        "dynamic_tanh",
        "egraph_invariant_restoration",
        "evm_state_checkpoints",
        "graph_mining_method_selection",
        "hypergraph_core_numbers",
        "ivf_pq_gpu_kernel_optimization",
        "join_tree_rewriting",
        "kmeans_gpu_kernel_optimization",
        "knn_gpu_kernel_optimization",
        "lwe_structured_recovery",
        "metaopt_combined_te_gap_search",
        "mujoco_playground_cpu",
        "nano_world_model",
        "parallel_dbscan",
        "parallel_radix_sort",
        "parbutterfly",
        "persistent_treap_update_engine",
        "polar_express",
        "progent_privilege_control",
        "ragen_multi_turn_rl",
        "representation_frechet_loss",
        "residual_state_advancement",
        "shockwave_dynamic_gpu_allocation",
        "sparse_video_attention_policy",
    }
)


class ModelSessionLimitTests(unittest.TestCase):
    def test_task_policies_share_the_approved_limit(self):
        policies = [
            p
            for p in (ROOT / "tasks").glob("**/policy.json")
            if p.parent.name in CAMPAIGN_TASKS
        ]
        self.assertEqual({p.parent.name for p in policies}, CAMPAIGN_TASKS)
        by_task = {}
        for path in policies:
            policy = json.loads(path.read_text())
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertEqual(policy["network"]["audit"]["max_model_sessions"], 1024)
                self.assertEqual(policy["quota"]["wall_time_seconds"], 18000)
                self.assertLessEqual(
                    policy["network"]["audit"]["max_bundle_bytes"], 32 * 1024**3
                )
                audit = dict(policy["network"]["audit"])
                # Published releases may raise these operational floors while
                # pending authoring copies retain their historical capacities.
                audit["max_bundle_bytes"] = max(audit["max_bundle_bytes"], 32 * 1024**3)
                audit["max_model_request_bytes"] = max(
                    audit["max_model_request_bytes"], 64 * 1024**2
                )
                if path.parent.name in by_task:
                    self.assertEqual(audit, by_task[path.parent.name])
                by_task[path.parent.name] = audit

    def test_agent_budget_examples_do_not_reintroduce_the_old_limit(self):
        for path in (ROOT / "tasks").glob("**/agent-base/frontier/task/bin/README.md"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                text = path.read_text()
                self.assertNotRegex(text, r'"max_model_sessions"\s*:\s*(256|992)\b')
                self.assertNotRegex(
                    text,
                    r"(allows 256 model sessions|Network model limits: 256 requests)",
                )

    def test_published_definitions_use_the_same_limit(self):
        definitions = json.loads((ROOT / "config/task-definitions.json").read_text())
        for name, definition in definitions.items():
            if name not in CAMPAIGN_TASKS:
                continue
            with self.subTest(task=name):
                self.assertEqual(
                    definition["network"]["audit"]["max_model_sessions"], 1024
                )


if __name__ == "__main__":
    unittest.main()
