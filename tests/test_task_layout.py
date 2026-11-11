from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import contextlib
import copy
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_task_definition as renderer
import task_layers  # noqa: E402
import task_manifest
import validate_task_layout as layout


class TaskLayoutTests(unittest.TestCase):
    def test_every_catalog_task_has_a_complete_native_source_lock(self):
        names = task_layers.load_definitions()
        for name in names:
            with self.subTest(task=name):
                self.assertIn(
                    layout.validate_task(renderer.task_directory(name)),
                    ("patch-workspace", "direct-payload", "patch-starter"),
                )

    def test_quota_drift_and_unlocked_source_are_rejected(self):
        # A real policy.json task with a source lock; task.toml tasks keep neither.
        def valid(path):
            try:
                return bool(layout.validate_task(path))
            except layout.LayoutFailure:
                return False

        source = next(
            (
                path
                for path in sorted(layout.DEFAULT_PROBLEMS_ROOT.glob("*/*"))
                if (path / "policy.json").is_file()
                and (path / "upstream.lock.json").is_file()
                and (path / "agent-base/frontier/task").is_dir()
                and valid(path)
            ),
            None,
        )
        if source is None:
            self.skipTest("no policy.json task with a lock remains")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / source.name
            shutil.copytree(source, root)
            path = root / "policy.json"
            lock_path = root / "upstream.lock.json"
            original, original_lock = path.read_bytes(), lock_path.read_bytes()
            policy = json.loads(original)
            policy["quota"]["max_train_calls"] += 1
            path.write_text(json.dumps(policy))
            # Keep the lock consistent, so the quota rule is what rejects the drift.
            lock = json.loads(original_lock)
            for record in lock["files"]:
                if record["path"] == "policy.json":
                    record.update(bytes=path.stat().st_size, sha256=layout.digest(path))
            lock_path.write_text(json.dumps(lock))
            with self.assertRaisesRegex(layout.LayoutFailure, "task-authored"):
                layout.validate_task(root)
            path.write_bytes(original)
            lock_path.write_bytes(original_lock)
            (root / "agent-base/frontier/task/unlocked.py").write_text(
                "unexpected source"
            )
            with self.assertRaisesRegex(layout.LayoutFailure, "complete task tree"):
                layout.validate_task(root)

    def test_generated_bytecode_is_rejected_even_if_included_in_source_lock(self):
        for name in (
            "judge/__pycache__/evaluator.cpython-314.pyc",
            "judge/evaluator.pyc",
            "judge/.pytest_cache/v/cache/nodeids",
        ):
            with self.subTest(path=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_bytes(b"generated cache")
                record = {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": layout.digest(path),
                }
                (root / "upstream.lock.json").write_text(
                    json.dumps(
                        {
                            "problem_id": root.name,
                            "file_count": 1,
                            "files": [record],
                        }
                    )
                )
                with self.assertRaisesRegex(
                    layout.LayoutFailure, "generated Python cache"
                ):
                    layout.check_inventory(root)


class SshBudgetGateTests(unittest.TestCase):
    def setUp(self):
        # svg is a task.toml task: its effective policy is task.toml over the shared defaults.
        self.policy = task_manifest.load_policy(
            renderer.problem_directory(
                layout.DEFAULT_PROBLEMS_ROOT, "sparse_video_attention_policy"
            )
        )

    def test_svg_retains_its_authored_bundle_without_multiplying_stream_ceilings(self):
        audit = self.policy["network"]["audit"]
        self.assertEqual(self.policy["quota"]["max_train_calls"], 25)
        self.assertEqual(
            audit["ssh_session_budgets"],
            {"agent": 1024, "verifier_train": 50, "verifier_eval": 2},
        )
        self.assertEqual(audit["max_ssh_bytes_per_direction"], 512 * 1024**2)
        self.assertEqual(audit["max_model_sessions"], 1024)
        self.assertEqual(audit["max_bundle_bytes"], 32 * 1024**3)
        self.assertLess(
            audit["max_bundle_bytes"],
            layout.partitioned_audit_bundle_bytes(self.policy),
        )
        mib = 1024**2
        floor = (
            self.policy["verifier"]["max_evidence_bytes"]
            + (24 + 4096 + 4) * mib
            + audit["max_model_request_bytes"]
            + audit["max_model_response_bytes"]
            + 2 * audit["max_ssh_bytes_per_direction"]
            + 24 * 1024
        )
        self.assertEqual(layout.native_audit_minimum_bytes(self.policy), floor)
        layout.check_ssh_session_budgets(self.policy)
        audit["max_bundle_bytes"] = 200 * mib
        with self.assertRaisesRegex(layout.LayoutFailure, "native exchange"):
            layout.check_ssh_session_budgets(self.policy)

    def test_independent_8gib_aggregate_accepts_large_session_ceilings(self):
        policy = copy.deepcopy(self.policy)
        audit = policy["network"]["audit"]
        audit.update(
            max_bundle_bytes=8 * 1024**3,
            max_model_sessions=256,
            max_ssh_sessions=1076,
            max_model_request_bytes=1024**2,
            max_model_response_bytes=8 * 1024**2,
            max_ssh_bytes_per_direction=64 * 1024**2,
            ssh_session_budgets={
                "agent": 1024,
                "verifier_train": 50,
                "verifier_eval": 2,
            },
        )
        policy["quota"]["max_train_calls"] = 25
        policy["verifier"]["max_evidence_bytes"] = 4 * 1024**2
        policy["submission"]["max_bytes"] = 2 * 1024**2
        self.assertEqual(layout.native_audit_minimum_bytes(policy), 4472201216)
        layout.check_ssh_session_budgets(policy)
        # Actual max_bundle_bytes still binds all traffic; no permission or
        # source quota is rewritten when session ceilings exceed that total.
        self.assertEqual(audit["max_bundle_bytes"], 8 * 1024**3)
        audit["max_ssh_bytes_per_direction"] = 512 * 1024**2
        layout.check_ssh_session_budgets(policy)
        audit["max_ssh_bytes_per_direction"] += 1
        with self.assertRaisesRegex(layout.LayoutFailure, "512 MiB"):
            layout.check_ssh_session_budgets(policy)

    def test_native_floor_exact_boundary_and_gateway_allocation_switch(self):
        policy = copy.deepcopy(self.policy)
        audit = policy["network"]["audit"]
        audit["max_bundle_bytes"] = 1024**3
        floor = layout.native_audit_minimum_bytes(policy)
        audit["max_bundle_bytes"] = floor
        layout.check_ssh_session_budgets(policy)
        audit["max_bundle_bytes"] -= 1
        with self.assertRaisesRegex(layout.LayoutFailure, "native exchange"):
            layout.check_ssh_session_budgets(policy)
        audit["max_bundle_bytes"] = 8 * 1024**3 - 1
        legacy_floor = layout.native_audit_minimum_bytes(policy)
        audit["max_bundle_bytes"] += 1
        self.assertEqual(
            layout.native_audit_minimum_bytes(policy) - legacy_floor,
            (4096 - 128) * 1024**2,
        )
        # CPU admission reserves no SSH channel, although its audit ceilings
        # remain configured and both model request/response caps still bind.
        policy["gpu"]["agent"]["max_instances"] = 0
        policy["gpu"]["verifier"]["max_gpu_instances"] = 0
        self.assertEqual(
            layout.native_audit_minimum_bytes(policy),
            legacy_floor
            + (4096 - 128) * 1024**2
            - 2 * audit["max_ssh_bytes_per_direction"]
            - 24 * 1024,
        )

    def test_current_runtime_32gib_cap_preserves_configured_budget(self):
        policy = copy.deepcopy(self.policy)
        audit = policy["network"]["audit"]
        audit["max_bundle_bytes"] = 32 * 1024**3
        before = copy.deepcopy(policy)
        layout.check_ssh_session_budgets(policy)
        self.assertEqual(policy, before)
        audit["max_bundle_bytes"] += 1
        with self.assertRaisesRegex(layout.LayoutFailure, "32 GiB"):
            layout.check_ssh_session_budgets(policy)

    def test_missing_unfunded_or_miscounted_pools_are_rejected(self):
        for pools in (
            None,
            {},
            {"agent": 32, "verifier_train": 12},
            {"agent": True, "verifier_train": 12, "verifier_eval": 4},
            {"agent": -1, "verifier_train": 12, "verifier_eval": 4},
            {"agent": 32, "verifier_train": 12, "verifier_eval": 5},
        ):
            with self.subTest(pools=pools):
                policy = copy.deepcopy(self.policy)
                policy["network"]["audit"]["ssh_session_budgets"] = pools
                with self.assertRaises(layout.LayoutFailure):
                    layout.check_ssh_session_budgets(policy)
        for field in ("agent", "verifier_train", "verifier_eval"):
            with self.subTest(unfunded=field):
                policy = copy.deepcopy(self.policy)
                audit = policy["network"]["audit"]
                audit["ssh_session_budgets"][field] = 0
                audit["max_ssh_sessions"] = sum(audit["ssh_session_budgets"].values())
                with self.assertRaisesRegex(layout.LayoutFailure, "SSH pool"):
                    layout.check_ssh_session_budgets(policy)

    def test_native_svg_sessions_and_independent_per_kind_ceilings(self):
        policy = copy.deepcopy(self.policy)
        audit = policy["network"]["audit"]
        audit["max_model_sessions"] = 1024
        audit["max_ssh_sessions"] = 1076
        audit["ssh_session_budgets"] = {
            "agent": 1056,
            "verifier_train": 12,
            "verifier_eval": 8,
        }
        audit["max_model_request_bytes"] = 1024**2
        audit["max_model_response_bytes"] = 1024**2
        audit["max_ssh_bytes_per_direction"] = 1024**2
        audit["max_bundle_bytes"] = 10 * 1024**3
        layout.check_ssh_session_budgets(policy)
        audit["max_model_sessions"] = 2049
        with self.assertRaisesRegex(
            layout.LayoutFailure, "invalid audit max_model_sessions"
        ):
            layout.check_ssh_session_budgets(policy)

    def test_v3_requires_separated_accounting_and_stream_caps_still_apply(self):
        policy = copy.deepcopy(self.policy)
        del policy["network"]["audit"]["ssh_session_budgets"]
        with self.assertRaisesRegex(layout.LayoutFailure, "requires partitioned"):
            layout.check_ssh_session_budgets(policy)
        policy["metadata"]["accelerator"]["agent_gpu_interface"] = (
            "frontier.gpu-helper/v2"
        )
        layout.check_ssh_session_budgets(
            policy
        )  # Historical shared V2 remains explicit.
        for field, value, message in (
            ("max_model_sessions", 2049, "invalid audit max_model_sessions"),
            ("max_ssh_bytes_per_direction", 512 * 1024**2 + 1, "512 MiB"),
            ("max_bundle_bytes", 32 * 1024**3 + 1, "32 GiB"),
        ):
            with self.subTest(field=field):
                policy = copy.deepcopy(self.policy)
                policy["network"]["audit"][field] = value
                with self.assertRaisesRegex(layout.LayoutFailure, message):
                    layout.check_ssh_session_budgets(policy)
        policy = copy.deepcopy(self.policy)
        policy["metadata"]["accelerator"]["agent_gpu_interface"] = (
            "frontier.gpu-helper/v2"
        )
        with self.assertRaisesRegex(
            layout.LayoutFailure, "requires Candidate helper V3"
        ):
            layout.check_ssh_session_budgets(policy)


class BeatSpecGateTests(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "unit": "One fixed graph and pattern batch.",
            "metrics": [
                {
                    "name": "latency_seconds",
                    "means": "Median elapsed time in seconds.",
                    "source": "The artifact's end-to-end batch execution.",
                    "better": "lower",
                    "clears_when": "candidate latency is strictly below reference latency",
                    "tolerance": "none",
                }
            ],
            "beats_reference": "All workloads must clear every metric.",
            "solver_feedback": [
                {
                    "name": "candidate_seconds",
                    "means": "Absolute candidate durations without workload IDs.",
                    "reveals_reference": "no — only candidate measurements are returned",
                }
            ],
            "withheld": ["reference_score", "margin", "beat_spec_workload_passes"],
            "statement_sentences": "Beat-reference requires lower latency on each workload.",
        }
        self.evaluation = {"solver_visible_metrics": ["candidate_seconds"]}

    def test_required_spec_cannot_be_grandfathered(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(layout.LayoutFailure, "beat_spec.json"):
                layout.check_beat_spec(Path(directory), self.evaluation)

    def test_metrics_and_feedback_have_exact_names_and_privacy_declarations(self):
        self.assertEqual(
            layout.check_beat_spec_schema(self.spec, self.evaluation),
            (["latency_seconds"], ["candidate_seconds"]),
        )
        mutations = [
            (lambda spec: spec.update(unit=""), "unit"),
            (lambda spec: spec.update(metrics=[]), "at least one metric"),
            (
                lambda spec: spec["metrics"].append(copy.deepcopy(spec["metrics"][0])),
                "duplicate",
            ),
            (lambda spec: spec["metrics"][0].update(better="sideways"), "direction"),
            (lambda spec: spec["metrics"][0].update(clears_when=""), "clears_when"),
            (
                lambda spec: spec["solver_feedback"][0].update(reveals_reference="no"),
                "privacy explanation",
            ),
            (
                lambda spec: spec["solver_feedback"][0].update(
                    name="reference_seconds"
                ),
                "private or reserved",
            ),
            (
                lambda spec: spec["solver_feedback"].append(
                    copy.deepcopy(spec["solver_feedback"][0])
                ),
                "duplicate",
            ),
            (
                lambda spec: spec["withheld"].append("candidate_seconds"),
                "withheld field",
            ),
        ]
        for mutate, expected in mutations:
            with self.subTest(error=expected):
                spec = copy.deepcopy(self.spec)
                mutate(spec)
                with self.assertRaisesRegex(layout.LayoutFailure, expected):
                    layout.check_beat_spec_schema(spec, self.evaluation)
        for declared in (
            None,
            "candidate_seconds",
            [],
            ["candidate_seconds", "margin"],
            ["candidate_seconds", "candidate_seconds"],
        ):
            with self.subTest(declared=declared):
                with self.assertRaisesRegex(layout.LayoutFailure, "exactly match"):
                    layout.check_beat_spec_schema(
                        self.spec, {"solver_visible_metrics": declared}
                    )

    def test_declared_aggregation_is_never_silently_defaulted(self):
        calibrated = {
            "kind": "score_majority_coverage",
            "minimum_score_margin": 0.1,
            "minimum_comparison_pass_fraction_exclusive": 0.5,
            "require_each_workload": True,
            "require_each_metric": True,
        }
        for aggregation in ({"kind": "conjunction"}, calibrated):
            with self.subTest(aggregation=aggregation):
                spec = dict(self.spec, beats_reference_aggregation=aggregation)
                layout.check_beat_spec_schema(spec, self.evaluation)
        invalid = [
            None,
            {},
            {"kind": "majority"},
            {"kind": "conjunction", "extra": True},
            {"kind": "score_majority_coverage"},
            dict(calibrated, extra=True),
            *[
                dict(calibrated, minimum_score_margin=value)
                for value in (-1, True, float("nan"), float("inf"))
            ],
            *[
                dict(calibrated, minimum_comparison_pass_fraction_exclusive=value)
                for value in (-0.1, 1, True, float("nan"), float("inf"))
            ],
            dict(calibrated, require_each_workload=1),
            dict(calibrated, require_each_metric="true"),
        ]
        for aggregation in invalid:
            with self.subTest(aggregation=aggregation):
                with self.assertRaises(layout.LayoutFailure):
                    layout.check_beat_spec_schema(
                        dict(self.spec, beats_reference_aggregation=aggregation),
                        self.evaluation,
                    )

    def test_only_referenced_local_helpers_establish_structural_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "evaluator.py"
            helper = root / "scoring.py"
            helper.write_text(
                'METRICS = ("latency_seconds",)\n'
                "def assemble(value):\n"
                '    return {"beat_spec_workload_passes": METRICS, "beats_reference": 1}\n'
            )
            evaluator.write_text(
                "from scoring import assemble\ndef evaluate(path):\n    return assemble(path)\n"
            )
            nodes = layout.referenced_local_evaluator_nodes(evaluator)
            strings = {
                n.value
                for n in nodes
                if isinstance(n, layout.ast.Constant) and isinstance(n.value, str)
            }
            self.assertTrue(
                {"latency_seconds", "beat_spec_workload_passes", "beats_reference"}
                <= strings
            )
            evaluator.write_text(
                "from scoring import assemble\ndef evaluate(path):\n    return 0\n"
            )
            nodes = layout.referenced_local_evaluator_nodes(evaluator)
            self.assertFalse(
                any(
                    isinstance(n, layout.ast.Constant) and n.value == "latency_seconds"
                    for n in nodes
                )
            )
            evaluator.write_text(
                "from scoring import assemble\ndef evaluate(path, assemble):\n    return assemble(path)\n"
            )
            nodes = layout.referenced_local_evaluator_nodes(evaluator)
            self.assertFalse(
                any(
                    isinstance(n, layout.ast.Constant) and n.value == "latency_seconds"
                    for n in nodes
                )
            )

    def test_referenced_helper_symlink_cannot_escape_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            judge = root / "judge"
            judge.mkdir()
            (root / "external.py").write_text(
                'def assemble(value): return "latency_seconds"\n'
            )
            (judge / "scoring.py").symlink_to(root / "external.py")
            evaluator = judge / "evaluator.py"
            evaluator.write_text(
                "from scoring import assemble\ndef evaluate(path):\n    return assemble(path)\n"
            )
            with self.assertRaisesRegex(
                layout.LayoutFailure, "regular local judge file"
            ):
                layout.referenced_local_evaluator_nodes(evaluator)

    def helper_contract_fixture(self, directory):
        root = Path(directory)
        (root / "judge").mkdir()
        (root / "verifier").mkdir()
        (root / "agent-base/frontier/task").mkdir(parents=True)
        (root / "agent-base/frontier/task/README.md").write_text(
            "beat_spec_progress passed failed total pass_percent fail_percent candidate_seconds\n"
        )
        (root / "judge/beat_spec.json").write_text(json.dumps(self.spec))
        # The legacy (policy.json) helper contract: a legacy task's own copy.
        legacy = next(
            (
                path / "verifier/beat_spec_feedback.py"
                for path in sorted(layout.DEFAULT_PROBLEMS_ROOT.glob("*/*"))
                if (path / "policy.json").is_file()
                and (path / "verifier/beat_spec_feedback.py").is_file()
            ),
            None,
        )
        if legacy is None:
            self.skipTest("no policy.json task with its own feedback helper remains")
        shutil.copyfile(legacy, root / "verifier/beat_spec_feedback.py")
        (root / "judge/scoring.py").write_text(
            'METRICS = ("latency_seconds",)\n'
            "def assemble(value):\n"
            '    return {"beat_spec_workload_passes": METRICS, "beats_reference": 1}\n'
        )
        (root / "judge/empty.py").write_text("def assemble(value): return 0\n")
        return root, root / "judge/evaluator.py"

    def test_rebound_or_nested_import_cannot_supply_missing_evaluator_fields(self):
        cases = (
            "from scoring import assemble\nassemble = lambda value: 0\n"
            "def evaluate(path): return assemble(path)\n",
            "from scoring import assemble\ndef evaluate(path):\n"
            "    from empty import assemble\n    return assemble(path)\n",
            "def evaluate(path):\n    def unused():\n"
            "        from scoring import assemble\n    return assemble(path)\n",
            "from scoring import assemble\nif True:\n    assemble = lambda value: 0\n"
            "def evaluate(path): return assemble(path)\n",
            "from scoring import assemble\nfrom empty import assemble\n"
            "def evaluate(path): return assemble(path)\n",
            "from scoring import assemble\ndef evaluate(path):\n"
            "    match (lambda value: 0):\n        case assemble:\n"
            "            return assemble(path)\n",
            "from scoring import assemble\ndef evaluate(path):\n"
            "    match [lambda value: 0]:\n        case [*assemble]:\n"
            "            return assemble[0](path)\n",
            "from scoring import assemble\ndef evaluate(path):\n"
            '    match {"f": lambda value: 0}:\n        case {**assemble}:\n'
            '            return assemble["f"](path)\n',
            "from scoring import assemble\ndef evaluate(path):\n"
            "    global assemble\n    assemble = lambda value: 0\n"
            "    return assemble(path)\n",
        )
        for code in cases:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                root, evaluator = self.helper_contract_fixture(directory)
                evaluator.write_text(code)
                with self.assertRaises(layout.LayoutFailure):
                    layout.check_beat_spec(root, self.evaluation)

    def test_actual_local_helper_keeps_feedback_and_native_audit_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root, evaluator = self.helper_contract_fixture(directory)
            evaluator.write_text(
                "from empty import assemble\ndef evaluate(path):\n"
                "    from scoring import assemble\n    return assemble(path)\n"
            )
            layout.check_beat_spec(root, self.evaluation)
            policy = task_manifest.load_policy(
                renderer.problem_directory(
                    layout.DEFAULT_PROBLEMS_ROOT, "sparse_video_attention_policy"
                )
            )
            policy["network"]["audit"]["max_bundle_bytes"] = 8 * 1024**3
            layout.check_ssh_session_budgets(policy)
            policy["network"]["audit"]["max_ssh_bytes_per_direction"] = 512 * 1024**2
            layout.check_ssh_session_budgets(policy)
            policy["network"]["audit"]["max_ssh_bytes_per_direction"] += 1
            with self.assertRaisesRegex(layout.LayoutFailure, "512 MiB"):
                layout.check_ssh_session_budgets(policy)
            (root / "verifier/beat_spec_feedback.py").write_text(
                "def public_metrics(values, **kwargs): return values\n"
            )
            with self.assertRaisesRegex(layout.LayoutFailure, "exposes private fields"):
                layout.check_beat_spec(root, self.evaluation)

    def test_comments_do_not_count_as_evaluator_row_emission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "judge").mkdir()
            (root / "judge/beat_spec.json").write_text(json.dumps(self.spec))
            evaluator = root / "judge/evaluator.py"
            evaluator.write_text(
                "# latency_seconds beat_spec_workload_passes\n"
                'def evaluate(path):\n    return {"beats_reference": 1}\n'
            )
            with self.assertRaisesRegex(
                layout.LayoutFailure, "does not name beat-spec metric"
            ):
                layout.check_beat_spec(root, self.evaluation)
            evaluator.write_text(
                'METRIC = "latency_seconds"\n# beat_spec_workload_passes\n'
                'def evaluate(path):\n    return {"beats_reference": 1}\n'
            )
            with self.assertRaisesRegex(
                layout.LayoutFailure, "does not emit private beat-spec field"
            ):
                layout.check_beat_spec(root, self.evaluation)


class RelaxationListingTests(unittest.TestCase):
    """scripts/validate_task_layout.py lists the layers that relax the boundary."""

    def relaxations(self, source: str) -> list[str]:
        with tempfile.TemporaryDirectory() as task:
            (Path(task) / "judge").mkdir()
            (Path(task) / "judge/layers.py").write_text(source)
            return layout.sandbox_relaxations(Path(task))

    def test_lists_relaxing_layers_only(self):
        self.assertEqual(
            self.relaxations("""
import isolate
class Tight(isolate.Layer):
    pass
class Ports(isolate.Layer):
    relaxes = ("tcp connect", "tcp bind")
"""),
            ["judge/layers.py:Ports relaxes tcp connect, tcp bind"],
        )

    def test_relaxes_must_be_literal(self):
        with self.assertRaisesRegex(Exception, "literal tuple"):
            self.relaxations(
                "from isolate import Layer\n"
                "class Ports(Layer):\n    relaxes = tuple(['tcp'])\n"
            )


if __name__ == "__main__":
    unittest.main()


class SharedBaseArgumentTests(unittest.TestCase):
    """Recipes take the shared bases by build argument (the build injects bases.json), so a base
    release moves every task at once; a literal reference, an undeclared argument, or a Verifier-
    GPU task without the shared adapter/verifier stages is refused."""

    CONTROL = (
        "ARG GPU_TOOLS\nARG VERIFIER_BASE\nFROM ${GPU_TOOLS} AS frontier_gpu_tools\n"
        "FROM ${VERIFIER_BASE}\n"
        "COPY --from=frontier_gpu_tools /opt/frontier/shared-gpu/fcs2_verifier_gpu.py /opt/frontier/\n"
        "ENV FRONTIER_VERIFIER_GPU_TYPE=A100-80GB FRONTIER_VERIFIER_GPU_COUNT=1\n"
    )
    GPU = (
        "ARG VERIFIER_BASE\nARG GPU_TOOLS\nFROM ${VERIFIER_BASE} AS frontier_verifier\n"
        "FROM ${GPU_TOOLS} AS frontier_gpu_tools\nFROM scratch\n"
        "COPY --from=frontier_verifier /opt/frontier/fcs2_verifier.py /opt/frontier/\n"
    )

    LEASE = "lease_seconds = { train = 60, eval = 60 }\n"
    COMMAND = '\n[test_sandboxes.shard]\naccess = "command"\n'

    def fixture(self, directory, control=None, gpu=None, settings=LEASE):
        root = Path(directory) / "gpu_task"
        (root / "docker").mkdir(parents=True)
        (root / "task.toml").write_text(
            "[gpu.verifier]\nmax_gpu_instances = 2\nmax_gpu_seconds = 60\n" + settings
        )
        (root / "docker/verifier.Dockerfile").write_text(control or self.CONTROL)
        (root / "docker/verifier-gpu.Dockerfile").write_text(gpu or self.GPU)
        return root

    def test_argument_recipes_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            layout.check_gpu_adapter_pins(self.fixture(directory))

    def test_a_literal_shared_reference_is_refused(self):
        literal = task_manifest.base_reference("verifier")
        for control in (
            self.CONTROL.replace("FROM ${VERIFIER_BASE}\n", f"FROM {literal}\n"),
            self.CONTROL.replace(
                "${GPU_TOOLS} AS", task_manifest.base_reference("gpu-tools") + " AS"
            ),
        ):
            with (
                self.subTest(control=control),
                tempfile.TemporaryDirectory() as directory,
            ):
                with self.assertRaisesRegex(layout.LayoutFailure, "by build argument"):
                    layout.check_gpu_adapter_pins(
                        self.fixture(directory, control=control)
                    )
        gpu = self.GPU.replace(
            "--from=frontier_verifier",
            "--from=docker.io/joyemang33/frontier-verifier-base@"
            + literal.partition("@")[2],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(layout.LayoutFailure, "by build argument"):
                layout.check_gpu_adapter_pins(self.fixture(directory, gpu=gpu))

    def test_an_undeclared_or_defaulted_argument_is_refused(self):
        for control in (
            self.CONTROL.replace("ARG GPU_TOOLS\n", ""),
            self.CONTROL.replace("ARG GPU_TOOLS\n", "ARG GPU_TOOLS=docker.io/x/y\n"),
        ):
            with (
                self.subTest(control=control),
                tempfile.TemporaryDirectory() as directory,
            ):
                with self.assertRaisesRegex(
                    layout.LayoutFailure, "without a global `ARG GPU_TOOLS`"
                ):
                    layout.check_gpu_adapter_pins(
                        self.fixture(directory, control=control)
                    )

    def test_only_lease_or_command_sandbox_tasks_need_the_shared_adapter_stages(self):
        bare = "ARG VERIFIER_BASE\nFROM ${VERIFIER_BASE}\n"
        with (
            tempfile.TemporaryDirectory() as directory
        ):  # e.g. a standalone Verifier (vllm)
            layout.check_gpu_adapter_pins(
                self.fixture(directory, control=bare, gpu=bare, settings="")
            )
        for settings, feature in (
            (self.LEASE, "lease_seconds"),
            (self.COMMAND, "access"),
        ):
            with (
                self.subTest(settings=settings),
                tempfile.TemporaryDirectory() as directory,
            ):
                with self.assertRaisesRegex(layout.LayoutFailure, feature):
                    layout.check_gpu_adapter_pins(
                        self.fixture(directory, control=bare, settings=settings)
                    )

    def test_a_gpu_verifier_needs_the_shared_adapter_and_verifier_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            control = "ARG VERIFIER_BASE\nFROM ${VERIFIER_BASE}\n"
            with self.assertRaisesRegex(
                layout.LayoutFailure, r"FROM \$\{GPU_TOOLS\} AS"
            ):
                layout.check_gpu_adapter_pins(self.fixture(directory, control=control))
        with tempfile.TemporaryDirectory() as directory:
            gpu = "ARG GPU_TOOLS\nFROM ${GPU_TOOLS} AS t\nFROM scratch\n"
            with self.assertRaisesRegex(
                layout.LayoutFailure, r"FROM \$\{VERIFIER_BASE\} AS"
            ):
                layout.check_gpu_adapter_pins(self.fixture(directory, gpu=gpu))

    def test_a_command_sandbox_control_pod_needs_no_gpu_environment(self):
        # The shared verifier keeps the Driver bearer for declared sandboxes (verifier.json
        # `sandboxes`), so the control image sets no FRONTIER_VERIFIER_GPU_TYPE for it.
        bare = self.CONTROL.replace(
            "ENV FRONTIER_VERIFIER_GPU_TYPE=A100-80GB FRONTIER_VERIFIER_GPU_COUNT=1\n",
            "",
        )
        with tempfile.TemporaryDirectory() as directory:
            layout.check_gpu_adapter_pins(
                self.fixture(directory, control=bare, settings=self.COMMAND)
            )

    def test_a_cpu_only_task_with_command_sandboxes_needs_only_the_control_adapter(
        self,
    ):
        bare = self.CONTROL.replace(
            "ENV FRONTIER_VERIFIER_GPU_TYPE=A100-80GB FRONTIER_VERIFIER_GPU_COUNT=1\n",
            "",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cpu_task"
            (root / "docker").mkdir(parents=True)
            (root / "task.toml").write_text(self.COMMAND)
            (root / "docker/verifier.Dockerfile").write_text(bare)
            layout.check_gpu_adapter_pins(
                root
            )  # no verifier-gpu recipe exists or is needed
            (root / "docker/verifier.Dockerfile").write_text(
                "ARG VERIFIER_BASE\nFROM ${VERIFIER_BASE}\n"
            )
            with self.assertRaisesRegex(
                layout.LayoutFailure, r"FROM \$\{GPU_TOOLS\} AS"
            ):
                layout.check_gpu_adapter_pins(root)


class BasesTagTests(unittest.TestCase):
    """bases.json names each shared image by its derived tag (src-<tree hash of its sources>)."""

    def test_a_hand_picked_or_stale_tag_is_refused(self):
        recorded = task_manifest.bases()
        for tag in ("v11", "src-000000000000"):
            bad = dict(
                recorded,
                **{
                    "gpu-tools": "docker.io/joyemang33/frontier-gpu-tools:"
                    f"{tag}@sha256:" + "0" * 64
                },
            )
            with (
                self.subTest(tag=tag),
                mock.patch.object(task_manifest, "bases", return_value=bad),
            ):
                with self.assertRaisesRegex(
                    layout.LayoutFailure, "its sources are src-"
                ):
                    layout.check_bases()

    def test_the_derived_tags_pass(self):
        import build_base_images

        derived = {
            role: f"docker.io/joyemang33/{build_base_images.IMAGE_NAMES[role]}:"
            f"{build_base_images.source_tag(role)}@sha256:" + "0" * 64
            for role in task_manifest.BASE_ROLES
        }
        with mock.patch.object(
            task_manifest,
            "bases",
            return_value={**derived, "schema_version": "frontier.task-bases/v2"},
        ):
            layout.check_bases()


class SidecarLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "minimal_task"
        shutil.copytree(ROOT / "tests/fixtures/export-minimal", self.root)
        self.policy = json.loads((self.root / "policy.json").read_text())
        self.policy["gpu"] = {
            "verifier": {"max_gpu_instances": 1, "max_gpu_seconds": 60}
        }

    def declare(self, name, recipe=True, **profile):
        if recipe:
            (self.root / f"sidecar-{name}").mkdir()
            (self.root / f"sidecar-{name}/Dockerfile").write_text(
                "FROM python:3.11-slim-bookworm\n"
            )
        self.policy.setdefault("sidecars", {})[name] = {
            "args": ["bash", "-lc", "exec sleep infinity"],
            "workdir": "/testbed",
            "phases": ["eval"],
            **profile,
        }

    def test_declared_sidecars_need_recipes_gpu_roles_and_valid_profiles(self):
        self.declare("scheduler", phases=["train", "eval"], attestation_key=True)
        self.declare("swe-000")
        layout.check_sidecars(self.root, self.policy)
        rejected = [
            ("main", {}),
            ("Swe_1", {}),
            ("swe-001", {"recipe": False}),
            ("swe-002", {"phases": ["agent"]}),
            ("swe-003", {"phases": ["eval", "eval"]}),
            ("swe-004", {"args": []}),
            ("swe-005", {"workdir": "relative"}),
            ("swe-006", {"attestation_key": "yes"}),
        ]
        for name, profile in rejected:
            with self.subTest(name=name):
                policy = copy.deepcopy(self.policy)
                saved = dict(self.policy)
                self.policy = policy
                self.declare(name, **profile)
                with self.assertRaises(layout.LayoutFailure):
                    layout.check_sidecars(self.root, self.policy)
                shutil.rmtree(self.root / f"sidecar-{name}", ignore_errors=True)
                self.policy = saved

    def test_sidecars_may_share_one_image_role(self):
        self.declare("swe-000-vanilla", image="swe-000")
        with self.assertRaises(layout.LayoutFailure):
            layout.check_sidecars(self.root, self.policy)
        shutil.move(
            self.root / "sidecar-swe-000-vanilla", self.root / "sidecar-swe-000"
        )
        self.declare("swe-000-reference", recipe=False, image="swe-000")
        self.declare("swe-000-check", recipe=False, image="swe-000")
        layout.check_sidecars(self.root, self.policy)
        self.policy["sidecars"]["swe-000-check"]["image"] = "Bad_Role"
        with self.assertRaises(layout.LayoutFailure):
            layout.check_sidecars(self.root, self.policy)

    def test_stray_sidecar_directory_is_rejected(self):
        self.declare("swe-000")
        (self.root / "sidecar-extra").mkdir()
        (self.root / "sidecar-extra/Dockerfile").write_text(
            "FROM python:3.11-slim-bookworm\n"
        )
        with self.assertRaises(layout.LayoutFailure):
            layout.check_sidecars(self.root, self.policy)

    def test_more_than_four_sidecars_need_a_parent_reservation(self):
        for index in range(5):
            self.declare(f"swe-{index:03}")
        with self.assertRaises(layout.LayoutFailure):
            layout.check_sidecars(self.root, self.policy)
        self.policy["gpu_parent_resources"] = {
            "verifier": {"cpu_millis": 159, "memory_bytes": 5 * 32 * 1024 * 1024}
        }
        with self.assertRaises(layout.LayoutFailure):
            layout.check_sidecars(self.root, self.policy)
        self.policy["gpu_parent_resources"]["verifier"]["cpu_millis"] = 160
        layout.check_sidecars(self.root, self.policy)
        self.policy["gpu_parent_resources"]["agent"] = {
            "cpu_millis": 1000,
            "memory_bytes": 1024 * 1024,
        }
        with self.assertRaises(layout.LayoutFailure):
            layout.check_sidecars(self.root, self.policy)


class UniformWorkloadMetricsTests(unittest.TestCase):
    NAMES = ["throughput", "restore_ns", "bytes"]

    def check(self, task: str, spec: dict) -> str:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            layout.check_uniform_workload_metrics(
                Path("/tasks") / task, spec, self.NAMES
            )
        return stderr.getvalue()

    def test_workloads_carrying_every_metric_pass(self):
        for spec in (
            {},
            {"workloads": [{"name": "spectrum"}, {"name": "micro"}]},
            {"workloads": ["spectrum", "micro"]},
            {"workloads": [{"name": "micro", "metrics": list(reversed(self.NAMES))}]},
            {
                "final_only_workloads": [{"name": "large"}],
                "final_only_metrics": list(self.NAMES),
            },
        ):
            with self.subTest(spec=spec):
                self.assertEqual(self.check("evm_state_checkpoints", spec), "")

    def test_workload_measuring_a_metric_subset_is_rejected(self):
        for spec in (
            # EVM: the micro-workload measured restore time but no throughput.
            {
                "workloads": [
                    {"name": "spectrum", "metrics": ["throughput"]},
                    {"name": "micro", "metrics": ["restore_ns", "bytes"]},
                ]
            },
            {"workloads": [{"name": "micro", "metrics": {"throughput": "x"}}]},
            {"workloads": [{"name": "micro", "metrics": self.NAMES + ["extra"]}]},
            {
                "workloads": [{"name": "spectrum"}],
                "final_only_workloads": [{"name": "large"}],
                "final_only_metrics": ["restore_ns", "bytes"],
            },
        ):
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(
                    layout.LayoutFailure,
                    "every workload must measure every declared metric",
                ):
                    self.check("evm_state_checkpoints", spec)

    def test_workload_groups_are_retired(self):
        spec = {
            "workload_groups": {
                "rpc": {"metrics": ["throughput"], "counts": {"train": 2, "final": 4}},
                "experiment": {
                    "metrics": ["restore_ns", "bytes"],
                    "counts": {"train": 1, "final": 2},
                },
            }
        }
        with self.assertRaisesRegex(layout.LayoutFailure, "workload_groups is retired"):
            self.check("evm_state_checkpoints", spec)

    def test_exempt_task_warns_and_stale_exemption_fails(self):
        spec = {"workloads": [{"name": "micro", "metrics": ["restore_ns"]}]}
        for task in layout.NONUNIFORM_WORKLOAD_METRICS_EXEMPT:
            with self.subTest(task=task):
                warning = self.check(task, spec)
                self.assertIn(f"WARN {task}: every workload must measure", warning)
                with self.assertRaisesRegex(layout.LayoutFailure, "stale exemption"):
                    self.check(task, {})

    def test_catalog_warns_only_for_exempt_tasks(self):
        names = task_layers.load_definitions()
        warned = set()
        for name in names:
            root = renderer.task_directory(name)
            spec_path = root / "judge/beat_spec.json"
            if not spec_path.is_file():
                continue
            spec = json.loads(spec_path.read_text())
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                layout.check_uniform_workload_metrics(
                    root, spec, [row["name"] for row in spec["metrics"]]
                )
            if stderr.getvalue():
                warned.add(root.name)
        self.assertEqual(warned, set(layout.NONUNIFORM_WORKLOAD_METRICS_EXEMPT))

    def test_feedback_probe_rejects_a_row_missing_a_declared_metric(self):
        # The legacy (policy.json) feedback probe, on a legacy task that still has a beat spec and
        # its own helper (task.toml tasks have neither).
        source = next(
            (
                path
                for path in sorted(layout.DEFAULT_PROBLEMS_ROOT.glob("*/*"))
                if (path / "policy.json").is_file()
                and (path / "judge/beat_spec.json").is_file()
                and (path / "verifier/beat_spec_feedback.py").is_file()
            ),
            None,
        )
        if source is None:
            self.skipTest(
                "no policy.json task with a beat spec and its own helper remains"
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / source.name
            for name in ("judge/beat_spec.json", "verifier/beat_spec_feedback.py"):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, root / name)
            spec = json.loads((root / "judge/beat_spec.json").read_text())
            names = [row["name"] for row in spec["metrics"]]
            visible = [row["name"] for row in spec["solver_feedback"]]
            layout.check_beat_spec_feedback(root, spec, names, visible)
            # A helper that fills in a row's missing metrics instead of rejecting it.
            helper = root / "verifier/beat_spec_feedback.py"
            helper.write_text(
                helper.read_text()
                + """

import copy as _copy
import json as _json
_strict_public_metrics = public_metrics


def public_metrics(metrics, **kwargs):
    metrics = _copy.deepcopy(metrics)
    spec = _json.loads(Path(kwargs["beat_spec_path"]).read_text())
    for row in metrics.get("beat_spec_workload_passes", []):
        if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
            for entry in spec["metrics"]:
                row["metrics"].setdefault(entry["name"], True)
    return _strict_public_metrics(metrics, **kwargs)
"""
            )
            with self.assertRaisesRegex(
                layout.LayoutFailure, "workload row missing a declared metric"
            ):
                layout.check_beat_spec_feedback(root, spec, names, visible)


class SharedFromTests(unittest.TestCase):
    """--shared-from: a staging branch (task-format/landing) must carry exactly main's
    tasks/shared, so every branch builds the same bases."""

    def git(self, root, *arguments):
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.email=t@example.com",
                "-c",
                "user.name=t",
                *arguments,
            ],
            check=True,
            capture_output=True,
        )

    def test_a_differing_or_lagging_shared_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks/shared/lib").mkdir(parents=True)
            (root / "tasks/shared/lib/isolate.py").write_text("main\n")
            (root / "tasks/task.txt").write_text("task\n")
            self.git(root, "init", "-q", "-b", "main")
            self.git(root, "add", "-A")
            self.git(root, "commit", "-qm", "main")
            self.git(root, "checkout", "-q", "-b", "landing")
            (root / "tasks/task.txt").write_text("a task change\n")
            layout.check_shared_from(
                "main", root
            )  # task changes are landing's business
            (root / "tasks/shared/lib/isolate.py").write_text("landing\n")
            self.git(root, "commit", "-qam", "shared edit on landing")
            with self.assertRaisesRegex(
                layout.LayoutFailure,
                "tasks/shared differs from main: tasks/shared/lib/isolate.py",
            ):
                layout.check_shared_from("main", root)
            with self.assertRaisesRegex(layout.LayoutFailure, "cannot compare"):
                layout.check_shared_from("no-such-ref", root)
