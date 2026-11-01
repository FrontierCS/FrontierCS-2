from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_task_definition as renderer
import validate_task_layout as layout


class TaskLayoutTests(unittest.TestCase):
    def test_grouped_feedback_probe_uses_declared_roles_and_rejects_incomplete_rows(self):
        # No catalog task declares workload groups any more (quarter_pir judges one workload per
        # experiment); the grouped feedback path is exercised on a copy of its spec with two groups.
        task = ROOT / "tasks/frontier-cs-2.0-demo/problems/Crypto/quarter_pir"
        spec = json.loads((task / "judge/beat_spec.json").read_text())
        names = [row["name"] for row in spec["metrics"]]
        visible = [row["name"] for row in spec["solver_feedback"]]
        layout.check_beat_spec_feedback(task, spec, names, visible)
        helper = task / "verifier/beat_spec_feedback.py"
        namespace = {"__name__": "grouped_feedback_test", "__file__": str(helper)}
        exec(compile(helper.read_text(), str(helper), "exec"), namespace)
        grouped = copy.deepcopy(spec)
        grouped["workload_groups"] = {
            "rpc": {"metrics": names[:2], "counts": {"train": 1000, "final": 2000}},
            "experiment": {"metrics": names[2:], "counts": {"train": 1, "final": 2}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            spec_path = Path(temporary) / "beat_spec.json"
            spec_path.write_text(json.dumps(grouped))
            for role in ("train", "final"):
                values = {"beats_reference": 1, "beat_spec_suite_role": role,
                          "beat_spec_workload_passes": []}
                for group, record in grouped["workload_groups"].items():
                    values["beat_spec_workload_passes"].extend(
                        {"workload": f"{group}-{i}", "group": group,
                         "metrics": {name: True for name in record["metrics"]}}
                        for i in range(record["counts"][role]))
                actual = namespace["public_metrics"](
                    values, beat_spec_path=spec_path, configured_visible=visible)
                for record in grouped["workload_groups"].values():
                    for name in record["metrics"]:
                        self.assertEqual(actual["beat_spec_progress"][name]["total"],
                                         record["counts"][role])
                for mutation in ("role", "row", "group"):
                    invalid = copy.deepcopy(values)
                    if mutation == "role":
                        invalid.pop("beat_spec_suite_role")
                    elif mutation == "row":
                        invalid["beat_spec_workload_passes"].pop()
                    else:
                        invalid["beat_spec_workload_passes"][0]["group"] = "unknown"
                    with self.assertRaises(ValueError):
                        namespace["public_metrics"](
                            invalid, beat_spec_path=spec_path, configured_visible=visible)

    def test_every_catalog_task_has_a_complete_native_source_lock(self):
        names = json.loads((ROOT / "config/task-definitions.json").read_text())
        for name in names:
            with self.subTest(task=name):
                self.assertIn(
                    layout.validate_task(
                        renderer.problem_directory(layout.DEFAULT_PROBLEMS_ROOT, name)
                    ),
                    ("patch-workspace", "direct-payload", "patch-starter"),
                )

    def test_quota_drift_and_unlocked_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "batch_local_counting"
            shutil.copytree(
                renderer.problem_directory(layout.DEFAULT_PROBLEMS_ROOT, root.name),
                root,
            )
            path = root / "policy.json"
            original = path.read_bytes()
            policy = json.loads(original)
            policy["quota"]["max_train_calls"] = 3
            path.write_text(json.dumps(policy))
            with self.assertRaisesRegex(layout.LayoutFailure, "task-authored"):
                layout.validate_task(root)
            path.write_bytes(original)
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
        self.policy = json.loads(
            (
                renderer.problem_directory(
                    layout.DEFAULT_PROBLEMS_ROOT, "sparse_video_attention_policy"
                )
                / "policy.json"
            ).read_text()
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
        stock = renderer.problem_directory(
            layout.DEFAULT_PROBLEMS_ROOT, "batch_local_counting"
        )
        shutil.copyfile(
            stock / "verifier/beat_spec_feedback.py",
            root / "verifier/beat_spec_feedback.py",
        )
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
            policy = json.loads(
                (
                    renderer.problem_directory(
                        layout.DEFAULT_PROBLEMS_ROOT, "sparse_video_attention_policy"
                    )
                    / "policy.json"
                ).read_text()
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


if __name__ == "__main__":
    unittest.main()


class SidecarLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "minimal_task"
        shutil.copytree(ROOT / "tests/fixtures/export-minimal", self.root)
        self.policy = json.loads((self.root / "policy.json").read_text())
        self.policy["gpu"] = {"verifier": {"max_gpu_instances": 1, "max_gpu_seconds": 60}}

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
        shutil.move(self.root / "sidecar-swe-000-vanilla", self.root / "sidecar-swe-000")
        self.declare("swe-000-reference", recipe=False, image="swe-000")
        self.declare("swe-000-check", recipe=False, image="swe-000")
        layout.check_sidecars(self.root, self.policy)
        self.policy["sidecars"]["swe-000-check"]["image"] = "Bad_Role"
        with self.assertRaises(layout.LayoutFailure):
            layout.check_sidecars(self.root, self.policy)

    def test_stray_sidecar_directory_is_rejected(self):
        self.declare("swe-000")
        (self.root / "sidecar-extra").mkdir()
        (self.root / "sidecar-extra/Dockerfile").write_text("FROM python:3.11-slim-bookworm\n")
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
