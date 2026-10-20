"""Explicit operator node qualification never changes scientific task definitions."""

import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import json
import unittest

from test_frontier_deploy import ROOT, Fixture, frontier_deploy


class VerifierPlacementTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / "config/site.example.json").read_text())
        self.policy = {
            "task_version_id": "taskver_5WM4FJ170HKR4WV7QDJP8DM8E7",
            "nodes": ["node-a.example", "node-b.example"],
        }

    def test_required_logical_ids_change_only_management_configuration(self):
        def render(raw):
            config = frontier_deploy.validate_config(raw)
            return json.loads(frontier_deploy.render_runtime_manifest(
                config, Fixture.outputs(config), Fixture.images(config), ["172.20.0.1/32"]))

        original = render(self.raw)
        selected = copy.deepcopy(self.raw)
        selected["execution"]["verifier_placement_required_task_ids"] = []
        self.assertEqual(original, render(selected))
        selected["execution"]["verifier_placement_required_task_ids"] = ["task-b", "task-a"]
        required = render(selected)
        cm = next(x for x in required["items"] if x["kind"] == "ConfigMap"
                  and x["metadata"]["name"] == "frontier-management-config")
        self.assertEqual(cm["data"].pop("MANAGEMENT_VERIFIER_PLACEMENT_REQUIRED_TASK_IDS"), '["task-a","task-b"]')
        before = next(x for x in original["items"] if x["kind"] == "Deployment"
                      and x["metadata"]["name"] == "frontier-management")
        after = next(x for x in required["items"] if x["kind"] == "Deployment"
                     and x["metadata"]["name"] == "frontier-management")
        self.assertNotEqual(before["spec"]["template"]["metadata"]["annotations"],
                            after["spec"]["template"]["metadata"]["annotations"])
        after["spec"]["template"]["metadata"]["annotations"] = before["spec"]["template"]["metadata"]["annotations"]
        self.assertEqual(original, required)

    def test_required_logical_ids_validate_and_schema_rejects_malformed_intent(self):
        import jsonschema
        schema = json.loads((ROOT / "config/site.schema.json").read_bytes())["properties"]["execution"]["properties"]["verifier_placement_required_task_ids"]
        for value in (None, {}, [True], [""], ["task", "task"], [" task"],
                      ["task\n"], ["ta\0sk"], ["task\x85"], ["t" * 129], ["t" + str(i) for i in range(65)]):
            with self.subTest(value=value):
                raw = copy.deepcopy(self.raw)
                raw["execution"]["verifier_placement_required_task_ids"] = value
                with self.assertRaises(frontier_deploy.DeployError):
                    frontier_deploy.validate_config(raw)
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(value, schema)
        jsonschema.validate(["frontier-cs-2.0-diffusion-policy-cpu"], schema)

    def test_required_ids_upgrade_preserves_none_but_explicit_empty_removes(self):
        from preview import cli
        key = "MANAGEMENT_VERIFIER_PLACEMENT_REQUIRED_TASK_IDS"
        cm = {"metadata": {"resourceVersion": "1"}, "data": {
            "MANAGEMENT_DRIVER_BACKENDS": "local-k8s", "MANAGEMENT_DRIVER_IMAGE": "old", key: '["task-old"]'}}
        dep = {"metadata": {"resourceVersion": "2"}, "spec": {"template": {
            "metadata": {"annotations": {}}, "spec": {"containers": [{"name": "management", "image": "old"}]}}}}
        images = {role: {"reference": "new-" + role} for role in ("management", "driver")}
        _, _, untouched = cli.runtime_upgrade_patches(cm, dep, images)
        self.assertEqual(untouched[key], cm["data"][key])
        patches, _, updated = cli.runtime_upgrade_patches(cm, dep, images, verifier_placement_required_task_ids=["task-new"])
        self.assertEqual(updated[key], '["task-new"]')
        self.assertIn({"op": "add", "path": "/data/" + key, "value": '["task-new"]'}, patches)
        patches, _, removed = cli.runtime_upgrade_patches(cm, dep, images, verifier_placement_required_task_ids=[])
        self.assertNotIn(key, removed)
        self.assertIn({"op": "remove", "path": "/data/" + key}, patches)
        dep["spec"]["template"]["spec"]["containers"][0]["env"] = [{"name": key, "value": "[]"}]
        with self.assertRaises(frontier_deploy.DeployError):
            cli.runtime_upgrade_patches(cm, dep, images, verifier_placement_required_task_ids=["task-new"])

    def test_opt_in_config_and_render_bind_only_management_environment(self):
        def render(raw):
            config = frontier_deploy.validate_config(raw)
            return json.loads(
                frontier_deploy.render_runtime_manifest(
                    config,
                    Fixture.outputs(config),
                    Fixture.images(config),
                    ["172.20.0.1/32"],
                )
            )

        original = render(self.raw)
        explicit_empty = copy.deepcopy(self.raw)
        explicit_empty["execution"]["verifier_node_allowlists"] = []
        self.assertEqual(original, render(explicit_empty))
        selected = copy.deepcopy(self.raw)
        selected["execution"]["verifier_node_allowlists"] = [self.policy]
        placed = render(selected)
        cm = next(
            x
            for x in placed["items"]
            if x["kind"] == "ConfigMap"
            and x["metadata"]["name"] == "frontier-management-config"
        )
        self.assertEqual(
            json.loads(cm["data"].pop("MANAGEMENT_VERIFIER_NODE_ALLOWLISTS")),
            [self.policy],
        )
        # Management rolls when its explicit configuration changes.
        before = next(
            x
            for x in original["items"]
            if x["kind"] == "Deployment"
            and x["metadata"]["name"] == "frontier-management"
        )
        after = next(
            x
            for x in placed["items"]
            if x["kind"] == "Deployment"
            and x["metadata"]["name"] == "frontier-management"
        )
        self.assertNotEqual(
            before["spec"]["template"]["metadata"]["annotations"],
            after["spec"]["template"]["metadata"]["annotations"],
        )
        after["spec"]["template"]["metadata"]["annotations"] = before["spec"][
            "template"
        ]["metadata"]["annotations"]
        self.assertEqual(original, placed)

    def test_invalid_empty_duplicate_or_unqualified_shaped_inputs_fail(self):
        cases = [
            None,
            {},
            [self.policy] * 2,
            [{**self.policy, "nodes": []}],
            [{**self.policy, "nodes": ["x"] * 33}],
            [{**self.policy, "nodes": ["x", "x"]}],
            [{**self.policy, "nodes": ["a..b"]}],
            [{**self.policy, "nodes": ["X"]}],
            [{**self.policy, "nodes": [True]}],
            [{**self.policy, "nodes": ["a" * 64]}],
            [{**self.policy, "task_version_id": "taskver_" + "Z" * 26}],
            [{**self.policy, "extra": True}],
        ]
        for value in cases:
            with self.subTest(value=value):
                raw = copy.deepcopy(self.raw)
                raw["execution"]["verifier_node_allowlists"] = value
                with self.assertRaises(frontier_deploy.DeployError):
                    frontier_deploy.validate_config(raw)

    def test_upgrade_updates_or_explicitly_removes_stale_policy_and_rejects_override(
        self,
    ):
        from preview import cli

        key = "MANAGEMENT_VERIFIER_NODE_ALLOWLISTS"
        cm = {
            "metadata": {"resourceVersion": "1"},
            "data": {
                "MANAGEMENT_DRIVER_BACKENDS": "local-k8s",
                "MANAGEMENT_DRIVER_IMAGE": "old",
                key: "stale",
            },
        }
        dep = {
            "metadata": {"resourceVersion": "2"},
            "spec": {
                "template": {
                    "metadata": {"annotations": {}},
                    "spec": {"containers": [{"name": "management", "image": "old"}]},
                }
            },
        }
        images = {k: {"reference": "new-" + k} for k in ("management", "driver")}
        for policy in ([self.policy], []):
            cp, _, data = cli.runtime_upgrade_patches(
                cm, dep, images, verifier_node_allowlists=policy
            )
            if policy:
                self.assertEqual(json.loads(data[key]), policy)
                self.assertTrue(
                    any(
                        p.get("path") == "/data/" + key and p["op"] == "add" for p in cp
                    )
                )
            else:
                self.assertNotIn(key, data)
                self.assertIn({"op": "remove", "path": "/data/" + key}, cp)
        dep["spec"]["template"]["spec"]["containers"][0]["env"] = [
            {"name": key, "value": "stale"}
        ]
        with self.assertRaises(frontier_deploy.DeployError):
            cli.runtime_upgrade_patches(
                cm, dep, images, verifier_node_allowlists=[self.policy]
            )

    def test_schema_accepts_the_same_operator_shape(self):
        import jsonschema

        schema = json.loads((ROOT / "config/site.schema.json").read_bytes())[
            "properties"
        ]["execution"]["properties"]["verifier_node_allowlists"]
        jsonschema.validate([self.policy], schema)
        for nodes in [[], ["x", "x"], ["bad..node"], ["UPPER"], ["a" * 64]]:
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate([{**self.policy, "nodes": nodes}], schema)
