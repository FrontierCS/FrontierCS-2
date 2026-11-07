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


class CpuMachineSiteTests(unittest.TestCase):
    """execution.cpu_machines: Verifier CPU machines on Kubernetes, declared by the site."""

    KEYS = (
        "MANAGEMENT_CPU_RUNNER_BACKEND",
        "MANAGEMENT_CPU_RUNNER_PLACEMENT",
        "MANAGEMENT_CPU_MACHINE_MAX_INSTANCES",
        "MANAGEMENT_CPU_MACHINE_MAX_CPU_MILLIS",
        "MANAGEMENT_CPU_MACHINE_MAX_MEMORY_BYTES",
    )
    EVAL16 = {
        "MANAGEMENT_CPU_RUNNER_BACKEND": "kubernetes",
        "MANAGEMENT_CPU_RUNNER_PLACEMENT":
            '{"node_selector":{"eks.amazonaws.com/nodegroup":"example-cpu-nodegroup"}}',
        "MANAGEMENT_CPU_MACHINE_MAX_INSTANCES": "4",
        "MANAGEMENT_CPU_MACHINE_MAX_CPU_MILLIS": "16000",
        "MANAGEMENT_CPU_MACHINE_MAX_MEMORY_BYTES": "68719476736",
    }

    def setUp(self):
        self.raw = json.loads((ROOT / "config/site.example.json").read_text())
        self.machines = copy.deepcopy(self.raw["execution"]["cpu_machines"])

    @staticmethod
    def render(raw):
        config = frontier_deploy.validate_config(raw)
        return json.loads(frontier_deploy.render_runtime_manifest(
            config, Fixture.outputs(config), Fixture.images(config), ["172.20.0.1/32"]))

    @staticmethod
    def item(manifest, kind, name):
        return next(x for x in manifest["items"]
                    if x["kind"] == kind and x["metadata"]["name"] == name)

    def test_the_example_site_renders_its_cpu_machines_and_absent_renders_no_key(self):
        declared = self.render(self.raw)
        cm = self.item(declared, "ConfigMap", "frontier-management-config")
        self.assertEqual({k: cm["data"][k] for k in self.KEYS}, self.EVAL16)
        absent = copy.deepcopy(self.raw)
        del absent["execution"]["cpu_machines"]
        plain = self.render(absent)
        plain_cm = self.item(plain, "ConfigMap", "frontier-management-config")
        self.assertFalse(set(self.KEYS) & set(plain_cm["data"]))
        # Only Management's configuration (and so its rollout annotation) differs.
        for key in self.KEYS:
            cm["data"].pop(key)
        before = self.item(plain, "Deployment", "frontier-management")
        after = self.item(declared, "Deployment", "frontier-management")
        self.assertNotEqual(before["spec"]["template"]["metadata"]["annotations"],
                            after["spec"]["template"]["metadata"]["annotations"])
        after["spec"]["template"]["metadata"]["annotations"] = before["spec"]["template"]["metadata"]["annotations"]
        self.assertEqual(plain, declared)

    def test_placement_is_the_canonical_json_the_contract_prints(self):
        from preview.config import cpu_machine_management_env
        self.machines["placement"] = {
            "node_selector": {"z": "1", "a": "2"},
            "tolerations": [{"operator": "Exists", "key": "k", "effect": "NoSchedule"}],
        }
        self.assertEqual(
            cpu_machine_management_env(self.machines)["MANAGEMENT_CPU_RUNNER_PLACEMENT"],
            '{"node_selector":{"a":"2","z":"1"},'
            '"tolerations":[{"effect":"NoSchedule","key":"k","operator":"Exists"}]}')

    def invalid(self):
        m = self.machines
        placement = m["placement"]
        return [
            None, [], {}, {**m, "extra": 1},
            {k: v for k, v in m.items() if k != "max_memory_bytes"},
            {**m, "placement": {}}, {**m, "placement": {"node_selector": {}}},
            {**m, "placement": {**placement, "extra": 1}},
            {**m, "placement": {"node_selector": {"": "x"}}},
            {**m, "placement": {"node_selector": {"k": "v" * 64}}},
            {**m, "placement": {"node_selector": {"k": 1}}},
            {**m, "placement": {**placement, "tolerations": []}},
            {**m, "placement": {**placement, "tolerations": [{}]}},
            {**m, "max_instances": 0}, {**m, "max_instances": 65}, {**m, "max_instances": True},
            {**m, "max_cpu_millis": 1500}, {**m, "max_cpu_millis": 65000}, {**m, "max_cpu_millis": "16000"},
            {**m, "max_memory_bytes": 0}, {**m, "max_memory_bytes": (256 << 30) + (1 << 20)},
            {**m, "max_memory_bytes": (8 << 30) + 1},
        ]

    def test_config_and_schema_refuse_malformed_or_partial_declarations(self):
        import jsonschema
        schema = json.loads((ROOT / "config/site.schema.json").read_bytes())[
            "properties"]["execution"]["properties"]["cpu_machines"]
        jsonschema.validate(self.machines, schema)
        for value in self.invalid():
            with self.subTest(value=value):
                raw = copy.deepcopy(self.raw)
                raw["execution"]["cpu_machines"] = value
                with self.assertRaises(frontier_deploy.DeployError):
                    frontier_deploy.validate_config(raw)
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(value, schema)

    def test_upgrade_sets_replaces_removes_and_rejects_overrides(self):
        from preview import cli
        dep = {"metadata": {"resourceVersion": "2"}, "spec": {"template": {
            "metadata": {"annotations": {}}, "spec": {"containers": [{"name": "management", "image": "old"}]}}}}
        images = {k: {"reference": "new-" + k} for k in ("management", "driver")}

        def configmap(extra):
            return {"metadata": {"resourceVersion": "1"}, "data": {
                "MANAGEMENT_DRIVER_BACKENDS": "local-k8s", "MANAGEMENT_DRIVER_IMAGE": "old", **extra}}

        # Not passed: the keys stay as they are.
        stale = {"MANAGEMENT_CPU_MACHINE_MAX_INSTANCES": "2"}
        _, _, untouched = cli.runtime_upgrade_patches(configmap(stale), dep, images)
        self.assertEqual(untouched["MANAGEMENT_CPU_MACHINE_MAX_INSTANCES"], "2")
        # Declared: every key is set, the stale one replaced.
        patches, _, data = cli.runtime_upgrade_patches(configmap(stale), dep, images, cpu_machines=self.machines)
        self.assertEqual({k: data[k] for k in self.KEYS}, self.EVAL16)
        self.assertIn({"op": "replace", "path": "/data/MANAGEMENT_CPU_MACHINE_MAX_INSTANCES", "value": "4"}, patches)
        self.assertIn({"op": "add", "path": "/data/MANAGEMENT_CPU_RUNNER_BACKEND", "value": "kubernetes"}, patches)
        # Absent from the site: every key is removed, none invented.
        patches, _, data = cli.runtime_upgrade_patches(configmap(self.EVAL16), dep, images, cpu_machines=None)
        self.assertFalse(set(self.KEYS) & set(data))
        for key in self.KEYS:
            self.assertIn({"op": "remove", "path": "/data/" + key}, patches)
        _, _, data = cli.runtime_upgrade_patches(configmap({}), dep, images, cpu_machines=None)
        self.assertFalse(set(self.KEYS) & set(data))
        with self.assertRaisesRegex(frontier_deploy.DeployError, "cpu_machines.max_cpu_millis"):
            cli.runtime_upgrade_patches(configmap({}), dep, images,
                                        cpu_machines={**self.machines, "max_cpu_millis": 1500})
        for key in self.KEYS:
            with self.subTest(key=key):
                overridden = copy.deepcopy(dep)
                overridden["spec"]["template"]["spec"]["containers"][0]["env"] = [{"name": key, "value": "x"}]
                with self.assertRaisesRegex(frontier_deploy.DeployError, "direct environment overrides"):
                    cli.runtime_upgrade_patches(configmap({}), overridden, images, cpu_machines=self.machines)
