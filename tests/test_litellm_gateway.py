from __future__ import annotations

#
import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import json
import unittest
from unittest import mock

from preview import cli, infra
from test_frontier_deploy import ROOT, Fixture, frontier_deploy


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.config = frontier_deploy.validate_config(
            json.loads((ROOT / "config/site.example.json").read_text())
        )
        self.outputs = Fixture.outputs(self.config)
        self.images = Fixture.images(self.config)

    def test_key_worker_namespace_absence_grant_is_read_only_and_worker_only(self):
        items = infra.litellm_worker_resources(self.config, self.outputs, {})
        name = "frontier-key-worker-namespace-reader"
        role = next(x for x in items if x["kind"] == "ClusterRole" and x["metadata"]["name"] == name)
        self.assertEqual(role["rules"], [{"apiGroups": [""], "resources": ["namespaces"], "verbs": ["get"]}])
        binding = next(x for x in items if x["kind"] == "ClusterRoleBinding" and x["metadata"]["name"] == name)
        self.assertEqual(binding["roleRef"], {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": name})
        self.assertEqual(binding["subjects"], [{"kind": "ServiceAccount", "name": "frontier-litellm-key-worker", "namespace": "frontier-litellm"}])

    def test_key_worker_gateway_egress_binds_service_and_backend(self):
        config = copy.deepcopy(self.config)
        config["network"]["litellm_cluster_ip"] = "172.20.0.77"
        items = infra.litellm_worker_resources(config, self.outputs, {})
        policy = next(x for x in items if x["kind"] == "NetworkPolicy")
        gateway = [x for x in policy["spec"]["egress"]
                   if x["ports"] == [{"protocol": "TCP", "port": 4000}]]
        self.assertEqual(gateway, [
            {"to": [{"podSelector": {"matchLabels": {"frontier.dev/network-role": "litellm"}}}],
             "ports": [{"protocol": "TCP", "port": 4000}]},
            {"to": [{"ipBlock": {"cidr": "172.20.0.77/32"}}],
             "ports": [{"protocol": "TCP", "port": 4000}]},
        ])

    def test_adapter_change_rolls_gateway_but_not_postgres(self):
        def rendered():
            return json.loads(
                infra.render_litellm_manifest(self.config, self.outputs, self.images)
            )["items"]

        def resource(items, kind, name):
            return next(
                x for x in items if x["kind"] == kind and x["metadata"]["name"] == name
            )

        before = rendered()
        self.assertEqual(before, rendered())
        original_read = type(ROOT).read_text

        def changed_read(path, *args, **kwargs):
            text = original_read(path, *args, **kwargs)
            return (
                text + "\n# changed adapter\n"
                if path.name == "preview_audit.py"
                else text
            )

        with mock.patch.object(type(ROOT), "read_text", changed_read):
            after = rendered()
        old = resource(before, "Deployment", "frontier-litellm")["spec"]["template"]
        new = resource(after, "Deployment", "frontier-litellm")["spec"]["template"]
        key = "frontier.dev/config-sha256"
        self.assertNotEqual(
            old["metadata"]["annotations"][key], new["metadata"]["annotations"][key]
        )
        self.assertEqual(old["spec"], new["spec"])
        self.assertEqual(
            resource(before, "StatefulSet", "frontier-litellm-postgres"),
            resource(after, "StatefulSet", "frontier-litellm-postgres"),
        )
        data = resource(after, "ConfigMap", "frontier-litellm-config")["data"]
        self.assertEqual(
            new["metadata"]["annotations"][key], infra.canonical_object_sha256(data)
        )

    def test_local_runtime_and_simulator_bootstrap(self):
        def settings(bootstrap):
            items = json.loads(
                infra.render_runtime_manifest(
                    self.config,
                    self.outputs,
                    self.images,
                    ["172.20.0.1/32"],
                    simulator_bootstrap=bootstrap,
                )
            )["items"]
            return next(
                x["data"]
                for x in items
                if x["kind"] == "ConfigMap"
                and x["metadata"]["name"] == "frontier-management-config"
            )

        live = settings(False)
        self.assertEqual(
            live["MANAGEMENT_LITELLM_KEY_WORKER_IMAGE"],
            self.images["management"]["reference"],
        )
        self.assertEqual(
            live["MANAGEMENT_LITELLM_SERVICE_CIDR"],
            self.config["network"]["litellm_cluster_ip"] + "/32",
        )
        self.assertNotIn("MANAGEMENT_LITELLM_SERVICE_CIDR", settings(True))

    def test_audit_storage_and_worker_boundary(self):
        items = json.loads(
            infra.render_litellm_manifest(self.config, self.outputs, self.images)
        )["items"]

        def item(kind, name):
            return next(
                x for x in items if x["kind"] == kind and x["metadata"]["name"] == name
            )

        gateway = item("Deployment", "frontier-litellm")["spec"]["template"]["spec"]
        env = {x["name"]: x for x in gateway["containers"][0]["env"]}
        self.assertEqual(
            env["FRONTIER_MODEL_AUDIT_ROOT"]["value"],
            "/var/lib/frontier-local/artifacts/" + self.outputs["artifact_bucket"],
        )
        self.assertFalse(gateway["automountServiceAccountToken"])
        volume = item("PersistentVolume", "frontier-litellm-objects")
        self.assertEqual(
            volume["spec"]["csi"]["volumeHandle"],
            self.outputs["efs_file_system_id"]
            + "::"
            + self.outputs["efs_object_store_access_point_id"],
        )
        role = item("Role", "frontier-key-worker-serving-ready-reader")
        self.assertEqual(
            role["rules"][0]["resourceNames"], ["frontier-litellm-serving-ready"]
        )
        self.assertEqual(role["rules"][0]["verbs"], ["get"])
        proxy = json.loads(
            item("ConfigMap", "frontier-litellm-config")["data"]["config.yaml"]
        )
        self.assertIn(
            "preview_audit.preview_audit", proxy["litellm_settings"]["callbacks"]
        )

    def test_serving_receipt_requires_ready_resources_and_binds_identity(self):
        service = {
            "metadata": {"uid": "service-uid"},
            "spec": {"clusterIP": "172.20.0.20"},
        }
        postgres = {"metadata": {"uid": "postgres-uid"}, "status": {"readyReplicas": 1}}
        pvc = {"metadata": {"uid": "volume-uid"}, "status": {"phase": "Bound"}}
        admin = {"metadata": {"uid": "admin-uid"}}
        deployment = {"status": {"readyReplicas": 1}}
        receipt = infra.litellm_serving_receipt(
            service, postgres, pvc, admin, deployment
        )
        self.assertTrue(receipt["immutable"])
        self.assertEqual(receipt["data"]["admin_identity_uid"], "admin-uid")
        self.assertEqual(
            receipt,
            infra.litellm_serving_receipt(service, postgres, pvc, admin, deployment),
        )
        replacement = copy.deepcopy(admin)
        replacement["metadata"]["uid"] = "replacement-uid"
        self.assertNotEqual(
            receipt["data"]["identity_generation"],
            infra.litellm_serving_receipt(
                service, postgres, pvc, replacement, deployment
            )["data"]["identity_generation"],
        )
        with self.assertRaises(infra.DeployError):
            infra.litellm_serving_receipt(service, postgres, pvc, admin, {"status": {}})

    def test_upgrade_keeps_worker_and_management_image_in_sync(self):
        cm = {
            "metadata": {"resourceVersion": "1"},
            "data": {
                "MANAGEMENT_DRIVER_BACKENDS": "local-k8s",
                "MANAGEMENT_DRIVER_IMAGE": "old",
            },
        }
        deployment = {
            "metadata": {"resourceVersion": "2"},
            "spec": {
                "template": {
                    "metadata": {},
                    "spec": {"containers": [{"name": "management", "image": "old"}]},
                }
            },
        }
        _, _, data = cli.runtime_upgrade_patches(
            cm, deployment, self.images, litellm_cidr="172.20.0.20/32"
        )
        self.assertEqual(
            data["MANAGEMENT_LITELLM_KEY_WORKER_IMAGE"],
            self.images["management"]["reference"],
        )
        with self.assertRaises(infra.DeployError):
            cli.runtime_upgrade_patches(
                cm, deployment, self.images, litellm_cidr="172.20.0.0/16"
            )


if __name__ == "__main__":
    unittest.main()
