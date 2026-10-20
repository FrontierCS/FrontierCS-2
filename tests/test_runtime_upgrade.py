# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from preview import cli

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "upgrade_deploy", ROOT / "scripts/frontier_deploy.py"
)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class RuntimeUpgradeTests(unittest.TestCase):
    def test_selection_validation_precedes_runtime_reads_or_mutations(self):
        with (
            mock.patch.object(
                cli,
                "load_task_selection",
                side_effect=d.DeployError("invalid selection"),
            ) as selection,
            mock.patch.object(cli, "load_runtime_images") as images,
        ):
            with self.assertRaisesRegex(d.DeployError, "invalid selection"):
                d.upgrade_runtime({}, Path("output"), Path("kubeconfig"))
        selection.assert_called_once_with({}, require_resolved=True)
        images.assert_not_called()

    def test_live_runs_block_even_after_earlier_empty_state_queries(self):
        for blocked in range(4):
            with (
                self.subTest(blocked=blocked),
                mock.patch.object(
                    cli,
                    "api_json",
                    side_effect=[[]] * blocked + [[{"id": "run_active"}]],
                ) as api,
            ):
                with self.assertRaisesRegex(d.DeployError, "zero"):
                    d.require_idle_runtime("http://management")
                self.assertEqual(api.call_count, blocked + 1)
                self.assertTrue(
                    all(
                        "state=" in call.args[2] and "limit=1" in call.args[2]
                        for call in api.call_args_list
                    )
                )

    def test_patch_preserves_existing_peers_gateway_and_deployment_settings(self):
        cm = {
            "metadata": {"resourceVersion": "42"},
            "data": {
                "MANAGEMENT_DRIVER_BACKENDS": "local-k8s",
                "MANAGEMENT_DRIVER_IMAGE": "old-driver",
                "MANAGEMENT_MODAL_ALLOWED_CIDRS": "1.2.3.4/32",
                "gateway": "existing",
                "capacity": "3",
            },
        }
        dep = {
            "metadata": {"resourceVersion": "71"},
            "spec": {
                "template": {
                    "metadata": {"annotations": {"keep": "original"}},
                    "spec": {
                        "containers": [
                            {"name": "sidecar", "image": "unchanged"},
                            {"name": "management", "image": "old-management"},
                        ],
                        "serviceAccountName": "existing",
                    },
                }
            },
        }
        original = copy.deepcopy((cm, dep))
        images = {k: {"reference": "new-" + k} for k in ("management", "driver")}
        cp, dp, data = d.runtime_upgrade_patches(cm, dep, images)
        self.assertEqual((cm, dep), original)
        self.assertEqual(
            cp[0], {"op": "test", "path": "/metadata/resourceVersion", "value": "42"}
        )
        self.assertEqual(
            dp[0], {"op": "test", "path": "/metadata/resourceVersion", "value": "71"}
        )
        self.assertEqual(cp[1]["path"], "/data/MANAGEMENT_DRIVER_IMAGE")
        self.assertEqual(dp[1]["path"], "/spec/template/spec/containers/1/image")
        self.assertEqual(
            dp[2]["value"],
            {
                "keep": "original",
                "frontier.dev/config-sha256": d.canonical_object_sha256(data),
            },
        )
        self.assertEqual(
            {k: v for k, v in data.items() if k != "MANAGEMENT_DRIVER_IMAGE"},
            {k: v for k, v in cm["data"].items() if k != "MANAGEMENT_DRIVER_IMAGE"},
        )

    def test_non_preview_profile_is_rejected(self):
        with self.assertRaisesRegex(d.DeployError, "local-k8s"):
            d.runtime_upgrade_patches(
                {"data": {"MANAGEMENT_DRIVER_BACKENDS": "aws"}}, {}, {}
            )

    def test_controller_patch_preserves_modal_environment_and_other_settings(self):
        dep = {
            "metadata": {"resourceVersion": "73"},
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "controller",
                                "image": "old",
                                "env": [
                                    {"name": "FRONTIER_MODAL_GPU", "value": "true"},
                                    {
                                        "name": "FRONTIER_MODAL_SECRET_NAME",
                                        "value": "existing",
                                    },
                                ],
                                "volumeMounts": [
                                    {"name": "existing", "mountPath": "/existing"}
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "existing", "secret": {"secretName": "existing"}}
                        ],
                    }
                },
            },
        }
        original = copy.deepcopy(dep)
        patch, expected = d.storage_controller_upgrade_patch(
            dep, "new", "172.20.0.50/32"
        )
        self.assertEqual(dep, original)
        self.assertEqual(patch[0]["value"], "73")
        container = expected["template"]["spec"]["containers"][0]
        self.assertEqual(
            container["env"][:2],
            original["spec"]["template"]["spec"]["containers"][0]["env"],
        )
        self.assertEqual(
            container["env"][-1],
            {"name": "FRONTIER_REGISTRY_SERVICE_CIDR", "value": "172.20.0.50/32"},
        )
        self.assertEqual(
            expected["template"]["spec"]["volumes"],
            original["spec"]["template"]["spec"]["volumes"],
        )
        with self.assertRaises(d.DeployError):
            d.storage_controller_upgrade_patch(dep, "new", "0.0.0.0/0")
