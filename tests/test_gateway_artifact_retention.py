from __future__ import annotations

#
import deployment_test_guard  # noqa: F401  # isort: skip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from preview import infra
from test_frontier_deploy import ROOT, Fixture, frontier_deploy


class RetainedGatewayArtifactTests(unittest.TestCase):
    def render(self, simulator=False):
        config = frontier_deploy.validate_config(
            json.loads((ROOT / "config/site.example.json").read_text())
        )
        return json.loads(
            infra.render_runtime_manifest(
                config,
                Fixture.outputs(config),
                Fixture.images(config),
                ["172.20.0.1/32"],
                simulator_bootstrap=simulator,
            )
        )["items"]

    def init(self):
        items = self.render()
        pod = next(
            x
            for x in items
            if x["kind"] == "Deployment"
            and x["metadata"]["name"] == "frontier-management"
        )["spec"]["template"]["spec"]
        return pod, pod["initContainers"][0]

    def test_retained_root_uses_management_data_not_driver_objects(self):
        items = self.render()
        settings = next(
            x["data"]
            for x in items
            if x["kind"] == "ConfigMap"
            and x["metadata"]["name"] == "frontier-management-config"
        )
        self.assertEqual(
            settings["MANAGEMENT_GATEWAY_ARTIFACT_SOURCE_ROOT"],
            "/var/lib/frontier-local/artifacts",
        )
        self.assertEqual(
            settings["MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT"],
            "/var/lib/frontier/gateway-terminal-retained",
        )
        pod, init = self.init()
        self.assertEqual(init["image"], pod["containers"][0]["image"])
        self.assertEqual(init["command"][:2], ["sh", "-ec"])
        for check in (
            'test ! -L "$retained"',
            'test -d "$retained"',
            'test "$(stat -c %u "$retained")" = 65532',
            'test "$(stat -c %a "$retained")" = 700',
        ):
            self.assertIn(check, init["command"][2])
        self.assertEqual(
            init["volumeMounts"], [{"name": "data", "mountPath": "/var/lib/frontier"}]
        )
        volumes = {x["name"]: x for x in pod["volumes"]}
        self.assertEqual(
            volumes["data"]["persistentVolumeClaim"]["claimName"],
            "frontier-management-data",
        )
        self.assertNotEqual(
            volumes["data"]["persistentVolumeClaim"]["claimName"],
            volumes["objects"]["persistentVolumeClaim"]["claimName"],
        )
        mounts = {
            x["name"]: x
            for x in pod["containers"][0]["volumeMounts"]
            if "subPath" not in x
        }
        self.assertTrue(mounts["objects"]["readOnly"])
        simulator = self.render(True)
        cm = next(
            x["data"]
            for x in simulator
            if x["kind"] == "ConfigMap"
            and x["metadata"]["name"] == "frontier-management-config"
        )
        self.assertNotIn("MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT", cm)
        deployment = next(
            x
            for x in simulator
            if x["kind"] == "Deployment"
            and x["metadata"]["name"] == "frontier-management"
        )
        self.assertNotIn("initContainers", deployment["spec"]["template"]["spec"])

    def run_init(self, root):
        command = list(self.init()[1]["command"])
        command[-1] = command[-1].replace(
            "/var/lib/frontier/gateway-terminal-retained", str(root)
        )
        command[-1] = command[-1].replace("= 65532", "= " + str(os.getuid()))
        return subprocess.run(command, capture_output=True, text=True)

    @unittest.skipUnless(sys.platform == "linux", "init container uses Linux stat")
    def test_initialization_creates_private_directory_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "retained"
            result = self.run_init(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            inode = root.stat().st_ino
            result = self.run_init(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(root.stat().st_ino, inode)

    @unittest.skipUnless(sys.platform == "linux", "init container uses Linux stat")
    def test_existing_symlink_is_rejected_without_changing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "target"
            target.mkdir(mode=0o755)
            root = Path(directory).resolve() / "retained"
            root.symlink_to(target, target_is_directory=True)
            result = self.run_init(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(root.is_symlink())
            self.assertEqual(target.stat().st_mode & 0o777, 0o755)

    @unittest.skipUnless(sys.platform == "linux", "init container uses Linux stat")
    def test_existing_shared_directory_is_rejected_without_chmod(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "retained"
            root.mkdir(mode=0o755)
            result = self.run_init(root)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(root.stat().st_mode & 0o777, 0o755)
