from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import ast
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/storage-controller"))
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_deploy
from preview import cli
from preview import images as image_ops


def load_storage_controller(*, enabled: bool = True):
    spec = importlib.util.spec_from_file_location(
        "frontier_storage_controller",
        ROOT / "runtime/storage-controller/controller.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    environment = {
        "KUBERNETES_SERVICE_HOST": "172.20.0.1",
        "KUBERNETES_SERVICE_PORT_HTTPS": "443",
        "FRONTIER_EFS_OBJECT_VOLUME_HANDLE": (
            "fs-0123456789abcdef0::fsap-0123456789abcdef0"
        ),
        "FRONTIER_OBJECT_STORE_GIB": "20",
        "FRONTIER_RECONCILE_SECONDS": "1",
        "FRONTIER_AGENT_EXTERNAL_EGRESS": "true" if enabled else "false",
    }
    if enabled:
        environment["FRONTIER_DNS_CIDR"] = "172.20.0.10/32"
    with (
        mock.patch.dict(os.environ, environment),
        mock.patch(
            "ssl.create_default_context", return_value=mock.sentinel.ssl_context
        ),
    ):
        spec.loader.exec_module(module)
    return module


class PreflightToolTests(unittest.TestCase):
    def test_registry_copy_tool_is_optional_and_honors_override(self):
        for found in (None, "/tools/custom-crane"):
            with (
                self.subTest(found=found),
                mock.patch.dict(
                    os.environ, {"FRONTIER_REGISTRY_COPY_TOOL": "custom-crane"}
                ),
                mock.patch.object(cli, "read_and_validate", return_value=({}, None)),
                mock.patch.object(cli, "modal_credentials"),
                mock.patch.object(
                    cli, "executable", side_effect=lambda name: "/tools/" + name
                ),
                mock.patch.object(cli.shutil, "which", return_value=found) as which,
                mock.patch.object(cli, "aws_identity") as identity,
                mock.patch("builtins.print") as output,
            ):
                cli.preflight_command(Path("/unused/site.json"))
                which.assert_called_once_with("custom-crane")
                identity.assert_called_once_with({})
                self.assertEqual(
                    [call.args[0] for call in output.call_args_list],
                    [
                        "terraform: /tools/terraform",
                        "aws: /tools/aws",
                        "kubectl: /tools/kubectl",
                        "docker: /tools/docker",
                        "registry copy tool (optional; runtime upgrades): "
                        + (found or "missing: custom-crane"),
                    ],
                )


class Fixture:
    @staticmethod
    def outputs(config: dict) -> dict:
        return {
            "account_id": config["site"]["expected_account_id"],
            "aws_region": config["site"]["aws_region"],
            "cluster_name": "frontiercs2-preview",
            "cluster_endpoint": "https://cluster.example.eks.amazonaws.com",
            "vpc_id": "vpc-0123456789abcdef0",
            "vpc_cidr": config["network"]["vpc_cidr"],
            "service_ipv4_cidr": config["network"]["service_ipv4_cidr"],
            "nat_gateway_public_ips": ["8.8.8.8"],
            "platform_kms_key_arn": "arn:aws:kms:us-west-1:123456789012:key/00000000-0000-0000-0000-000000000001",
            "artifact_bucket": "frontiercs2-preview-artifacts",
            "management_artifact_reader_role_arn": "arn:aws:iam::123456789012:role/frontiercs2-preview-management-artifact-reader",
            "management_artifact_reader_role_id": "AROA1234567890ABCDEF",
            "system_node_group_name": "frontiercs2-preview-system",
            "cpu_node_group_name": "frontiercs2-preview-cpu",
            "efs_file_system_id": "fs-0123456789abcdef0",
            "efs_object_store_access_point_id": "fsap-0123456789abcdef0",
            "efs_registry_access_point_id": "fsap-1123456789abcdef0",
        }

    @staticmethod
    def images(config: dict) -> dict:
        endpoint = f"{config['network']['registry_cluster_ip']}:5000"
        result = {}
        for index, name in enumerate(frontier_deploy.RUNTIME_IMAGE_NAMES):
            digest = "sha256:" + format(index + 1, "x") * 64
            repository = f"{endpoint}/frontier/{name.replace('_', '-')}"
            result[name] = {
                "tag": f"{repository}:src-0123456789abcdef",
                "digest": digest,
                "reference": f"{repository}@{digest}",
                "source_sha256": "sha256:" + "a" * 64,
            }
        return result


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = json.loads((ROOT / "config/site.example.json").read_text())

    def test_active_driver_cap_matches_management_range(self) -> None:
        import jsonschema

        schema = json.loads((ROOT / "config/site.schema.json").read_text())
        cap_schema = schema["properties"]["execution"]["properties"][
            "max_active_drivers"
        ]
        for value in (1, 32, 35, 1000):
            with self.subTest(valid=value):
                candidate = copy.deepcopy(self.raw)
                candidate["execution"]["max_active_drivers"] = value
                self.assertEqual(
                    frontier_deploy.validate_config(candidate)["execution"][
                        "max_active_drivers"
                    ],
                    value,
                )
                jsonschema.validate(value, cap_schema)
        for value in (0, -1, 1001, 35.5, True, "35", None):
            with self.subTest(invalid=value):
                candidate = copy.deepcopy(self.raw)
                candidate["execution"]["max_active_drivers"] = value
                with self.assertRaisesRegex(
                    frontier_deploy.DeployError, "max_active_drivers"
                ):
                    frontier_deploy.validate_config(candidate)
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(value, cap_schema)

    def test_optional_model_domains_accept_hosts_and_reject_urls(self):
        config = copy.deepcopy(self.raw)
        config["execution"]["model_domains"] = ["api.moonshot.ai", "openrouter.ai"]
        frontier_deploy.validate_config(config)
        config["execution"]["model_domains"] = []
        frontier_deploy.validate_config(config)
        for invalid in [
            "api.moonshot.ai",
            ["https://api.moonshot.ai/v1"],
            ["host:443"],
            ["*.moonshot.ai"],
            [None],
        ]:
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(frontier_deploy.DeployError, "model_domains"),
            ):
                config["execution"]["model_domains"] = invalid
                frontier_deploy.validate_config(config)

    def test_example_enables_cpu_preview_and_admin_configured_litellm(self) -> None:
        config = frontier_deploy.validate_config(copy.deepcopy(self.raw))
        self.assertEqual(config["management"]["profile"], "executable-preview")
        self.assertEqual(
            config["application"]["repository"], frontier_deploy.APPLICATION_REPOSITORY
        )
        self.assertRegex(config["application"]["commit"], r"^[0-9a-f]{40}$")
        self.assertIs(config["application"]["require_source"], False)
        self.assertEqual(
            config["features"],
            {"run_execution": True, "litellm": True, "modal_gpu": False,
             "task_model_aliases": ["progent-agent-gpt-4o-mini"]},
        )
        self.assertTrue(config["litellm"]["store_model_in_db"])
        self.assertIs(config["execution"]["agent_external_egress"], False)
        self.assertGreaterEqual(config["compute"]["cpu"]["desired_size"], 1)
        for name in ("registry", "management", "driver", "litellm", "postgres"):
            self.assertRegex(
                config["images"][name]["reference"], r"@sha256:[0-9a-f]{64}$"
            )
            self.assertNotIn(":latest", config["images"][name]["reference"])

    def test_management_and_driver_tags_must_match_application_commit(self) -> None:
        for name in ("management", "driver"):
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.raw)
                candidate["images"][name]["tag"] = "sha-" + "f" * 40
                with self.assertRaisesRegex(
                    frontier_deploy.DeployError, "tag must equal sha-"
                ):
                    frontier_deploy.validate_config(candidate)

    def test_runtime_sources_are_explicit_and_each_tag_is_bound(self) -> None:
        frontier_deploy.validate_config(copy.deepcopy(self.raw))
        # Exercise split-source releases even when the current pair shares a commit.
        raw = copy.deepcopy(self.raw)
        raw["application"]["runtime_source_commits"]["driver"] = "d" * 40
        raw["images"]["driver"]["tag"] = "sha-" + "d" * 40
        config = frontier_deploy.validate_config(copy.deepcopy(raw))
        self.assertNotEqual(config["application"]["runtime_source_commits"]["driver"], config["application"]["commit"])
        for role in ("management", "driver"):
            self.assertEqual(config["images"][role]["tag"], "sha-" + config["application"]["runtime_source_commits"][role])
        legacy = copy.deepcopy(raw)
        del legacy["application"]["runtime_source_commits"]
        legacy["images"]["driver"]["tag"] = "sha-" + legacy["application"]["commit"]
        frontier_deploy.validate_config(legacy)
        for sources in ({"management": "f" * 40, "driver": "e" * 40},
                        {"management": self.raw["application"]["commit"]},
                        {"management": self.raw["application"]["commit"], "driver": "main"}):
            candidate = copy.deepcopy(raw)
            candidate["application"]["runtime_source_commits"] = sources
            with self.assertRaises(frontier_deploy.DeployError):
                frontier_deploy.validate_config(candidate)
        candidate = copy.deepcopy(raw)
        del candidate["application"]["runtime_source_commits"]
        with self.assertRaises(frontier_deploy.DeployError):
            frontier_deploy.validate_config(candidate)

    def test_management_and_driver_image_records_are_required(self) -> None:
        for name in ("management", "driver"):
            with self.subTest(name=name):
                candidate = copy.deepcopy(self.raw)
                del candidate["images"][name]
                with self.assertRaisesRegex(
                    frontier_deploy.DeployError, f"missing=\\['{name}'\\]"
                ):
                    frontier_deploy.validate_config(candidate)

    def test_application_require_source_is_required(self) -> None:
        candidate = copy.deepcopy(self.raw)
        del candidate["application"]["require_source"]
        with self.assertRaisesRegex(
            frontier_deploy.DeployError, "missing=\\['require_source'\\]"
        ):
            frontier_deploy.validate_config(candidate)

    def test_mutable_platform_image_or_application_commit_is_rejected(self) -> None:
        for field, value in (
            ("tag", "latest"),
            ("digest", "sha256:bad"),
            ("repository", "repo@digest"),
        ):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.raw)
                candidate["images"]["registry"][field] = value
                with self.assertRaises(frontier_deploy.DeployError):
                    frontier_deploy.validate_config(candidate)
        candidate = copy.deepcopy(self.raw)
        candidate["application"]["commit"] = "main"
        with self.assertRaisesRegex(frontier_deploy.DeployError, "40-character"):
            frontier_deploy.validate_config(candidate)

    def test_secret_fields_are_rejected_at_any_depth(self) -> None:
        candidate = copy.deepcopy(self.raw)
        candidate["execution"]["smoke"]["aws_secret_access_key"] = "forbidden"
        with self.assertRaisesRegex(
            frontier_deploy.DeployError, "secret-bearing field"
        ):
            frontier_deploy.validate_config(candidate)

    def test_agent_external_egress_is_an_explicit_boolean(self) -> None:
        missing = copy.deepcopy(self.raw)
        del missing["execution"]["agent_external_egress"]
        with self.assertRaisesRegex(
            frontier_deploy.DeployError, "missing=\\['agent_external_egress'\\]"
        ):
            frontier_deploy.validate_config(missing)
        for value in ("true", 1, None):
            with self.subTest(value=value):
                invalid = copy.deepcopy(self.raw)
                invalid["execution"]["agent_external_egress"] = value
                with self.assertRaisesRegex(
                    frontier_deploy.DeployError, "must be a boolean"
                ):
                    frontier_deploy.validate_config(invalid)
        disabled = copy.deepcopy(self.raw)
        disabled["execution"]["agent_external_egress"] = False
        self.assertIs(
            frontier_deploy.validate_config(disabled)["execution"][
                "agent_external_egress"
            ],
            False,
        )

        schema = json.loads((ROOT / "config/site.schema.json").read_text())
        self.assertIn(
            "agent_external_egress", schema["properties"]["execution"]["required"]
        )
        self.assertEqual(
            schema["properties"]["execution"]["properties"]["agent_external_egress"],
            {"type": "boolean"},
        )

    def test_operator_ingress_is_exact_host_only(self) -> None:
        for cidr in ("0.0.0.0/0", "198.51.100.0/24", "2001:db8::1/128"):
            candidate = copy.deepcopy(self.raw)
            candidate["network"]["operator_cidrs"] = [cidr]
            with self.assertRaises(frontier_deploy.DeployError):
                frontier_deploy.validate_config(candidate)

    def test_service_network_must_be_disjoint_and_fixed_ips_must_be_safe(self) -> None:
        candidates = []
        same = copy.deepcopy(self.raw)
        same["network"]["service_ipv4_cidr"] = same["network"]["vpc_cidr"]
        same["network"]["management_cluster_ip"] = "10.42.0.20"
        same["network"]["registry_cluster_ip"] = "10.42.0.50"
        candidates.append(same)
        reserved = copy.deepcopy(self.raw)
        reserved["network"]["management_cluster_ip"] = "172.20.0.1"
        candidates.append(reserved)
        duplicate = copy.deepcopy(self.raw)
        duplicate["network"]["management_cluster_ip"] = duplicate["network"][
            "registry_cluster_ip"
        ]
        candidates.append(duplicate)
        duplicate_litellm = copy.deepcopy(self.raw)
        duplicate_litellm["network"]["litellm_cluster_ip"] = duplicate_litellm[
            "network"
        ]["management_cluster_ip"]
        candidates.append(duplicate_litellm)
        for candidate in candidates:
            with self.assertRaises(frontier_deploy.DeployError):
                frontier_deploy.validate_config(candidate)

    def test_cpu_group_cannot_be_cold_for_executable_profile(self) -> None:
        candidate = copy.deepcopy(self.raw)
        candidate["compute"]["cpu"]["min_size"] = 0
        candidate["compute"]["cpu"]["desired_size"] = 0
        with self.assertRaises(frontier_deploy.DeployError):
            frontier_deploy.validate_config(candidate)

    def test_feature_profile_fails_closed(self) -> None:
        without_litellm = copy.deepcopy(self.raw)
        without_litellm["features"]["litellm"] = False
        with self.assertRaisesRegex(frontier_deploy.DeployError, "LiteLLM enabled"):
            frontier_deploy.validate_config(without_litellm)
        with_modal = copy.deepcopy(self.raw)
        with_modal["features"]["modal_gpu"] = True
        del with_modal["modal"]
        with self.assertRaisesRegex(frontier_deploy.DeployError, "modal"):
            frontier_deploy.validate_config(with_modal)

    def test_litellm_ui_contract_is_closed(self) -> None:
        for field, value in (
            ("exposure", "LoadBalancer"),
            ("ui_access", "all"),
            ("store_model_in_db", False),
        ):
            candidate = copy.deepcopy(self.raw)
            candidate["litellm"][field] = value
            with self.assertRaises(frontier_deploy.DeployError):
                frontier_deploy.validate_config(candidate)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema_version":"one","schema_version":"two"}')
            with self.assertRaisesRegex(
                frontier_deploy.DeployError, "duplicate JSON key"
            ):
                frontier_deploy.load_json(path)


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = json.loads((ROOT / "config/site.example.json").read_text())
        self.config = frontier_deploy.validate_config(raw)
        self.outputs = Fixture.outputs(self.config)
        self.images = Fixture.images(self.config)

    def test_bootstrap_is_registry_first_and_efs_persistent(self) -> None:
        first = frontier_deploy.render_bootstrap_manifest(self.config, self.outputs)
        second = frontier_deploy.render_bootstrap_manifest(self.config, self.outputs)
        self.assertEqual(first, second)
        value = json.loads(first)
        kinds = [(item["kind"], item["metadata"]["name"]) for item in value["items"]]
        self.assertIn(("PersistentVolume", "frontier-registry"), kinds)
        self.assertIn(("Deployment", "frontier-registry"), kinds)
        self.assertNotIn(("Deployment", "frontier-management"), kinds)
        text = first.decode()
        self.assertIn("efs.csi.aws.com", text)
        self.assertIn(self.config["images"]["registry"]["reference"], text)
        self.assertNotIn('kind":"Secret', text)

    def test_registry_admits_only_owned_same_namespace_task_seed_pods(self) -> None:
        value = json.loads(frontier_deploy.render_bootstrap_manifest(self.config, self.outputs))
        policy = next(item for item in value["items"]
                      if item["metadata"]["name"] == "frontier-registry-boundary")
        seed_rules = [rule for rule in policy["spec"]["ingress"]
                      if any(peer.get("podSelector", {}).get("matchLabels", {}).get(
                          "app.kubernetes.io/name") == "frontier-task-seed"
                             for peer in rule["from"])]
        self.assertEqual(len(seed_rules), 1)
        self.assertEqual(policy["metadata"]["namespace"], "frontier-system")
        self.assertEqual(seed_rules[0], {
            "from": [{"podSelector": {"matchLabels": {
                "app.kubernetes.io/name": "frontier-task-seed",
                "app.kubernetes.io/part-of": "frontiercs2",
                "frontier.dev/owner": "frontier-deploy",
                "frontier.dev/site": self.config["site"]["name"],
            }}}],
            "ports": [{"protocol": "TCP", "port": 5000}],
        })
        self.assertEqual(policy["spec"]["egress"], [])
        egress = next(item for item in value["items"]
                      if item["metadata"]["name"] == "frontier-task-seed-registry-egress")
        self.assertEqual(egress["metadata"]["namespace"], "frontier-system")
        self.assertEqual(egress["spec"], {
            "podSelector": seed_rules[0]["from"][0]["podSelector"],
            "policyTypes": ["Egress"],
            "egress": [{
                "to": [
                    {"podSelector": {"matchLabels": {"frontier.dev/network-role": "registry"}}},
                    {"ipBlock": {"cidr": self.config["network"]["registry_cluster_ip"] + "/32"}},
                ],
                "ports": [{"protocol": "TCP", "port": 5000}],
            }],
        })

    def test_runtime_contains_management_controller_seed_and_run_enabled_config(
        self,
    ) -> None:
        encoded = frontier_deploy.render_runtime_manifest(
            self.config, self.outputs, self.images, ["172.20.0.1/32", "10.42.1.20/32"]
        )
        value = json.loads(encoded)
        identities = {
            (item["kind"], item["metadata"]["name"]) for item in value["items"]
        }
        dataset_config_name, seed_job_name = frontier_deploy.dataset_seed_names()
        for identity in (
            ("Deployment", "frontier-management"),
            ("Deployment", "frontier-storage-controller"),
            ("ConfigMap", dataset_config_name),
            ("Job", seed_job_name),
            ("ClusterRole", "frontier-management-preview"),
        ):
            self.assertIn(identity, identities)
        management_config = next(
            item
            for item in value["items"]
            if item["kind"] == "ConfigMap"
            and item["metadata"]["name"] == "frontier-management-config"
        )
        self.assertEqual(
            management_config["data"]["MANAGEMENT_DRIVER_MODE"], "kubernetes"
        )
        self.assertEqual(
            management_config["data"]["MANAGEMENT_DRIVER_BACKENDS"], "local-k8s"
        )
        self.assertEqual(
            management_config["data"]["MANAGEMENT_DRIVER_IMAGE"],
            self.images["driver"]["reference"],
        )
        self.assertEqual(
            management_config["data"]["MANAGEMENT_ARTIFACT_LOCAL_ROOT"],
            "/var/lib/frontier-local",
        )
        self.assertNotIn(
            "MANAGEMENT_ARTIFACT_PRESIGN_ENABLED", management_config["data"]
        )
        deployment_spec = next(
            item
            for item in value["items"]
            if item["kind"] == "Deployment"
            and item["metadata"]["name"] == "frontier-management"
        )["spec"]["template"]["spec"]
        management = next(c for c in deployment_spec['containers'] if c['name'] == 'management')
        self.assertEqual(management['resources'], {
            'requests': {'cpu': '100m', 'memory': '4Gi', 'ephemeral-storage': '34Gi'},
            'limits': {'cpu': '1', 'memory': '4Gi', 'ephemeral-storage': '34Gi'},
        })
        tmp = next(volume for volume in deployment_spec['volumes'] if volume['name'] == 'tmp')
        self.assertEqual(tmp['emptyDir'], {'sizeLimit': '34Gi'})
        self.assertEqual(self.config['compute']['system']['instance_types'], ['m6i.xlarge'])
        self.assertEqual(self.config['compute']['system']['min_size'], 1)
        self.assertEqual(self.config['compute']['system']['desired_size'], 1)
        objects_volume = next(
            volume
            for volume in deployment_spec["volumes"]
            if volume["name"] == "objects"
        )
        self.assertEqual(
            objects_volume["persistentVolumeClaim"],
            {"claimName": "frontier-object-seed"},
        )
        objects_mount = next(
            mount
            for mount in deployment_spec["containers"][0]["volumeMounts"]
            if mount["name"] == "objects"
        )
        self.assertEqual(objects_mount["mountPath"], "/var/lib/frontier-local")
        self.assertTrue(objects_mount["readOnly"])
        self.assertEqual(
            sum(
                v.get("persistentVolumeClaim", {}).get("claimName")
                == "frontier-object-seed"
                for v in deployment_spec["volumes"]
            ),
            1,
        )
        self.assertIn(
            {
                "name": "objects",
                "mountPath": "/var/lib/frontier-local/artifacts",
                "subPath": "artifacts",
            },
            deployment_spec["containers"][0]["volumeMounts"],
        )
        service_account = next(
            item
            for item in value["items"]
            if item["kind"] == "ServiceAccount"
            and item["metadata"]["name"] == "frontier-management"
        )
        self.assertEqual(
            service_account["metadata"]["annotations"]["eks.amazonaws.com/role-arn"],
            self.outputs["management_artifact_reader_role_arn"],
        )
        deployment = next(
            item
            for item in value["items"]
            if item["kind"] == "Deployment"
            and item["metadata"]["name"] == "frontier-management"
        )
        self.assertEqual(
            deployment["spec"]["template"]["spec"]["containers"][0]["image"],
            self.images["management"]["reference"],
        )
        self.assertIn(
            "172.20.0.50:5000",
            management_config["data"]["MANAGEMENT_DRIVER_LOCAL_REGISTRY_MAP_JSON"],
        )
        text = encoded.decode()
        self.assertNotIn(":latest", text)
        self.assertNotIn('kind":"Secret', text)

    def test_modal_observer_rbac_is_scoped_to_configured_secret_and_management(
        self,
    ) -> None:
        for simulator in (False, True):
            for secret_name in ("frontier-modal-credentials", "custom-site-modal"):
                with self.subTest(simulator=simulator, secret_name=secret_name):
                    self.config["features"]["modal_gpu"] = True
                    self.config["modal"]["credentials_secret"] = secret_name
                    items = json.loads(
                        frontier_deploy.render_runtime_manifest(
                            self.config,
                            self.outputs,
                            self.images,
                            ["172.20.0.1/32"],
                            simulator_bootstrap=simulator,
                        )
                    )["items"]
                    config_data = next(
                        item["data"]
                        for item in items
                        if item["kind"] == "ConfigMap"
                        and item["metadata"]["name"] == "frontier-management-config"
                    )
                    role_name = "frontier-management-modal-observer"
                    roles = [
                        item
                        for item in items
                        if item["kind"] == "Role"
                        and item["metadata"]["name"] == role_name
                    ]
                    self.assertEqual(len(roles), 1)
                    self.assertEqual(
                        roles[0]["metadata"]["namespace"],
                        config_data["MANAGEMENT_NAMESPACE"],
                    )
                    self.assertEqual(
                        config_data["MANAGEMENT_MODAL_CREDENTIALS_SECRET"], secret_name
                    )
                    self.assertEqual(
                        roles[0]["rules"],
                        [
                            {
                                "apiGroups": [""],
                                "resources": ["secrets"],
                                "verbs": ["get"],
                                "resourceNames": [secret_name],
                            }
                        ],
                    )
                    bindings = [
                        item
                        for item in items
                        if item["kind"] in ("RoleBinding", "ClusterRoleBinding")
                        and item["roleRef"]["name"] == role_name
                    ]
                    self.assertEqual(len(bindings), 1)
                    binding = bindings[0]
                    self.assertEqual(binding["kind"], "RoleBinding")
                    self.assertEqual(
                        binding["metadata"]["namespace"],
                        config_data["MANAGEMENT_NAMESPACE"],
                    )
                    self.assertEqual(
                        binding["roleRef"],
                        {
                            "apiGroup": "rbac.authorization.k8s.io",
                            "kind": "Role",
                            "name": role_name,
                        },
                    )
                    self.assertEqual(
                        binding["subjects"],
                        [
                            {
                                "kind": "ServiceAccount",
                                "name": "frontier-management",
                                "namespace": config_data["MANAGEMENT_NAMESPACE"],
                            }
                        ],
                    )
                    cluster_role = next(
                        item
                        for item in items
                        if item["kind"] == "ClusterRole"
                        and item["metadata"]["name"] == "frontier-management-preview"
                    )
                    self.assertFalse(
                        any(
                            "secrets" in rule["resources"]
                            for rule in cluster_role["rules"]
                        )
                    )

    def test_cpu_site_has_no_modal_observer_rbac(self) -> None:
        self.config["features"]["modal_gpu"] = False
        self.config.pop("modal", None)
        for simulator in (False, True):
            with self.subTest(simulator=simulator):
                items = json.loads(
                    frontier_deploy.render_runtime_manifest(
                        self.config,
                        self.outputs,
                        self.images,
                        ["172.20.0.1/32"],
                        simulator_bootstrap=simulator,
                    )
                )["items"]
                self.assertFalse(
                    any(
                        item["metadata"]["name"] == "frontier-management-modal-observer"
                        for item in items
                    )
                )

    def test_storage_controller_renders_agent_egress_env_and_narrow_rbac(self) -> None:
        self.config["features"]["modal_gpu"] = True
        self.config["execution"]["agent_external_egress"] = True
        encoded = frontier_deploy.render_runtime_manifest(
            self.config, self.outputs, self.images, ["172.20.0.1/32", "10.42.1.20/32"]
        )
        items = json.loads(encoded)["items"]
        deployment = next(
            item
            for item in items
            if item["kind"] == "Deployment"
            and item["metadata"]["name"] == "frontier-storage-controller"
        )
        environment = {
            entry["name"]: entry["value"]
            for entry in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        self.assertEqual(environment["FRONTIER_AGENT_EXTERNAL_EGRESS"], "true")
        self.assertEqual(environment["FRONTIER_DNS_CIDR"], "172.20.0.10/32")
        self.assertEqual(environment["FRONTIER_MODAL_GPU"], "true")
        pod = deployment["spec"]["template"]["spec"]
        self.assertEqual(pod["securityContext"]["fsGroup"], 65532)
        self.assertTrue(pod["containers"][0]["volumeMounts"][0]["readOnly"])
        self.assertEqual(
            pod["volumes"][0]["secret"]["secretName"],
            self.config["modal"]["credentials_secret"],
        )
        config_data = next(
            item["data"]
            for item in items
            if item["kind"] == "ConfigMap"
            and item["metadata"]["name"] == "frontier-management-config"
        )
        self.assertEqual(config_data["MANAGEMENT_MODAL_INBOUND_CIDRS"], "8.8.8.8/32")
        self.assertEqual(config_data["MANAGEMENT_MODAL_ECR_PULL_SECRET_REF"], "")
        self.assertEqual(
            config_data["MANAGEMENT_MODAL_CREDENTIALS_SECRET"],
            self.config["modal"]["credentials_secret"],
        )

        role = next(
            item
            for item in items
            if item["kind"] == "ClusterRole"
            and item["metadata"]["name"] == "frontier-storage-controller"
        )
        network_policy_rules = [
            rule for rule in role["rules"] if rule["apiGroups"] == ["networking.k8s.io"]
        ]
        self.assertEqual(
            network_policy_rules,
            [
                {
                    "apiGroups": ["networking.k8s.io"],
                    "resources": ["networkpolicies"],
                    "verbs": ["create", "delete", "get", "update"],
                }
            ],
        )

        boundary = next(
            item
            for item in items
            if item["kind"] == "NetworkPolicy"
            and item["metadata"]["name"] == "frontier-storage-controller-boundary"
        )
        self.assertEqual(
            boundary["spec"],
            {
                "podSelector": {
                    "matchLabels": {"frontier.dev/network-role": "storage-controller"}
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [],
                "egress": [
                    {
                        "to": [
                            {"ipBlock": {"cidr": "172.20.0.1/32"}},
                            {"ipBlock": {"cidr": "10.42.1.20/32"}},
                        ],
                        "ports": [{"protocol": "TCP", "port": 443}],
                    }
                ],
            },
        )

        disabled = copy.deepcopy(self.config)
        disabled["execution"]["agent_external_egress"] = False
        disabled_items = json.loads(
            frontier_deploy.render_runtime_manifest(
                disabled, self.outputs, self.images, ["172.20.0.1/32"]
            )
        )["items"]
        disabled_deployment = next(
            item
            for item in disabled_items
            if item["kind"] == "Deployment"
            and item["metadata"]["name"] == "frontier-storage-controller"
        )
        disabled_environment = {
            entry["name"]: entry["value"]
            for entry in disabled_deployment["spec"]["template"]["spec"]["containers"][
                0
            ]["env"]
        }
        self.assertEqual(
            disabled_environment["FRONTIER_AGENT_EXTERNAL_EGRESS"], "false"
        )

    def test_runtime_image_map_is_exact_and_digest_pinned(self) -> None:
        validated = frontier_deploy.validate_runtime_images(self.images, self.config)
        self.assertEqual(set(validated), set(frontier_deploy.RUNTIME_IMAGE_NAMES))
        broken = copy.deepcopy(self.images)
        broken["agent_base"]["reference"] = broken["agent_base"]["tag"]
        with self.assertRaises(frontier_deploy.DeployError):
            frontier_deploy.validate_runtime_images(broken, self.config)

    def test_litellm_ui_postgres_and_network_boundary_are_rendered_without_secrets(
        self,
    ) -> None:
        encoded = frontier_deploy.render_litellm_manifest(
            self.config, self.outputs, self.images
        )
        value = json.loads(encoded)
        identities = {
            (item["kind"], item["metadata"]["name"]) for item in value["items"]
        }
        for identity in (
            ("Namespace", "frontier-litellm"),
            ("StorageClass", "frontier-litellm-gp3"),
            ("StatefulSet", "frontier-litellm-postgres"),
            ("Deployment", "frontier-litellm"),
            ("Service", "frontier-litellm"),
            ("NetworkPolicy", "frontier-litellm-default-deny"),
        ):
            self.assertIn(identity, identities)
        config_map = next(
            item for item in value["items"] if item["kind"] == "ConfigMap"
        )
        proxy_config = json.loads(config_map["data"]["config.yaml"])
        self.assertEqual(proxy_config["model_list"], [])
        self.assertTrue(proxy_config["general_settings"]["store_model_in_db"])
        self.assertEqual(
            proxy_config["general_settings"]["ui_access_mode"], "admin_only"
        )
        self.assertEqual(proxy_config["router_settings"]["num_retries"], 0)
        self.assertEqual(proxy_config["router_settings"]["retry_after"], 5)
        self.assertEqual(
            proxy_config["router_settings"]["retry_policy"],
            {
                "AuthenticationErrorRetries": 16,
                "RateLimitErrorRetries": 16,
                "InternalServerErrorRetries": 16,
                "TimeoutErrorRetries": 16,
            },
        )
        self.assertEqual(proxy_config["router_settings"]["fallbacks"], [])
        self.assertEqual(proxy_config["router_settings"]["context_window_fallbacks"], [])
        deployment = next(
            item for item in value["items"] if item["kind"] == "Deployment"
        )
        self.assertEqual(
            deployment["spec"]["template"]["spec"]["containers"][0]["image"],
            self.images["litellm"]["reference"],
        )
        environment = {
            item["name"]: item
            for item in deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        self.assertEqual(environment["UI_USERNAME"]["value"], "admin")
        self.assertEqual(
            environment["UI_PASSWORD"]["valueFrom"]["secretKeyRef"]["key"],
            "LITELLM_MASTER_KEY",
        )
        statefulset = next(
            item for item in value["items"] if item["kind"] == "StatefulSet"
        )
        self.assertEqual(
            statefulset["spec"]["template"]["spec"]["containers"][0]["image"],
            self.images["postgres"]["reference"],
        )
        self.assertEqual(
            statefulset["spec"]["volumeClaimTemplates"][0]["spec"]["storageClassName"],
            "frontier-litellm-gp3",
        )
        text = encoded.decode()
        self.assertNotIn('"kind":"Secret"', text)
        self.assertNotIn("OPENROUTER_API_KEY", text)
        self.assertNotIn(":latest", text)

    def test_litellm_secret_state_is_generated_once_and_owner_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory) / "site"
            first_secret, first_admin = frontier_deploy.ensure_litellm_secret_files(
                self.config, private
            )
            secret_bytes, admin_bytes = (
                first_secret.read_bytes(),
                first_admin.read_bytes(),
            )
            second_secret, second_admin = frontier_deploy.ensure_litellm_secret_files(
                self.config, private
            )
            self.assertEqual(secret_bytes, second_secret.read_bytes())
            self.assertEqual(admin_bytes, second_admin.read_bytes())
            self.assertEqual(first_secret.stat().st_mode & 0o777, 0o600)
            self.assertEqual(first_admin.stat().st_mode & 0o777, 0o600)
            self.assertEqual(private.stat().st_mode & 0o777, 0o700)
            document = json.loads(secret_bytes)
            self.assertEqual(document["metadata"]["namespace"], "frontier-litellm")
            self.assertNotIn("OPENROUTER_API_KEY", document["stringData"])
            self.assertIn(
                document["stringData"]["LITELLM_MASTER_KEY"].encode(), admin_bytes
            )

    def test_cross_account_outputs_are_rejected(self) -> None:
        outputs = dict(self.outputs)
        outputs["account_id"] = "999999999999"
        with self.assertRaisesRegex(frontier_deploy.DeployError, "account differs"):
            frontier_deploy.render_bootstrap_manifest(self.config, outputs)

    def test_smoke_catalog_binds_exact_runtime_digests_and_cpu_only_quota(self) -> None:
        candidate, task = frontier_deploy.smoke_documents(self.images)
        self.assertEqual(
            candidate["driver_config"]["overlay_image"]["digest"],
            self.images["candidate_overlay"]["digest"],
        )
        self.assertEqual(
            task["driver_config"]["images"]["verifier"]["digest"],
            self.images["verifier"]["digest"],
        )
        self.assertEqual(
            task["driver_config"]["quota"]["agent_gpu"]["max_instances"], 0
        )
        self.assertEqual(
            task["driver_config"]["quota"]["verifier"]["max_gpu_instances"], 0
        )
        self.assertRegex(task["definition_digest"], r"^sha256:[0-9a-f]{64}$")

    def test_smoke_version_ids_change_with_immutable_image_identity(self) -> None:
        first_candidate, first_task = frontier_deploy.smoke_documents(self.images)
        changed = copy.deepcopy(self.images)
        changed["candidate_overlay"]["digest"] = "sha256:" + "e" * 64
        changed["agent_base"]["digest"] = "sha256:" + "f" * 64
        second_candidate, second_task = frontier_deploy.smoke_documents(changed)
        self.assertNotEqual(
            first_candidate["driver_config"]["candidate_version_id"],
            second_candidate["driver_config"]["candidate_version_id"],
        )
        self.assertNotEqual(
            first_task["driver_config"]["task_version_id"],
            second_task["driver_config"]["task_version_id"],
        )
        self.assertRegex(
            second_task["driver_config"]["task_version_id"],
            r"^taskver_[0-7][0-9A-HJKMNP-TV-Z]{25}$",
        )


class ModalConfigTests(unittest.TestCase):
    def setUp(self):
        raw = json.loads((ROOT / "config/site.example.json").read_text())
        raw["features"]["modal_gpu"] = True
        self.config = frontier_deploy.validate_config(raw)

    def test_cpu_catalog_excludes_gpu_tasks(self):
        self.config["features"]["modal_gpu"] = False
        catalog, selection = frontier_deploy.load_task_selection(self.config)
        # Check native declarations, not the number of GPU tasks currently
        # published. Adding a second GPU task must not invalidate this gate.
        import render_task_definition as renderer

        roots = renderer.catalog_problem_directories(
            renderer.catalog_task_names(), renderer.TASK_ROOT / "problems"
        )
        gpu_names = set()
        for name, root in roots.items():
            gpu = json.loads((root / "policy.json").read_text()).get("gpu", {})
            if (
                gpu.get("agent", {}).get("max_instances", 0) > 0
                or gpu.get("verifier", {}).get("max_gpu_instances", 0) > 0
            ):
                gpu_names.add(name)
        self.assertTrue(gpu_names)
        self.assertEqual(
            selection.excluded,
            {name: "requires features.modal_gpu" for name in gpu_names},
        )
        self.assertEqual(
            [task["name"] for task in catalog["tasks"]], selection.selected
        )
        self.assertEqual(set(selection.selected), set(roots) - gpu_names)
        self.config["features"]["modal_gpu"] = True
        enabled_catalog, enabled = frontier_deploy.load_task_selection(self.config)
        self.assertEqual(enabled.excluded, {})
        self.assertEqual(set(enabled.selected), set(roots))
        self.assertEqual(
            [task["name"] for task in enabled_catalog["tasks"]], enabled.selected
        )

    def test_missing_credentials_fail_before_aws_apply(self):
        from test_task_seeding import resolved_catalog

        catalog = resolved_catalog()
        selection = frontier_deploy.select_tasks(
            self.config, catalog, frontier_deploy.TASK_DEFINITIONS
        )
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                cli, "load_task_selection", return_value=(catalog, selection)
            ),
            mock.patch.object(
                cli, "render_bundle", return_value=(self.config, Path("unused"))
            ),
            mock.patch.object(cli, "aws_identity") as aws,
        ):
            with self.assertRaisesRegex(frontier_deploy.DeployError, "MODAL_TOKEN_ID"):
                frontier_deploy.apply_site(
                    Path("unused"),
                    self.config["site"]["expected_account_id"],
                    self.config["site"]["name"],
                )
            aws.assert_not_called()

    def test_credentials_are_stdin_only_and_never_written_to_config(self):
        environment = {
            "MODAL_TOKEN_ID": "fixture-id",
            "MODAL_TOKEN_SECRET": "fixture-secret",
        }
        with (
            mock.patch.dict(os.environ, environment),
            mock.patch.object(cli, "executable", return_value="kubectl"),
        ):
            data = frontier_deploy.modal_credentials(self.config)
            replies = [
                mock.Mock(returncode=0, stdout=""),
                mock.Mock(returncode=0),
                mock.Mock(returncode=0, stdout=json.dumps({"data": data})),
            ]
            with mock.patch.object(cli.subprocess, "run", side_effect=replies) as run:
                frontier_deploy.deploy_modal_credentials(
                    self.config, Path("kubeconfig")
                )
                self.assertEqual(len(run.call_args_list), 3)
                for call in run.call_args_list:
                    self.assertNotIn("fixture-secret", str(call.args))
                submitted = json.loads(run.call_args_list[1].kwargs["input"])
                self.assertEqual(submitted["data"], data)
            self.assertNotIn("fixture-secret", json.dumps(self.config))

    def test_gpu_nat_routes_must_be_public(self):
        outputs = Fixture.outputs(self.config)
        for address in ("10.0.0.1", "2001:4860:4860::8888"):
            outputs["nat_gateway_public_ips"] = [address]
            with self.assertRaisesRegex(frontier_deploy.DeployError, "public NAT"):
                frontier_deploy.render_runtime_manifest(
                    self.config, outputs, Fixture.images(self.config), ["172.20.0.1/32"]
                )


class SourceTests(unittest.TestCase):
    def test_installer_retains_exact_application_source_and_task_handoff(self) -> None:
        source = (ROOT / "scripts/preview/images.py").read_text()
        self.assertIn('"management-plane/Dockerfile"', source)
        self.assertIn('"driver/Dockerfile"', source)
        self.assertIn("FRONTIERCS2_SOURCE_DIR", source)
        self.assertIn('"status", "--porcelain", "--untracked-files=all"', source)
        self.assertIn('"skills/add-frontier-task/SKILL.md"', source)
        self.assertIn('"skills/add-frontier-task/scripts/publish_task.py"', source)
        self.assertIn("Task authoring skill:", source)

    def test_required_application_source_is_rejected(self) -> None:
        raw = json.loads((ROOT / "config/site.example.json").read_text())
        raw["application"]["require_source"] = True
        with self.assertRaisesRegex(
            frontier_deploy.DeployError, "require_source=true is not supported"
        ):
            frontier_deploy.validate_config(raw)

    def test_optional_application_source_skips_git_and_reports_unavailable_handoff(
        self,
    ) -> None:
        raw = json.loads((ROOT / "config/site.example.json").read_text())
        raw["application"]["require_source"] = False
        config = frontier_deploy.validate_config(raw)
        with (
            mock.patch.object(image_ops, "executable") as executable,
            mock.patch.object(image_ops, "run_command") as run_command,
            mock.patch("builtins.print") as print_line,
        ):
            result = frontier_deploy.ensure_application_source(config)

        self.assertIsNone(result)
        executable.assert_not_called()
        run_command.assert_not_called()
        print_line.assert_called_once_with(
            "Task authoring handoff: UNAVAILABLE (application source is disabled for this site profile)"
        )

    def test_installer_mirrors_management_and_driver_release_digests(self) -> None:
        raw = json.loads((ROOT / "config/site.example.json").read_text())
        config = frontier_deploy.validate_config(raw)
        application_source = Path("/tmp/frontiercs2-application-source")
        upstream_digest = "sha256:" + "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            port_forward = mock.MagicMock()
            port_forward.return_value.__enter__.return_value = 5500
            with (
                mock.patch.object(
                    image_ops,
                    "ensure_application_source",
                    return_value=application_source,
                ),
                mock.patch.object(image_ops, "executable", return_value="docker"),
                mock.patch.object(image_ops, "port_forward", port_forward),
                mock.patch.object(image_ops, "run_command") as run_command,
                mock.patch.object(
                    image_ops, "registry_digest", return_value=upstream_digest
                ),
                mock.patch.object(
                    image_ops, "source_tree_sha256", return_value="sha256:" + "c" * 64
                ),
                mock.patch.object(
                    image_ops, "state_directory", return_value=Path(directory)
                ),
            ):
                records = frontier_deploy.build_runtime_images(
                    config, Path("/tmp/kubeconfig")
                )

        commands = [call.args[0] for call in run_command.call_args_list]
        for name in ("management", "driver"):
            source = config["images"][name]
            self.assertIn(
                [
                    "docker",
                    "pull",
                    "--platform",
                    "linux/amd64",
                    f"{source['repository']}@{source['digest']}",
                ],
                commands,
            )
            self.assertEqual(records[name]["source_sha256"], source["digest"])
            self.assertEqual(
                records[name]["tag"],
                f"{config['network']['registry_cluster_ip']}:5000/frontier/{name}:git-{config['application'].get('runtime_source_commits', {}).get(name, config['application']['commit'])[:20]}",
            )
        build_commands = [command for command in commands if command[1] == "build"]
        self.assertEqual(len(build_commands), 4)
        for command in build_commands:
            self.assertNotIn(str(application_source), command)
            self.assertFalse(
                any("/frontier/management:" in argument for argument in command)
            )
            self.assertFalse(
                any("/frontier/driver:" in argument for argument in command)
            )

    def test_all_non_scratch_runtime_base_images_are_digest_pinned(self) -> None:
        dockerfiles = list((ROOT / "runtime").rglob("Dockerfile")) + list(
            (ROOT / "smoke").rglob("Dockerfile")
        )
        self.assertGreaterEqual(len(dockerfiles), 4)
        for dockerfile in dockerfiles:
            for line in dockerfile.read_text().splitlines():
                if line.startswith("FROM ") and not line.startswith("FROM scratch"):
                    self.assertRegex(line, r"@sha256:[0-9a-f]{64}(?:\s+AS\s+\w+)?$")

    def test_terraform_has_efs_pod_identity_access_points_and_node_registry_config(
        self,
    ) -> None:
        versions = (ROOT / "terraform/platform/versions.tf").read_text()
        main = (ROOT / "terraform/platform/main.tf").read_text()
        self.assertIn('version = "21.24.0"', main)
        self.assertIn('version = "6.6.1"', main)
        self.assertIn('version = "~> 6.60.0"', versions)
        self.assertIn("aws-efs-csi-driver", main)
        self.assertIn("AmazonEFSCSIDriverPolicy", main)
        self.assertEqual(main.count('resource "aws_efs_access_point"'), 2)
        self.assertIn("cloudinit_pre_nodeadm", main)
        self.assertIn("/etc/containerd/certs.d/", main)
        self.assertGreaterEqual(main.count("prevent_destroy = true"), 3)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", main)
        self.assertIn('resource "aws_iam_role" "management_artifact_reader"', main)
        self.assertIn('resources = ["${aws_s3_bucket.artifacts.arn}/runs/*"]', main)
        self.assertIn('actions   = ["kms:Decrypt"]', main)

    def test_storage_controller_has_closed_identity_and_no_secret_api(self) -> None:
        source = (ROOT / "runtime/storage-controller/controller.py").read_text()
        self.assertIn("RUN_NAMESPACE", source)
        self.assertIn("frontier.dev/resource-kind", source)
        self.assertIn("ReadWriteMany", source)
        self.assertIn("cleanup_orphan_pvs", source)
        self.assertNotIn('"resources": ["secrets"]', source)

    def test_litellm_verification_forwards_the_litellm_namespace(self) -> None:
        source = ast.unparse(
            ast.parse((ROOT / "scripts/preview/acceptance.py").read_text())
        )
        self.assertIn(
            "port_forward(kubeconfig, LITELLM_SERVICE, 4000, namespace=LITELLM_NAMESPACE)",
            source,
        )


class StorageControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controller = load_storage_controller()

    def test_agent_external_egress_policy_document_is_exact(self) -> None:
        namespace = "frontier-run-01arz3ndektsv4rrffq69g5fav"
        self.assertEqual(
            self.controller.network_policy_document(namespace, "172.20.0.10/32"),
            {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": "frontier-agent-external-egress",
                    "namespace": namespace,
                    "labels": {
                        "app.kubernetes.io/part-of": "frontiercs2",
                        "frontier.dev/owner": "storage-controller",
                    },
                },
                "spec": {
                    "podSelector": {
                        "matchLabels": {"frontier.dev/network-role": "agent"},
                    },
                    "policyTypes": ["Egress"],
                    "egress": [
                        {
                            "to": [{"ipBlock": {"cidr": "172.20.0.10/32"}}],
                            "ports": [
                                {"protocol": "UDP", "port": 53},
                                {"protocol": "TCP", "port": 53},
                            ],
                        },
                        {
                            "to": [
                                {
                                    "ipBlock": {
                                        "cidr": "0.0.0.0/0",
                                        "except": [
                                            "10.0.0.0/8",
                                            "100.64.0.0/10",
                                            "127.0.0.0/8",
                                            "169.254.0.0/16",
                                            "172.16.0.0/12",
                                            "192.168.0.0/16",
                                        ],
                                    },
                                }
                            ],
                            "ports": [{"protocol": "TCP", "port": 443}],
                        },
                    ],
                },
            },
        )

    def test_agent_external_egress_policy_is_idempotent_and_replaces_drift(
        self,
    ) -> None:
        namespace = "frontier-run-01arz3ndektsv4rrffq69g5fav"
        desired = self.controller.network_policy_document(namespace, "172.20.0.10/32")
        current = {
            **desired,
            "metadata": {**desired["metadata"], "resourceVersion": "7"},
            "spec": {**desired["spec"], "policyTypes": ["Ingress", "Egress"]},
        }
        reconciled = {
            **desired,
            "metadata": {**desired["metadata"], "resourceVersion": "8"},
        }
        with mock.patch.object(
            self.controller,
            "api",
            side_effect=[current, {}, reconciled],
        ) as api:
            self.controller.create_or_replace_network_policy(namespace, desired)

        policy_path = (
            "/apis/networking.k8s.io/v1/namespaces/"
            f"{namespace}/networkpolicies/frontier-agent-external-egress"
        )
        self.assertEqual(
            api.call_args_list,
            [
                mock.call("GET", policy_path, absent_ok=True),
                mock.call(
                    "PUT",
                    policy_path,
                    {
                        **desired,
                        "metadata": {**desired["metadata"], "resourceVersion": "7"},
                    },
                ),
                mock.call("GET", policy_path),
            ],
        )

        with mock.patch.object(self.controller, "api", return_value=reconciled) as api:
            self.controller.create_or_replace_network_policy(namespace, desired)
        api.assert_called_once_with("GET", policy_path, absent_ok=True)

    def test_disabled_controller_does_not_require_dns_configuration(self) -> None:
        controller = load_storage_controller(enabled=False)
        self.assertFalse(controller.AGENT_EXTERNAL_EGRESS)
        self.assertIsNone(controller.DNS_CIDR)

    def test_disabled_controller_deletes_an_existing_agent_external_egress_policy(
        self,
    ) -> None:
        controller = load_storage_controller(enabled=False)
        namespace = "frontier-run-01arz3ndektsv4rrffq69g5fav"
        existing = self.controller.network_policy_document(namespace, "172.20.0.10/32")
        policy_path = (
            "/apis/networking.k8s.io/v1/namespaces/"
            f"{namespace}/networkpolicies/frontier-agent-external-egress"
        )
        with (
            mock.patch.object(controller, "api", side_effect=[existing, {}]) as api,
            mock.patch("builtins.print"),
        ):
            self.assertTrue(controller.remove_agent_external_egress_policy(namespace))
        self.assertEqual(
            api.call_args_list,
            [
                mock.call("GET", policy_path, absent_ok=True),
                mock.call("DELETE", policy_path, absent_ok=True),
            ],
        )

        with mock.patch.object(controller, "api", return_value=None) as api:
            self.assertFalse(controller.remove_agent_external_egress_policy(namespace))
        api.assert_called_once_with("GET", policy_path, absent_ok=True)

    def test_disabled_controller_withdraws_egress_before_storage_reconcile(
        self,
    ) -> None:
        controller = load_storage_controller(enabled=False)
        namespace = {
            "metadata": {
                "name": "frontier-run-01arz3ndektsv4rrffq69g5fav",
                "labels": {
                    "frontier.dev/owner": "management",
                    "frontier.dev/resource-kind": "run-namespace",
                    "frontier.dev/run-id": "run_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                    "frontier.dev/attempt-id": "attempt_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                },
            }
        }
        with (
            mock.patch.object(
                controller, "remove_agent_external_egress_policy"
            ) as remove,
            mock.patch.object(controller, "api", side_effect=RuntimeError("PV 403")),
        ):
            with self.assertRaisesRegex(RuntimeError, "PV 403"):
                controller.reconcile_namespace(namespace)
        remove.assert_called_once_with("frontier-run-01arz3ndektsv4rrffq69g5fav")

    def test_enabled_controller_never_deletes_agent_external_egress(self) -> None:
        with mock.patch.object(
            self.controller, "remove_agent_external_egress_policy"
        ) as remove, mock.patch.object(
            self.controller, "api", side_effect=RuntimeError("PV 403")
        ):
            with self.assertRaises(RuntimeError):
                self.controller.reconcile_namespace(
                    {
                        "metadata": {
                            "name": "frontier-run-01arz3ndektsv4rrffq69g5fav",
                            "labels": {
                                "frontier.dev/owner": "management",
                                "frontier.dev/resource-kind": "run-namespace",
                                "frontier.dev/run-id": "run_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                                "frontier.dev/attempt-id": "attempt_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                            },
                        }
                    }
                )
        remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
