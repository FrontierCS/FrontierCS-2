#
import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from preview import cli, infra
from test_frontier_deploy import ROOT, Fixture, frontier_deploy


def apply_patch(document, patches):
    result = copy.deepcopy(document)
    for operation in patches:
        parts = operation["path"].strip("/").split("/")
        parent = result
        for part in parts[:-1]:
            parent = parent[int(part)] if isinstance(parent, list) else parent[part]
        key = int(parts[-1]) if isinstance(parent, list) else parts[-1]
        if operation["op"] == "test":
            if parent[key] != operation["value"]:
                raise AssertionError("resourceVersion conflict")
        elif operation['op'] == 'remove':
            del parent[key]
        else:
            parent[key] = copy.deepcopy(operation["value"])
    return result


class RetentionUpgradeTests(unittest.TestCase):
    def fixture(self, *, modal_gpu=False, raw_config=None):
        config = frontier_deploy.validate_config(
            raw_config if raw_config is not None else
            json.loads((ROOT / "config/site.example.json").read_text())
        )
        config["features"]["modal_gpu"] = modal_gpu
        images = Fixture.images(config)
        items = json.loads(
            infra.render_runtime_manifest(
                config, Fixture.outputs(config), images, ["172.20.0.1/32"]
            )
        )["items"]
        cm = next(
            x
            for x in items
            if x["kind"] == "ConfigMap"
            and x["metadata"]["name"] == "frontier-management-config"
        )
        dep = next(
            x
            for x in items
            if x["kind"] == "Deployment"
            and x["metadata"]["name"] == "frontier-management"
        )
        cm["metadata"]["resourceVersion"] = "42"
        dep["metadata"]["resourceVersion"] = "73"
        dep["metadata"]["generation"] = 74
        return config, images, cm, dep

    def patch(self, cm, dep, images):
        return cli.runtime_upgrade_patches(
            cm,
            dep,
            images,
            litellm_cidr="172.20.0.60/32",
            gateway_artifact_retention=True,
        )

    def test_existing_small_tmp_and_memory_upgrade_to_current_result_resources(self):
        _, images, cm, dep = self.fixture()
        pod = dep["spec"]["template"]["spec"]
        management = next(c for c in pod["containers"] if c["name"] == "management")
        management["resources"] = {"requests": {"cpu": "100m", "memory": "1Gi"},
                                   "limits": {"cpu": "1", "memory": "1Gi"}}
        next(v for v in pod["volumes"] if v["name"] == "tmp")["emptyDir"]["sizeLimit"] = "64Mi"
        other = {"name": "other", "image": "unchanged", "resources": {"limits": {"memory": "32Mi"}}}
        pod["containers"].append(other)
        original = copy.deepcopy(dep)
        _, patches, _ = cli.runtime_upgrade_patches(cm, dep, images, large_verifier_results=True)
        after = apply_patch(dep, patches)["spec"]["template"]["spec"]
        self.assertEqual(dep, original)
        self.assertEqual(after["containers"][-1], other)
        self.assertEqual(after["containers"][0]["resources"], infra.management_result_resources())
        self.assertEqual(next(v for v in after["volumes"] if v["name"] == "tmp")["emptyDir"]["sizeLimit"], "34Gi")
        self.assertEqual([v for v in after["volumes"] if v["name"] != "tmp"],
                         [v for v in pod["volumes"] if v["name"] != "tmp"])

    def test_large_result_upgrade_rejects_non_disk_or_mismatched_temporary_storage(self):
        _, images, cm, dep = self.fixture()
        for mutation in ("memory", "persistent", "missing", "wrong-mount"):
            with self.subTest(mutation=mutation):
                altered = copy.deepcopy(dep)
                pod = altered["spec"]["template"]["spec"]
                tmp = next(v for v in pod["volumes"] if v["name"] == "tmp")
                if mutation == "memory":
                    tmp["emptyDir"]["medium"] = "Memory"
                elif mutation == "persistent":
                    tmp.pop("emptyDir")
                    tmp["persistentVolumeClaim"] = {"claimName": "custom"}
                elif mutation == "missing":
                    pod["volumes"].remove(tmp)
                else:
                    next(m for m in pod["containers"][0]["volumeMounts"] if m["name"] == "tmp")["mountPath"] = "/custom"
                with self.assertRaisesRegex(infra.DeployError, "disk-backed Management /tmp"):
                    cli.runtime_upgrade_patches(cm, altered, images, large_verifier_results=True)

    def test_existing_site_acquires_paired_roots_and_private_init_without_other_changes(
        self,
    ):
        _, images, cm, dep = self.fixture()
        original = copy.deepcopy((cm, dep))
        for key in (
            "MANAGEMENT_GATEWAY_ARTIFACT_SOURCE_ROOT",
            "MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT",
        ):
            cm["data"].pop(key)
        dep["spec"]["template"]["spec"].pop("initContainers")
        before = copy.deepcopy((cm, dep))
        cp, dp, data = self.patch(cm, dep, images)
        after = apply_patch(dep, dp)
        self.assertEqual((cm, dep), before)
        self.assertEqual(apply_patch(cm, cp)["data"], data)
        self.assertEqual(
            data["MANAGEMENT_GATEWAY_ARTIFACT_SOURCE_ROOT"],
            "/var/lib/frontier-local/artifacts",
        )
        self.assertEqual(
            data["MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT"],
            "/var/lib/frontier/gateway-terminal-retained",
        )
        self.assertEqual(
            after["spec"]["template"]["spec"], original[1]["spec"]["template"]["spec"]
        )
        drift = copy.deepcopy(dep)
        drift["metadata"]["resourceVersion"] = "74"
        with self.assertRaisesRegex(AssertionError, "resourceVersion"):
            apply_patch(drift, dp)

    def test_existing_init_is_repinned_and_other_init_preserved_without_input_mutation(
        self,
    ):
        _, images, cm, dep = self.fixture()
        pod = dep["spec"]["template"]["spec"]
        other = {"name": "existing-other", "image": "unchanged", "command": ["true"]}
        pod["initContainers"].insert(0, other)
        before = copy.deepcopy((cm, dep))
        new_images = copy.deepcopy(images)
        new_images["management"]["reference"] = "registry/management@sha256:" + "a" * 64
        _, dp, _ = self.patch(cm, dep, new_images)
        inits = apply_patch(dep, dp)["spec"]["template"]["spec"]["initContainers"]
        self.assertEqual((cm, dep), before)
        self.assertEqual(inits[0], other)
        self.assertEqual(
            inits[1],
            infra.gateway_artifact_retention_init(
                new_images["management"]["reference"]
            ),
        )
        self.assertEqual(len(inits), 2)

    def test_kubernetes_defaulted_init_can_be_upgraded_but_other_message_contract_rejected(
        self,
    ):
        _, images, cm, dep = self.fixture()
        init = dep["spec"]["template"]["spec"]["initContainers"][0]
        init["terminationMessagePath"] = "/dev/termination-log"
        init["terminationMessagePolicy"] = "File"
        _, dp, _ = self.patch(cm, dep, images)
        self.assertEqual(
            apply_patch(dep, dp)["spec"]["template"]["spec"]["initContainers"],
            [infra.gateway_artifact_retention_init(images["management"]["reference"])],
        )
        init["terminationMessagePolicy"] = "FallbackToLogsOnError"
        with self.assertRaisesRegex(cli.DeployError, "existing init differs"):
            self.patch(cm, dep, images)

    def test_changed_same_name_init_rejected(self):
        _, images, cm, dep = self.fixture()
        for field, value in (
            ("command", ["sh", "-ec", "chmod 777 /var/lib/frontier"]),
            ("volumeMounts", [{"name": "objects", "mountPath": "/var/lib/frontier"}]),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(dep)
                changed["spec"]["template"]["spec"]["initContainers"][0][field] = value
                with self.assertRaisesRegex(cli.DeployError, "existing init differs"):
                    self.patch(cm, changed, images)

    def test_nonisolated_or_writable_mounts_rejected(self):
        _, images, cm, dep = self.fixture()
        for mutation in ("shared", "source-write", "data-readonly", "wrong-uid"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(dep)
                pod = changed["spec"]["template"]["spec"]
                mounts = {
                    x["name"]: x
                    for x in pod["containers"][0]["volumeMounts"]
                    if "subPath" not in x
                }
                if mutation == "shared":
                    next(x for x in pod["volumes"] if x["name"] == "objects")[
                        "persistentVolumeClaim"
                    ]["claimName"] = "frontier-management-data"
                elif mutation == "source-write":
                    mounts["objects"]["readOnly"] = False
                elif mutation == "data-readonly":
                    mounts["data"]["readOnly"] = True
                else:
                    pod["securityContext"]["runAsUser"] = 0
                with self.assertRaises(cli.DeployError):
                    self.patch(cm, changed, images)

    def test_conflicting_root_or_direct_environment_override_rejected(self):
        _, images, cm, dep = self.fixture()
        changed = copy.deepcopy(cm)
        changed["data"]["MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT"] = (
            "/var/lib/frontier-local/unsafe"
        )
        with self.assertRaisesRegex(cli.DeployError, "root differs"):
            self.patch(changed, dep, images)
        changed = copy.deepcopy(dep)
        changed["spec"]["template"]["spec"]["containers"][0]["env"] = [
            {"name": "MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT", "value": "/unsafe"}
        ]
        with self.assertRaisesRegex(cli.DeployError, "environment overrides"):
            self.patch(cm, changed, images)

    def test_supported_upgrade_enables_retention_and_checks_complete_spec(self):
        for modal_gpu in (False, True):
            with self.subTest(modal_gpu=modal_gpu):
                self.check_supported_upgrade(modal_gpu)

    def test_governance_upgrade_adds_only_artifact_write_mount_and_preserves_poll(self):
        config, images, cm, dep = self.fixture()
        cm["data"]["MANAGEMENT_DRIVER_CONTROLLER_POLL_MILLIS"] = "23000"
        pod = dep["spec"]["template"]["spec"]
        pod["volumes"] = [v for v in pod["volumes"] if v["name"] != "diagnostic-source"]
        next(v for v in pod["volumes"] if v["name"] == "objects")[
            "persistentVolumeClaim"
        ]["claimName"] = "site-object-store"
        next(v for v in pod["volumes"] if v["name"] == "objects")[
            "persistentVolumeClaim"
        ]["readOnly"] = True
        container = pod["containers"][0]
        container["volumeMounts"] = [
            v for v in container["volumeMounts"] if "subPath" not in v
        ]
        _, patches, data = self.patch(cm, dep, images)
        after = apply_patch(dep, patches)["spec"]["template"]["spec"]
        volume, mount = infra.management_diagnostic_source_mount("site-object-store")
        self.assertEqual(
            after["volumes"],
            [volume if v["name"] == "objects" else v for v in pod["volumes"]],
        )
        self.assertEqual(
            after["containers"][0]["volumeMounts"], [*container["volumeMounts"], mount]
        )
        self.assertEqual(data["MANAGEMENT_DRIVER_CONTROLLER_POLL_MILLIS"], "23000")
        objects = next(
            m for m in after["containers"][0]["volumeMounts"] if m["name"] == "objects"
        )
        self.assertTrue(objects["readOnly"])
        grants = infra.management_diagnostics_rbac(config)
        self.assertEqual(
            grants[0]["rules"],
            [
                {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]},
                {"apiGroups": [""], "resources": ["events"], "verbs": ["list"]},
            ],
        )
        self.assertEqual(
            grants[1]["subjects"],
            [
                {
                    "kind": "ServiceAccount",
                    "name": "frontier-management",
                    "namespace": "frontier-system",
                }
            ],
        )

    def test_governance_mount_conflicts_are_rejected(self):
        _, images, cm, dep = self.fixture()
        for mutation in ("subpath", "claim", "collision"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(dep)
                pod = changed["spec"]["template"]["spec"]
                mounts = pod["containers"][0]["volumeMounts"]
                mount = next(m for m in mounts if "subPath" in m)
                if mutation == "subpath":
                    mount["subPath"] = "datasets"
                elif mutation == "claim":
                    pod["volumes"].append(
                        {
                            "name": "diagnostic-source",
                            "persistentVolumeClaim": {"claimName": "foreign-data"},
                        }
                    )
                else:
                    mounts.append({"name": "foreign", "mountPath": mount["mountPath"]})
                with self.assertRaisesRegex(cli.DeployError, "diagnostic source"):
                    self.patch(cm, changed, images)

    def test_duplicate_pvc_alias_migrates_and_single_volume_rerun_is_idempotent(self):
        _, images, cm, dep = self.fixture()
        pod = dep["spec"]["template"]["spec"]
        objects = next(v for v in pod["volumes"] if v["name"] == "objects")
        objects["persistentVolumeClaim"] = {
            "claimName": "custom-object-store",
            "readOnly": True,
        }
        pod["volumes"].append(
            {
                "name": "diagnostic-source",
                "persistentVolumeClaim": {"claimName": "custom-object-store"},
            }
        )
        next(m for m in pod["containers"][0]["volumeMounts"] if "subPath" in m)[
            "name"
        ] = "diagnostic-source"
        before = copy.deepcopy(dep)
        _, patches, _ = self.patch(cm, dep, images)
        after = apply_patch(dep, patches)
        self.assertEqual(dep, before)
        result = after["spec"]["template"]["spec"]
        object_volumes = [
            v
            for v in result["volumes"]
            if v.get("persistentVolumeClaim", {}).get("claimName")
            == "custom-object-store"
        ]
        self.assertEqual(
            object_volumes,
            [
                {
                    "name": "objects",
                    "persistentVolumeClaim": {"claimName": "custom-object-store"},
                }
            ],
        )
        mounts = result["containers"][0]["volumeMounts"]
        self.assertEqual(
            [m for m in mounts if m["name"] == "objects"],
            [
                {
                    "name": "objects",
                    "mountPath": "/var/lib/frontier-local",
                    "readOnly": True,
                },
                {
                    "name": "objects",
                    "mountPath": "/var/lib/frontier-local/artifacts",
                    "subPath": "artifacts",
                },
            ],
        )
        _, repeated, _ = self.patch(cm, after, images)
        self.assertEqual(apply_patch(after, repeated), after)
        stale = copy.deepcopy(dep)
        stale["metadata"]["resourceVersion"] = "stale"
        with self.assertRaisesRegex(AssertionError, "resourceVersion"):
            apply_patch(stale, patches)

    def test_alias_migration_rejects_other_consumers_and_pvc_aliases(self):
        _, images, cm, dep = self.fixture()
        for mutation in (
            "legacy-consumer",
            "writable-parent-consumer",
            "another-alias",
        ):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(dep)
                pod = changed["spec"]["template"]["spec"]
                if mutation == "another-alias":
                    pod["volumes"].append(
                        {
                            "name": "other-alias",
                            "persistentVolumeClaim": {
                                "claimName": "frontier-object-seed"
                            },
                        }
                    )
                else:
                    name = (
                        "diagnostic-source"
                        if mutation == "legacy-consumer"
                        else "objects"
                    )
                    pod["initContainers"].append(
                        {
                            "name": "other",
                            "volumeMounts": [{"name": name, "mountPath": "/other"}],
                        }
                    )
                with self.assertRaisesRegex(cli.DeployError, "diagnostic source"):
                    self.patch(cm, changed, images)

    def test_single_volume_preserves_additional_readonly_mount(self):
        _, images, cm, dep = self.fixture()
        extra = {"name": "objects", "mountPath": "/readonly-datasets", "readOnly": True}
        dep["spec"]["template"]["spec"]["containers"][0]["volumeMounts"].append(extra)
        _, patches, _ = self.patch(cm, dep, images)
        self.assertIn(
            extra,
            apply_patch(dep, patches)["spec"]["template"]["spec"]["containers"][0][
                "volumeMounts"
            ],
        )

    def check_supported_upgrade(self, modal_gpu, *, retain_existing_init=False, expected_failure=False):
        config, images, cm, dep = self.fixture(modal_gpu=modal_gpu)
        if not retain_existing_init:
            for key in (
                "MANAGEMENT_GATEWAY_ARTIFACT_SOURCE_ROOT",
                "MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT",
            ):
                cm["data"].pop(key)
            dep["spec"]["template"]["spec"].pop("initContainers")
        observed = {"configmap": cm, "deployment": dep, "networkpolicy": {"spec": {}},
                    "clusterrole": {"metadata": {"uid": "role-uid", "resourceVersion": "9"}, "rules": []}}

        def discover(*args, receipt_path=None):
            if receipt_path:
                receipt_path.write_text('{"cidrs":["172.20.0.1/32"]}')
            return ["172.20.0.1/32"]

        calls = []

        def command(argv, **kwargs):
            calls.append(argv)
            if "apply" in argv:
                manifest = json.loads(Path(argv[argv.index("-f") + 1]).read_text())
                expected = (
                    infra.management_diagnostics_rbac(config)
                    if "management-diagnostics-rbac.json" in argv[argv.index("-f") + 1]
                    else infra.management_api_routes_rbac()
                    if "management-api-routes-rbac.json" in argv[argv.index("-f") + 1]
                    else infra.modal_observer_rbac(config)
                )
                self.assertEqual(manifest["items"], expected)
                if "--dry-run=server" not in argv:
                    self.assertEqual(idle.call_count, 2)
                    self.assertFalse(
                        any(
                            "patch" in call and "--dry-run=server" not in call
                            for call in calls
                        )
                    )
            if "get" in argv:
                kind = argv[argv.index("get") + 1]
                return json.dumps(observed[kind])
            if "patch" in argv and "--dry-run=server" not in argv:
                kind = argv[argv.index("patch") + 1]
                patches = json.loads(
                    Path(argv[argv.index("--patch-file") + 1]).read_text()
                )
                observed[kind] = apply_patch(observed[kind], patches)
                if kind == "deployment":
                    for init in observed[kind]["spec"]["template"]["spec"][
                        "initContainers"
                    ]:
                        init.setdefault(
                            "terminationMessagePath", "/dev/termination-log"
                        )
                        init.setdefault("terminationMessagePolicy", "File")
            return ""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(cli, "load_task_selection"),
                mock.patch.object(cli, "api_cidrs", side_effect=discover),
                mock.patch.multiple(
                    cli,
                    verify_api_routes=mock.DEFAULT,
                    verify_api_route_reader_rbac=mock.DEFAULT,
                ),
                mock.patch.object(cli, "api_policy_patch", return_value=([], {})),
                mock.patch.object(
                    cli, "read_gateway_snapshot", return_value={}
                ) as gateway,
                mock.patch.object(
                    cli, "load_runtime_images", return_value=copy.deepcopy(images)
                ),
                mock.patch.object(cli, "executable", side_effect=lambda value: value),
                mock.patch.object(cli, "kubernetes_environment", return_value={}),
                mock.patch.object(
                    cli,
                    "port_forward",
                    return_value=mock.MagicMock(
                        __enter__=lambda s: 1234, __exit__=lambda *a: None
                    ),
                ) as forwards,
                mock.patch.object(cli, "require_idle_runtime") as idle,
                mock.patch.object(
                    cli,
                    "cached_registry_image",
                    side_effect=lambda port, name, digest: digest,
                ) as cached,
                mock.patch.object(cli, "run_command", side_effect=command),
                mock.patch.object(cli, "deploy_litellm") as litellm,
                mock.patch.object(cli, "platform_outputs", return_value=Fixture.outputs(config)),
                mock.patch.object(cli, "api_json", return_value={}),
                mock.patch.object(cli, "state_directory", return_value=root),
            ):
                if expected_failure:
                    with self.assertRaises(infra.DeployError):
                        cli.upgrade_runtime(config, root, root / "kubeconfig")
                    cached.assert_not_called()
                    self.assertFalse(any(call.args[1] == cli.REGISTRY_SERVICE for call in forwards.call_args_list))
                    self.assertFalse(any(any(verb in argv for verb in ("copy", "apply", "patch")) for argv in calls))
                    self.assertFalse((root / "runtime-images.json").exists())
                    return
                cli.upgrade_runtime(config, root, root / "kubeconfig")
            if retain_existing_init:
                cached.assert_not_called()
                self.assertFalse(any(call.args[1] == cli.REGISTRY_SERVICE for call in forwards.call_args_list))
                self.assertFalse(any("copy" in argv for argv in calls))
                persisted = json.loads((root / "runtime-images.json").read_text())
                self.assertEqual(infra.validate_runtime_images(persisted, config), persisted)
                receipt = json.loads(next(root.glob("runtime-upgrade-*/accepted.json")).read_text())
                for name in ("management", "driver"):
                    self.assertEqual(receipt["images"][name]["reference"], config["images"][name]["reference"])
                self.assertEqual(observed["configmap"]["data"]["MANAGEMENT_DRIVER_IMAGE"], config["images"]["driver"]["reference"])
                self.assertEqual(observed["configmap"]["data"]["MANAGEMENT_LITELLM_KEY_WORKER_IMAGE"], config["images"]["management"]["reference"])
            self.assertEqual(idle.call_count, 3)
            litellm.assert_not_called()
            self.assertEqual(gateway.call_count, 3)
            self.assertIn(
                "MANAGEMENT_GATEWAY_ARTIFACT_RETAINED_ROOT",
                observed["configmap"]["data"],
            )
            init = observed["deployment"]["spec"]["template"]["spec"]["initContainers"]
            management_image = observed["deployment"]["spec"]["template"]["spec"][
                "containers"
            ][0]["image"]
            self.assertEqual(
                init,
                [
                    {
                        **infra.gateway_artifact_retention_init(management_image),
                        "terminationMessagePath": "/dev/termination-log",
                        "terminationMessagePolicy": "File",
                    }
                ],
            )
            self.assertEqual(
                sum("--dry-run=server" in argv for argv in calls), 8 if modal_gpu else 7
            )
            self.assertEqual(
                sum("apply" in argv for argv in calls), 6 if modal_gpu else 4
            )
            self.assertTrue(
                (
                    root
                    / ("runtime-upgrade-" + config["application"]["commit"][:12])
                    / "accepted.json"
                ).exists()
            )

    def test_modal_observer_upgrade_rejects_existing_credential_scope_drift(self):
        config, _, cm, dep = self.fixture(modal_gpu=True)
        for key in (
            "MANAGEMENT_NAMESPACE",
            "MANAGEMENT_MODAL_CREDENTIALS_SECRET",
            "serviceAccountName",
        ):
            with self.subTest(key=key):
                changed_cm, changed_dep = copy.deepcopy((cm, dep))
                if key == "serviceAccountName":
                    changed_dep["spec"]["template"]["spec"][key] = "other"
                else:
                    changed_cm["data"][key] = "other"
                with self.assertRaisesRegex(cli.DeployError, "differs"):
                    cli.modal_observer_upgrade_manifest(config, changed_cm, changed_dep)


if __name__ == "__main__":
    unittest.main()
