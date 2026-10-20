"""Runtime upgrades must preserve the actual serving blue-green gateway."""

import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import unittest
from pathlib import Path
from unittest import mock

from preview import cli
from preview.gateway_preservation import validate_snapshot
from preview.infra import DeployError, litellm_serving_receipt


def fixture():
    def obj(kind, name, **extra):
        return dict(
            kind=kind,
            metadata=dict(
                name=name, namespace="litellm", uid=name + "-uid", generation=1
            ),
            **extra,
        )

    image = "gateway@sha256:" + "a" * 64
    labels = {"app": "audit-gateway"}
    service = obj(
        "Service",
        "frontier-litellm",
        spec=dict(selector=labels, clusterIP="172.20.0.51"),
    )
    deploy = obj(
        "Deployment",
        "audit-gateway",
        spec=dict(
            template=dict(
                metadata=dict(labels=labels),
                spec=dict(
                    containers=[
                        dict(
                            name="gateway",
                            image=image,
                            env=[
                                dict(
                                    name="MOONSHOT_API_KEY",
                                    valueFrom={
                                        "secretKeyRef": {
                                            "name": "provider",
                                            "key": "MOONSHOT_API_KEY",
                                        }
                                    },
                                )
                            ],
                        )
                    ]
                ),
            )
        ),
        status=dict(
            observedGeneration=1,
            availableReplicas=1,
            updatedReplicas=1,
            readyReplicas=1,
        ),
    )
    pg = obj("StatefulSet", "frontier-litellm-postgres", status=dict(readyReplicas=1))
    pvc = obj(
        "PersistentVolumeClaim",
        "data-frontier-litellm-postgres-0",
        status=dict(phase="Bound"),
    )
    secret = obj(
        "Secret",
        "litellm-master-key",
        data=dict(master="encoded", MOONSHOT_API_KEY="extra"),
    )
    cm = obj(
        "ConfigMap",
        "audit-config",
        data={"preview_audit.py": "exact audit", "config.yaml": "provider config"},
    )
    receipt = litellm_serving_receipt(service, pg, pvc, secret, deploy)
    receipt["metadata"]["uid"] = "receipt-uid"
    pod = obj(
        "Pod",
        "gateway-pod",
        spec=copy.deepcopy(deploy["spec"]["template"]["spec"]),
        status=dict(
            phase="Running",
            podIP="10.0.0.1",
            conditions=[dict(type="Ready", status="True")],
            containerStatuses=[
                dict(name="gateway", ready=True, imageID="docker-pullable://" + image)
            ],
        ),
    )
    pod["metadata"].update(
        labels=labels, ownerReferences=[dict(controller=True, uid="rs-uid")]
    )
    rs = obj("ReplicaSet", "rs")
    rs["metadata"]["ownerReferences"] = [dict(controller=True, uid="audit-gateway-uid")]
    endpoint = obj(
        "Endpoints",
        "frontier-litellm",
        subsets=[
            dict(
                addresses=[
                    dict(
                        ip="10.0.0.1", targetRef=dict(kind="Pod", uid="gateway-pod-uid")
                    )
                ]
            )
        ],
    )
    base = copy.deepcopy(deploy)
    base["metadata"].update(name="frontier-litellm", uid="base-uid")
    base["spec"]["template"]["metadata"]["labels"] = {"app": "base-gateway"}
    return dict(
        services=[service],
        deployments=[base, deploy],
        statefulsets=[pg],
        persistentvolumeclaims=[pvc],
        secrets=[secret],
        configmaps=[cm, receipt],
        pods=[pod],
        replicasets=[rs],
        endpoints=[endpoint],
    )


class GatewayPreservationTests(unittest.TestCase):
    def test_bluegreen_and_extra_provider_keys_preserved(self):
        data = fixture()
        snapshot = validate_snapshot(data)
        self.assertEqual(
            snapshot["services"][0]["spec"]["selector"], {"app": "audit-gateway"}
        )
        self.assertEqual(snapshot["secrets"][0]["data"]["MOONSHOT_API_KEY"], "extra")
        self.assertEqual(
            snapshot["configmaps"][0]["data"]["preview_audit.py"], "exact audit"
        )

    def test_controller_observation_churn_is_ignored(self):
        data = fixture()
        before = validate_snapshot(data)
        for values in data.values():
            for obj in values:
                obj["metadata"].update(resourceVersion="new", managedFields=[{}])
                obj.setdefault("status", {})["timestamp"] = "later"
        self.assertEqual(before, validate_snapshot(data))

    def test_configuration_and_identity_changes_detected(self):
        for kind, field, value in (
            ("secrets", "data", {"changed": "key"}),
            ("configmaps", "data", {"preview_audit.py": "different"}),
            ("services", "metadata", {"name": "frontier-litellm", "uid": "changed"}),
        ):
            with self.subTest(kind=kind):
                data = fixture()
                before = validate_snapshot(data)
                data[kind][0][field] = value
                try:
                    after = validate_snapshot(data)
                except DeployError:
                    continue
                self.assertNotEqual(before, after)

    def test_terminating_serving_resources_rejected(self):
        for kind, index in (
            ("services", 0),
            ("statefulsets", 0),
            ("persistentvolumeclaims", 0),
            ("secrets", 0),
            ("configmaps", 1),
        ):
            with self.subTest(kind=kind):
                data = fixture()
                data[kind][index]["metadata"]["deletionTimestamp"] = "now"
                with self.assertRaisesRegex(DeployError, "terminating"):
                    validate_snapshot(data)

    def test_unready_or_wrong_serving_identity_rejected(self):
        mutations = [
            lambda r: r["pods"][0]["metadata"].update(deletionTimestamp="now"),
            lambda r: r["deployments"][1]["metadata"].update(deletionTimestamp="now"),
            lambda r: r["pods"][0]["status"].update(phase="Succeeded"),
            lambda r: r["endpoints"][0].update(subsets=[]),
            lambda r: r["services"][0]["spec"].update(selector={"app": "absent"}),
            lambda r: r["deployments"][1]["status"].update(observedGeneration=0),
            lambda r: r["pods"][0]["status"]["containerStatuses"][0].update(
                imageID="wrong"
            ),
            lambda r: r["replicasets"][0]["metadata"].update(ownerReferences=[]),
            lambda r: r["configmaps"][1]["data"].update(lease_nonce="wrong"),
            lambda r: r["pods"][0]["status"].update(podIP="wrong"),
            lambda r: r.update(services=[]),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                data = fixture()
                mutate(data)
                with self.assertRaises(DeployError):
                    validate_snapshot(data)

    def test_bad_gateway_stops_before_runtime_reads_or_writes(self):
        with (
            mock.patch.object(cli, "load_task_selection"),
            mock.patch.object(
                cli, "read_gateway_snapshot", side_effect=DeployError("gateway unready")
            ),
            mock.patch.object(cli, "load_runtime_images") as images,
            mock.patch.object(cli, "run_command") as command,
            mock.patch.object(cli, "deploy_litellm") as deploy,
        ):
            with self.assertRaisesRegex(DeployError, "unready"):
                cli.upgrade_runtime({}, Path("output"), Path("kubeconfig"))
        images.assert_not_called()
        command.assert_not_called()
        deploy.assert_not_called()

    def test_changed_gateway_rejects_before_any_management_mutation(self):
        self.test_success_upgrades_only_runtime_and_preserves_gateway(
            gateway_changed=True
        )

    def test_success_upgrades_only_runtime_and_preserves_gateway(
        self, gateway_changed=False
    ):
        import contextlib
        import json
        import tempfile

        digest = "sha256:" + "b" * 64
        config = dict(
            execution=dict(verifier_node_allowlists=[]),
            application=dict(commit="c" * 40),
            site=dict(aws_region='us-west-1'),
            network=dict(
                registry_cluster_ip="172.20.0.50", litellm_cluster_ip="172.20.0.51"
            ),
            images={
                name: dict(reference="public/" + name + "@" + digest, digest=digest)
                for name in ("management", "driver")
            },
        )
        cm = dict(metadata=dict(resourceVersion="1"), data=dict(keep="old"))
        dep = dict(
            metadata=dict(resourceVersion="2", generation=1),
            spec=dict(
                template=dict(
                    metadata=dict(annotations={}),
                    spec=dict(
                        serviceAccountName="frontier-management",
                        containers=[dict(name="management", image="old")],
                    ),
                )
            ),
        )
        after = copy.deepcopy(dep)
        after["spec"]["template"]["spec"]["containers"][0]["image"] = (
            "172.20.0.50:5000/frontier/management@" + digest
        )
        after["spec"]["template"]["metadata"]["annotations"] = {"checksum": "new"}
        reads = iter([cm, dep, cm, after])

        def discover(*args, receipt_path=None):
            if receipt_path:
                receipt_path.write_text('{"cidrs":["172.20.0.1/32"]}')
            return ["172.20.0.1/32"]

        def run(command, **kwargs):
            if 'get' in command and 'clusterrole' in command:
                return json.dumps({'metadata': {'uid': 'role-uid', 'resourceVersion': '3'},
                                   'rules': [{'apiGroups': [''], 'resources': ['nodes'], 'verbs': ['get', 'list']}]})
            if "get" in command and "networkpolicy" in command:
                return json.dumps({"spec": {}})
            if "get" in command:
                return json.dumps(next(reads))
            return ""

        gateway = validate_snapshot(fixture())
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(cli, "load_task_selection"),
            mock.patch.object(
                cli,
                "read_gateway_snapshot",
                side_effect=[gateway, {} if gateway_changed else gateway, gateway],
            ) as snapshots,
            mock.patch.object(cli, "load_runtime_images", return_value={}),
            mock.patch.object(cli, "api_cidrs", side_effect=discover),
            mock.patch.multiple(
                cli,
                verify_api_routes=mock.DEFAULT,
                verify_api_route_reader_rbac=mock.DEFAULT,
            ),
            mock.patch.object(cli, "api_policy_patch", return_value=([], {})),
            mock.patch.object(cli, "executable", side_effect=lambda s: s),
            mock.patch.object(cli, "kubernetes_environment", return_value={}),
            mock.patch.object(
                cli,
                "port_forward",
                side_effect=lambda *a: contextlib.nullcontext(12345),
            ),
            mock.patch.object(cli, "require_idle_runtime"),
            mock.patch.object(cli, "cached_registry_image", return_value=digest),
            mock.patch.object(cli, "run_command", side_effect=run) as commands,
            mock.patch.object(
                cli, "modal_observer_upgrade_manifest", return_value=None
            ),
            mock.patch.object(cli, "management_diagnostics_rbac", return_value=[]),
            mock.patch.object(
                cli,
                "runtime_upgrade_patches",
                return_value=(
                    [],
                    [
                        {
                            "path": "/spec/template/metadata/annotations",
                            "value": {"checksum": "new"},
                        }
                    ],
                    cm["data"],
                ),
            ),
            mock.patch.object(cli, "api_json"),
            mock.patch.object(cli, "state_directory", return_value=Path(tmp)),
            mock.patch.object(cli, "deploy_litellm") as install,
            mock.patch.object(cli, "platform_outputs", return_value={'cluster_name': 'frontiercs2-preview'}) as outputs,
        ):
            if gateway_changed:
                with self.assertRaisesRegex(DeployError, "changed before"):
                    cli.upgrade_runtime(config, Path(tmp), Path("kubeconfig"))
                install.assert_not_called()
                for call in commands.call_args_list:
                    command = call.args[0]
                    if "apply" in command or "patch" in command:
                        self.assertIn("--dry-run=server", command)
                return
            cli.upgrade_runtime(config, Path(tmp), Path("kubeconfig"))
            install.assert_not_called()
            outputs.assert_called_once_with(config)
            self.assertEqual(snapshots.call_count, 3)
            self.assertTrue(
                (Path(tmp) / ("runtime-upgrade-" + "c" * 12) / "accepted.json").exists()
            )
            for call in commands.call_args_list:
                self.assertNotIn("frontier-litellm", call.args[0])
            for name in ("gateway-before.json", "gateway-after.json"):
                path = Path(tmp) / ("runtime-upgrade-" + "c" * 12) / name
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(json.loads(path.read_text()), gateway)
