import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import json
import unittest
from pathlib import Path
from unittest import mock

from preview import api_routes as routes
from preview import cli


class ApiRouteTests(unittest.TestCase):
    def setUp(self):
        self.config = {"network": {"service_ipv4_cidr": "172.20.0.0/16"}}

        def meta(name="kubernetes"):
            return {
                "name": name,
                "namespace": "default",
                "uid": "uid-" + name,
                "resourceVersion": "12",
            }

        self.ports = [{"name": "https", "port": 443, "protocol": "TCP"}]
        self.service = {
            "kind": "Service",
            "metadata": meta(),
            "spec": {"clusterIP": "172.20.0.1", "ports": self.ports},
        }
        self.endpoints = {
            "kind": "Endpoints",
            "metadata": meta(),
            "subsets": [
                {
                    "addresses": [{"ip": "10.42.0.106"}, {"ip": "10.42.1.204"}],
                    "ports": self.ports,
                }
            ],
        }
        self.slices = {
            "kind": "List",
            "items": [
                {
                    "kind": "EndpointSlice",
                    "metadata": {
                        **meta(),
                        "labels": {"kubernetes.io/service-name": "kubernetes"},
                    },
                    "addressType": "IPv4",
                    "ports": self.ports,
                    "endpoints": [
                        {
                            "addresses": ["10.42.0.106", "10.42.1.204"],
                            "conditions": {"ready": True},
                        }
                    ],
                }
            ],
        }
        self.expected = ["10.42.0.106/32", "10.42.1.204/32", "172.20.0.1/32"]

    def snapshot(self):
        return routes.route_snapshot(
            self.config, self.service, self.endpoints, self.slices
        )

    def test_external_public_dns_is_not_an_authority(self):
        # A public endpoint seen by the installer must not replace private DNAT peers.
        with mock.patch(
            "socket.getaddrinfo", return_value=[(2, 1, 6, "", ("50.18.190.35", 443))]
        ) as dns:
            self.assertEqual(self.snapshot()["cidrs"], self.expected)
            dns.assert_not_called()

    def test_mixed_public_private_service_targets_are_exact_not_vpc_ranges(self):
        self.endpoints["subsets"][0]["addresses"].append({"ip": "50.18.190.35"})
        self.slices["items"][0]["endpoints"][0]["addresses"].append("50.18.190.35")
        self.assertEqual(self.snapshot()["cidrs"], self.expected + ["50.18.190.35/32"])

    def test_incomplete_mismatched_and_transitioning_discovery_fails_closed(self):
        mutations = [
            lambda: self.service["metadata"].update(namespace="other"),
            lambda: self.service["spec"].update(clusterIP="172.20.0.2"),
            lambda: self.slices["items"][0]["metadata"]["labels"].update(
                {"kubernetes.io/service-name": "other"}
            ),
            lambda: self.slices["items"][0]["endpoints"][0].update(
                addresses=["10.42.0.106"]
            ),
            lambda: self.slices["items"][0]["endpoints"][0]["conditions"].update(
                ready=False
            ),
            lambda: self.slices["items"][0]["endpoints"][0]["conditions"].update(
                terminating=True
            ),
            lambda: self.slices.update(items=[]),
            lambda: self.endpoints["subsets"][0].update(
                notReadyAddresses=[{"ip": "10.42.2.1"}]
            ),
            lambda: self.ports[0].update(port=6443),
            lambda: self.endpoints["subsets"][0]["addresses"][0].update(
                ip="10.0.0.0/8"
            ),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.setUp()
                mutate()
                with self.assertRaises(routes.DeployError):
                    self.snapshot()

    def test_two_snapshots_must_have_same_identity_and_routes(self):
        values = [self.service, self.endpoints, self.slices]
        changed = copy.deepcopy(self.endpoints)
        changed["metadata"]["resourceVersion"] = "13"
        with (
            mock.patch.object(routes, "executable", return_value="kubectl"),
            mock.patch.object(routes, "kubernetes_environment", return_value={}),
            mock.patch.object(
                routes,
                "run_command",
                side_effect=[
                    json.dumps(v) for v in values + [self.service, changed, self.slices]
                ],
            ),
        ):
            with self.assertRaisesRegex(routes.DeployError, "changed during"):
                routes.discover_api_routes(self.config, Path("kubeconfig"))

    def test_policy_cas_preserves_non_api_rules(self):
        old = ["172.20.0.1/32", "50.18.190.35/32"]
        policy = {
            "metadata": {"resourceVersion": "123"},
            "spec": {
                "egress": [
                    {
                        "ports": [{"protocol": "UDP", "port": 53}],
                        "to": [{"namespaceSelector": {}}],
                    },
                    {
                        "ports": [{"port": 443, "protocol": "TCP"}],
                        "to": [{"ipBlock": {"cidr": c}} for c in old],
                    },
                ],
                "ingress": [],
                "podSelector": {"matchLabels": {"role": "management"}},
            },
        }
        original = copy.deepcopy(policy)
        patch, spec = routes.api_policy_patch(policy, self.expected, old)
        self.assertEqual(policy, original)
        self.assertEqual(
            patch[0],
            {"op": "test", "path": "/metadata/resourceVersion", "value": "123"},
        )
        self.assertEqual(spec["egress"][0], original["spec"]["egress"][0])
        self.assertEqual(spec["podSelector"], original["spec"]["podSelector"])
        self.assertEqual(
            spec["egress"][1]["to"], [{"ipBlock": {"cidr": c}} for c in self.expected]
        )
        with self.assertRaises(routes.DeployError):
            routes.api_policy_patch(policy, self.expected, ["10.0.0.0/8"])

    def test_verify_rejects_rotated_routes_without_any_mutation(self):
        with (
            mock.patch.object(routes, "verify_api_route_reader_rbac"),
            mock.patch.object(
                routes, "discover_api_routes", return_value={"cidrs": self.expected}
            ),
            mock.patch.object(routes, "executable", return_value="kubectl"),
            mock.patch.object(routes, "kubernetes_environment", return_value={}),
            mock.patch.object(
                routes,
                "run_command",
                return_value=json.dumps(
                    {
                        "data": {
                            "MANAGEMENT_KUBERNETES_API_CIDRS": "172.20.0.1/32,50.18.190.35/32"
                        }
                    }
                ),
            ) as commands,
        ):
            with self.assertRaisesRegex(routes.DeployError, "idle upgrade-runtime"):
                routes.verify_api_routes(self.config, Path("kubeconfig"))
            self.assertEqual(commands.call_count, 1)
            self.assertIn("get", commands.call_args.args[0])

    def test_runtime_upgrade_binds_routes_without_changing_scientific_settings(self):
        cm = {
            "metadata": {"resourceVersion": "1"},
            "data": {
                "MANAGEMENT_DRIVER_BACKENDS": "local-k8s",
                "MANAGEMENT_DRIVER_IMAGE": "old",
                "MANAGEMENT_KUBERNETES_API_CIDRS": "50.18.190.35/32",
                "MANAGEMENT_MODAL_ALLOWED_CIDRS": "1.2.3.4/32",
                "agent_budget": "18000",
            },
        }
        dep = {
            "metadata": {"resourceVersion": "2"},
            "spec": {
                "template": {
                    "metadata": {},
                    "spec": {"containers": [{"name": "management", "image": "old"}]},
                }
            },
        }
        images = {k: {"reference": "new-" + k} for k in ("management", "driver")}
        patch, _, data = cli.runtime_upgrade_patches(
            cm, dep, images, kubernetes_api_cidrs=self.expected
        )
        self.assertEqual(
            data["MANAGEMENT_KUBERNETES_API_CIDRS"], ",".join(self.expected)
        )
        self.assertEqual(data["agent_budget"], "18000")
        self.assertEqual(data["MANAGEMENT_MODAL_ALLOWED_CIDRS"], "1.2.3.4/32")
        self.assertTrue(
            any(p["path"] == "/data/MANAGEMENT_KUBERNETES_API_CIDRS" for p in patch)
        )
        with self.assertRaises(routes.DeployError):
            cli.runtime_upgrade_patches(
                cm, dep, images, kubernetes_api_cidrs=["10.0.0.0/8"]
            )


if __name__ == "__main__":
    unittest.main()
