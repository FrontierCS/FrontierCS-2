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
        self.config = {
            "network": {"service_ipv4_cidr": "172.20.0.0/16", "vpc_cidr": "10.42.0.0/16"}
        }
        self.allowed = ["10.42.200.0/28", "10.42.200.16/28", "172.20.0.1/32"]
        self.outputs = {"kubernetes_api_cidrs": list(self.allowed)}

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
                    "addresses": [{"ip": "10.42.200.6"}, {"ip": "10.42.200.20"}],
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
                            "addresses": ["10.42.200.6", "10.42.200.20"],
                            "conditions": {"ready": True},
                        }
                    ],
                }
            ],
        }
        self.expected = ["10.42.200.20/32", "10.42.200.6/32", "172.20.0.1/32"]

    def snapshot(self):
        return routes.route_snapshot(
            self.config, self.service, self.endpoints, self.slices
        )

    def test_external_public_dns_is_not_an_authority(self):
        # A public endpoint seen by the installer must not replace private DNAT peers.
        with mock.patch(
            "socket.getaddrinfo", return_value=[(2, 1, 6, "", ("50.18.190.35", 443))]
        ) as dns:
            self.assertEqual(self.snapshot()["observed_cidrs"], self.expected)
            dns.assert_not_called()

    def discover(self):
        values = [self.service, self.endpoints, self.slices] * 2
        with (
            mock.patch.object(routes, "executable", return_value="kubectl"),
            mock.patch.object(routes, "kubernetes_environment", return_value={}),
            mock.patch.object(
                routes, "run_command", side_effect=[json.dumps(v) for v in values]
            ),
        ):
            return routes.discover_api_routes(
                self.config, Path("kubeconfig"), self.allowed
            )

    def rotate(self, *addresses):
        self.endpoints["subsets"][0]["addresses"] = [{"ip": a} for a in addresses]
        self.slices["items"][0]["endpoints"][0]["addresses"] = list(addresses)

    def test_rotated_endpoints_inside_the_control_plane_subnets_are_admitted(self):
        # EKS replaces its ENIs; any address inside the dedicated /28s is fine.
        for addresses in (("10.42.200.6", "10.42.200.20"), ("10.42.200.14", "10.42.200.17")):
            with self.subTest(addresses=addresses):
                self.rotate(*addresses)
                snapshot = self.discover()
                self.assertEqual(snapshot["allowed_cidrs"], self.allowed)
                self.assertEqual(
                    snapshot["observed_cidrs"],
                    sorted([a + "/32" for a in addresses] + ["172.20.0.1/32"]),
                )

    def test_endpoints_outside_the_allowed_cidrs_are_refused_by_name(self):
        for outside in ("10.42.0.60", "10.42.200.32", "50.18.190.35"):
            with self.subTest(outside=outside):
                self.rotate("10.42.200.6", outside)
                with self.assertRaisesRegex(
                    routes.DeployError, rf"\['{outside}/32'\] lie outside"
                ):
                    self.discover()

    def test_terraform_allowlist_is_strict(self):
        cli_infra = routes.allowed_api_cidrs
        self.assertEqual(cli_infra(self.config, self.outputs), self.allowed)
        bad = [
            None,
            [],
            ["172.20.0.1/32"],  # no control-plane subnet
            ["10.42.200.0/28", "10.42.200.16/28"],  # no Service ClusterIP
            ["10.42.200.0/27", "172.20.0.1/32"],  # wider than /28
            ["10.42.0.0/16", "172.20.0.1/32"],
            ["10.42.200.1/28", "172.20.0.1/32"],  # not canonical
            ["10.43.200.0/28", "172.20.0.1/32"],  # outside the VPC
            ["50.18.190.32/28", "172.20.0.1/32"],  # public
            ["172.20.0.1/32", "10.42.200.0/28"],  # unsorted
            ["10.42.200.0/28", "10.42.200.0/28", "172.20.0.1/32"],
            ["10.42.200.0/28", "10.42.200.8/29", "172.20.0.1/32"],  # overlap
            [f"10.42.{i}.0/28" for i in range(8)] + ["172.20.0.1/32"],  # too many
            ["2001:db8::/124", "172.20.0.1/32"],
        ]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(routes.DeployError):
                    cli_infra(self.config, {"kubernetes_api_cidrs": value})

    def test_transitional_exact_pins_extend_the_terraform_allowlist(self):
        self.config["network"]["transitional_kubernetes_api_cidrs"] = [
            "10.42.1.54/32",
            "10.42.0.60/32",
        ]
        allowed = routes.allowed_api_cidrs(self.config, self.outputs)
        self.assertEqual(
            allowed,
            ["10.42.0.60/32", "10.42.1.54/32", "10.42.200.0/28", "10.42.200.16/28", "172.20.0.1/32"],
        )
        self.rotate("10.42.0.60", "10.42.1.54")  # before the cluster subnet move
        self.allowed = allowed
        self.discover()
        self.rotate("10.42.200.3", "10.42.200.30")  # after it
        self.discover()
        self.config["network"]["transitional_kubernetes_api_cidrs"] = ["10.42.200.5/32"]
        with self.assertRaisesRegex(routes.DeployError, "overlap"):
            routes.allowed_api_cidrs(self.config, self.outputs)

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
                routes.discover_api_routes(
                    self.config, Path("kubeconfig"), self.allowed
                )

    def test_policy_cas_preserves_non_api_rules(self):
        # Migration: the live policy still pins the old exact ENI /32s.
        old = ["10.42.0.168/32", "10.42.1.62/32", "172.20.0.1/32"]
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
        patch, spec = routes.api_policy_patch(policy, self.allowed, old)
        self.assertEqual(policy, original)
        self.assertEqual(
            patch[0],
            {"op": "test", "path": "/metadata/resourceVersion", "value": "123"},
        )
        self.assertEqual(spec["egress"][0], original["spec"]["egress"][0])
        self.assertEqual(spec["podSelector"], original["spec"]["podSelector"])
        self.assertEqual(
            spec["egress"][1]["to"], [{"ipBlock": {"cidr": c}} for c in self.allowed]
        )
        self.assertEqual(spec["egress"][1]["ports"], [{"port": 443, "protocol": "TCP"}])
        with self.assertRaises(routes.DeployError):
            routes.api_policy_patch(policy, self.allowed, ["10.0.0.0/8"])
        for wide in (["10.0.0.0/8"], ["10.42.200.0/24"], ["0.0.0.0/0"]):
            with self.subTest(wide=wide), self.assertRaises(routes.DeployError):
                routes.api_policy_patch(policy, wide, old)

    def test_verify_rejects_rotated_routes_without_any_mutation(self):
        with (
            mock.patch.object(routes, "verify_api_route_reader_rbac"),
            mock.patch.object(routes, "discover_api_routes", return_value={}),
            mock.patch.object(routes, "executable", return_value="kubectl"),
            mock.patch.object(routes, "kubernetes_environment", return_value={}),
            mock.patch.object(
                routes,
                "run_command",
                return_value=json.dumps(
                    {
                        "data": {
                            "MANAGEMENT_KUBERNETES_API_CIDRS": "10.42.0.168/32,10.42.1.62/32,172.20.0.1/32"
                        }
                    }
                ),
            ) as commands,
        ):
            with self.assertRaisesRegex(routes.DeployError, "idle upgrade-runtime"):
                routes.verify_api_routes(self.config, self.outputs, Path("kubeconfig"))
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
            cm, dep, images, kubernetes_api_cidrs=self.allowed
        )
        self.assertEqual(
            data["MANAGEMENT_KUBERNETES_API_CIDRS"], ",".join(self.allowed)
        )
        self.assertEqual(data["agent_budget"], "18000")
        self.assertEqual(data["MANAGEMENT_MODAL_ALLOWED_CIDRS"], "1.2.3.4/32")
        self.assertTrue(
            any(p["path"] == "/data/MANAGEMENT_KUBERNETES_API_CIDRS" for p in patch)
        )
        for wide in (["10.0.0.0/8"], ["10.42.200.0/27"], ["50.18.190.35/32"]):
            with self.subTest(wide=wide), self.assertRaises(routes.DeployError):
                cli.runtime_upgrade_patches(cm, dep, images, kubernetes_api_cidrs=wide)


if __name__ == "__main__":
    unittest.main()
