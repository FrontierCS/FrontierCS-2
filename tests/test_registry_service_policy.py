# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import unittest
from unittest import mock

from tests.test_frontier_deploy import load_storage_controller


class RegistryServicePolicyTests(unittest.TestCase):
    def setUp(self):
        self.c = load_storage_controller()
        self.namespace = "frontier-run-" + "0" * 26
        self.run_id = "run_" + "0" * 26
        self.attempt_id = "attempt_" + "1" * 26

    def policy(self, cidr="172.20.0.50/32"):
        return self.c.registry_policy_document(
            self.namespace, self.run_id, self.attempt_id, cidr
        )

    def test_exact_driver_identity_and_single_registry_port(self):
        policy = self.policy()
        self.assertEqual(
            policy["spec"]["podSelector"]["matchLabels"],
            {
                "frontier.dev/owner": "management",
                "frontier.dev/network-role": "driver",
                "frontier.dev/run-id": self.run_id,
                "frontier.dev/attempt-id": self.attempt_id,
            },
        )
        self.assertEqual(
            policy["spec"]["egress"],
            [
                {
                    "to": [{"ipBlock": {"cidr": "172.20.0.50/32"}}],
                    "ports": [{"protocol": "TCP", "port": 5000}],
                }
            ],
        )
        self.assertNotIn("ingress", policy["spec"])

    def test_broad_ipv6_or_noncanonical_network_is_rejected(self):
        for cidr in ("0.0.0.0/0", "172.20.0.0/16", "172.20.0.50/24", "::1/128"):
            with self.subTest(cidr=cidr), self.assertRaises(ValueError):
                self.policy(cidr)

    def test_foreign_policy_is_never_adopted(self):
        desired = self.policy()
        foreign = copy.deepcopy(desired)
        foreign["metadata"]["labels"]["frontier.dev/owner"] = "someone-else"
        with mock.patch.object(self.c, "api", return_value=foreign) as api:
            with self.assertRaisesRegex(RuntimeError, "ownership"):
                self.c.create_or_replace_network_policy(self.namespace, desired)
            self.assertEqual(api.call_count, 1)

    def test_registry_reconciliation_uses_separate_policy_name(self):
        desired = self.policy()
        with mock.patch.object(self.c, "api", side_effect=[None, {}, desired]) as api:
            self.c.create_or_replace_network_policy(self.namespace, desired)
        self.assertTrue(
            api.call_args_list[0].args[1].endswith("/frontier-driver-local-registry")
        )
        self.assertEqual(api.call_args_list[1].args[0], "POST")

    def test_unowned_namespace_does_not_receive_policy_or_storage(self):
        with (
            mock.patch.object(self.c, "REGISTRY_CIDR", "172.20.0.50/32"),
            mock.patch.object(self.c, "api") as api,
        ):
            self.c.reconcile_namespace(
                {"metadata": {"name": self.namespace, "labels": {}}}
            )
        api.assert_not_called()
