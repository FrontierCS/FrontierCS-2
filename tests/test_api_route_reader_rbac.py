import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import json
import unittest
from pathlib import Path
from unittest import mock

from preview import api_routes, infra
from test_frontier_deploy import Fixture


class ApiRouteReaderRbacTests(unittest.TestCase):
    def test_precise_namespace_resources_verbs_and_subject(self):
        role, binding = infra.management_api_routes_rbac()
        self.assertEqual(
            role["metadata"],
            {"name": "frontier-management-api-routes", "namespace": "default"},
        )
        self.assertEqual(
            role["rules"],
            [
                {
                    "apiGroups": [""],
                    "resources": ["services", "endpoints"],
                    "resourceNames": ["kubernetes"],
                    "verbs": ["get"],
                },
                {
                    "apiGroups": ["discovery.k8s.io"],
                    "resources": ["endpointslices"],
                    "verbs": ["list"],
                },
            ],
        )
        self.assertEqual(binding["metadata"], role["metadata"])
        self.assertEqual(
            binding["subjects"],
            [
                {
                    "kind": "ServiceAccount",
                    "name": "frontier-management",
                    "namespace": "frontier-system",
                }
            ],
        )
        self.assertEqual(
            binding["roleRef"],
            {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": role["metadata"]["name"],
            },
        )

    def test_readback_accepts_only_exact_grant(self):
        expected = infra.management_api_routes_rbac()
        for mutation in ("none", "secret", "write", "extra-subject", "wrong-role"):
            objects = copy.deepcopy(expected)
            if mutation == "secret":
                objects[0]["rules"][0]["resources"].append("secrets")
            if mutation == "write":
                objects[0]["rules"][0]["verbs"].append("patch")
            if mutation == "extra-subject":
                objects[1]["subjects"].append(
                    {"kind": "ServiceAccount", "name": "agent", "namespace": "default"}
                )
            if mutation == "wrong-role":
                objects[1]["roleRef"]["name"] = "cluster-admin"
            with (
                self.subTest(mutation=mutation),
                mock.patch.object(api_routes, "executable", return_value="kubectl"),
                mock.patch.object(
                    api_routes, "kubernetes_environment", return_value={}
                ),
                mock.patch.object(
                    api_routes,
                    "run_command",
                    side_effect=[json.dumps(o) for o in objects],
                ) as command,
            ):
                if mutation == "none":
                    api_routes.verify_api_route_reader_rbac(Path("kubeconfig"))
                else:
                    with self.assertRaises(infra.DeployError):
                        api_routes.verify_api_route_reader_rbac(Path("kubeconfig"))
                self.assertTrue(
                    all(
                        "get" in c.args[0] and "default" in c.args[0]
                        for c in command.call_args_list
                    )
                )

    def test_fresh_manifest_grants_only_native_management(self):
        raw = infra.load_json(infra.REPOSITORY_ROOT / "config/site.example.json")
        from preview.config import validate_config

        config = validate_config(raw)
        for simulator in (True, False):
            rendered = json.loads(
                infra.render_runtime_manifest(
                    config,
                    Fixture.outputs(config),
                    Fixture.images(config),
                    ["172.20.0.1/32"],
                    simulator_bootstrap=simulator,
                )
            )
            found = [
                x
                for x in rendered["items"]
                if x.get("metadata", {}).get("name") == "frontier-management-api-routes"
            ]
            self.assertEqual(
                found, [] if simulator else infra.management_api_routes_rbac()
            )


if __name__ == "__main__":
    unittest.main()
