"""An optional named ECR Secret reference never widens site permissions."""
import deployment_test_guard  # noqa: F401

import copy
import json
import sys
import unittest

import jsonschema
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_deploy
from test_frontier_deploy import Fixture


class ModalEcrReferenceTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / "config/site.example.json").read_text())
        self.raw["features"]["modal_gpu"] = True

    def render(self, raw):
        config = frontier_deploy.validate_config(raw)
        return frontier_deploy.render_runtime_manifest(
            config, Fixture.outputs(config), Fixture.images(config),
            ["172.20.0.1/32", "10.42.1.20/32"],
        )

    def test_omitted_or_explicit_empty_preserves_public_image_behavior(self):
        omitted = self.render(self.raw)
        self.raw["modal"]["ecr_pull_secret_ref"] = ""
        self.assertEqual(omitted, self.render(self.raw))

    def test_named_reference_changes_only_reference_and_management_rollout_hash(self):
        expected = json.loads(self.render(self.raw))
        configmap = next(item for item in expected["items"]
                         if item["kind"] == "ConfigMap"
                         and item["metadata"]["name"] == "frontier-management-config")
        configmap["data"]["MANAGEMENT_MODAL_ECR_PULL_SECRET_REF"] = "modal-ecr-pull"
        deployment = next(item for item in expected["items"]
                          if item["kind"] == "Deployment"
                          and item["metadata"]["name"] == "frontier-management")
        annotations = deployment["spec"]["template"]["metadata"]["annotations"]
        previous_hash = annotations["frontier.dev/config-sha256"]
        annotations["frontier.dev/config-sha256"] = frontier_deploy.canonical_object_sha256(configmap["data"])
        self.assertNotEqual(previous_hash, annotations["frontier.dev/config-sha256"])
        self.raw["modal"]["ecr_pull_secret_ref"] = "modal-ecr-pull"
        self.assertEqual(expected, json.loads(self.render(self.raw)))

    def test_malformed_reference_and_secret_values_are_rejected(self):
        for value in (None, True, False, 0, [], {}, " bad", "bad ", "Secret_Name",
                      "../../role", "arn:aws:iam::123456789012:role/ecr", "x" * 64):
            with self.subTest(value=value):
                self.raw["modal"]["ecr_pull_secret_ref"] = value
                with self.assertRaises(frontier_deploy.DeployError):
                    frontier_deploy.validate_config(self.raw)

    def test_json_schema_and_executable_validator_accept_the_same_references(self):
        schema = json.loads((ROOT / "config/site.schema.json").read_text())
        validator = jsonschema.Draft202012Validator(schema)
        for value in ("", "modal-ecr-pull", "a" * 63):
            with self.subTest(valid=value):
                self.raw["modal"]["ecr_pull_secret_ref"] = value
                validator.validate(self.raw)
                frontier_deploy.validate_config(self.raw)
        for value in (None, True, 0, [], {}, "a", "a" * 64, "bad\n", "Secret_Name", "../../role"):
            with self.subTest(invalid=value):
                self.raw["modal"]["ecr_pull_secret_ref"] = value
                with self.assertRaises(jsonschema.ValidationError):
                    validator.validate(self.raw)
                with self.assertRaises(frontier_deploy.DeployError):
                    frontier_deploy.validate_config(self.raw)

    def test_extra_or_missing_keys_still_reject(self):
        for change in ("extra", "missing"):
            with self.subTest(change=change):
                value = copy.deepcopy(self.raw)
                value["modal"]["ecr_pull_secret_ref"] = "modal-ecr-pull"
                if change == "extra":
                    value["modal"]["unknown_registry_authority"] = "unapproved"
                else:
                    del value["modal"]["credentials_secret"]
                with self.assertRaises(frontier_deploy.DeployError):
                    frontier_deploy.validate_config(value)

    def test_reference_does_not_enable_gpu_or_change_cpu_site_rbac(self):
        self.raw["features"]["modal_gpu"] = False
        unchanged = self.render(self.raw)
        self.raw["modal"]["ecr_pull_secret_ref"] = "modal-ecr-pull"
        self.assertEqual(unchanged, self.render(self.raw))


if __name__ == "__main__":
    unittest.main()
