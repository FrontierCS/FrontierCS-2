"""Offline config/schema and bundle coverage for optional EKS operators."""

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_deploy as deploy
from preview import config as site_config


class OperatorPrincipalsTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/site.example.json").read_text())
        self.schema = json.loads((ROOT / "config/site.schema.json").read_text())

    def test_omitted_empty_and_user_role_paths_are_accepted(self):
        for principals in (
            None,
            [],
            [
                "arn:aws:iam::123456789012:user/operator",
                "arn:aws:iam::123456789012:role/team/preview-admin",
            ],
            ["arn:aws-cn:iam::123456789012:role/team/admin"],
            ["arn:aws-us-gov:iam::123456789012:user/operator"],
        ):
            raw = copy.deepcopy(self.config)
            if principals is not None:
                raw["terraform"]["operator_principals"] = principals
            with self.subTest(principals=principals):
                jsonschema.validate(raw, self.schema)
                validated = deploy.validate_config(raw)
                # Optional defaults must not rewrite the existing config/release hash.
                self.assertEqual(validated["terraform"], raw["terraform"])

    def test_invalid_principals_fail_in_both_validators(self):
        arn = "arn:aws:iam::123456789012:user/operator"
        invalid = (
            None,
            arn,
            {},
            [None],
            [1],
            [True],
            [arn, arn],
            [""],
            [arn + "\n"],
            [" " + arn],
            [arn + " "],
            ["arn:aws:sts::123456789012:assumed-role/admin/session"],
            ["arn:aws:iam::123456789012:root"],
            ["arn:aws:iam::123456789012:policy/admin"],
            ["arn:aws:iam::123:user/operator"],
            ["arn:aws:iam::123456789012:role/*"],
            ["arn:aws:iam::123456789012:role/team//admin"],
            ["arn:aws:iam::123456789012:user/operator/"],
        )
        for principals in invalid:
            raw = copy.deepcopy(self.config)
            raw["terraform"]["operator_principals"] = principals
            with self.subTest(principals=principals):
                with self.assertRaisesRegex(deploy.DeployError, "operator_principals"):
                    deploy.validate_config(raw)
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(raw, self.schema)

    def test_bundle_passes_exact_list_and_empty_default_without_state_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for principals in (
                None,
                [],
                [
                    "arn:aws:iam::123456789012:user/operator",
                    "arn:aws:iam::123456789012:role/team/admin",
                ],
            ):
                raw = copy.deepcopy(self.config)
                if principals is not None:
                    raw["terraform"]["operator_principals"] = principals
                config_path = root / "site.json"
                config_path.write_text(json.dumps(raw))
                with (
                    self.subTest(principals=principals),
                    mock.patch.object(
                        site_config,
                        "generated_directory",
                        return_value=root / "generated",
                    ),
                    mock.patch.object(
                        site_config,
                        "state_directory",
                        return_value=root / "unused-state",
                    ),
                ):
                    _, output = deploy.render_bundle(config_path)
                    variables = json.loads(
                        (output / "platform.auto.tfvars.json").read_text()
                    )
                    self.assertEqual(variables["operator_principals"], principals or [])
                    self.assertFalse((root / "unused-state").exists())


if __name__ == "__main__":
    unittest.main()
