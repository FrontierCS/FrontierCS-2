import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("zcode_launcher", Path(__file__).with_name("launcher.py"))
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class LauncherTests(unittest.TestCase):
    def test_run_local_endpoint_rejects_provider_and_other_port(self):
        for value in ("https://provider.example/v1", "http://127.0.0.1:8083/v1",
                      "http://secret@127.0.0.1:8082/v1", "http://127.0.0.1:8082/v1?key=x"):
            with patch.dict(os.environ, {"FRONTIER_MODEL_API_BASE_URL": value}):
                with self.assertRaises(launch.LaunchError):
                    launch.endpoint("FRONTIER_MODEL_API_BASE_URL", 8082, "/v1")

    def test_config_binds_model_and_max_effort_without_legacy_import(self):
        result = launch.provider_config("http://10.0.0.1:8082/v1", "glm-5-3", "run-token", "max")
        config = result["config"]
        provider = config["providerConfigRules"]["providerRules"][0]
        self.assertEqual(provider["config"]["access"]["apiKey"], "run-token")
        self.assertEqual(config["defaultModelSelection"]["options"]["reasoningLevel"], "max")
        model = config["modelConfigRules"]["providerModelRules"][0]["config"]
        self.assertEqual(model["properties"]["contextWindow"], 1048576)
        self.assertEqual(model["optionSpecs"]["maxOutputTokens"]["max"], 262144)
        self.assertEqual(model["optionSpecs"]["reasoningLevel"]["map"], '{"reasoning_effort": reasoningLevel}')

    def test_saved_answer_is_not_an_accepted_final(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launch, "WORKSPACE", Path(temp)):
            path = Path(temp) / ".frontier/final-submission.json"
            path.parent.mkdir()
            result = {"schema_version": "frontier.candidate-submit-result/v1", "operation": "store-final",
                      "state": "stored", "payload_id": "payload_x", "verification_id": "verification_x"}
            path.write_text(json.dumps(result)); path.chmod(0o600)
            self.assertFalse(launch.accepted_final())
            result.update(operation="final", state="accepted")
            path.write_text(json.dumps(result))
            self.assertTrue(launch.accepted_final())
            path.chmod(0o644)
            self.assertFalse(launch.accepted_final())

    def test_official_headless_command_has_no_private_timeout(self):
        command = launch.command()
        self.assertIn("--output-format", command)
        self.assertIn("stream-json", command)
        self.assertIn("yolo", command)
        self.assertIn("--require", command)
        self.assertIn("zcode.cjs", command[3])
        self.assertEqual(Path(command[0]).name, "node")
        self.assertNotIn("--no-wait", " ".join(command))
        self.assertNotIn("--timeout", command)


if __name__ == "__main__":
    unittest.main()
