#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("launcher.py")
SPEC = importlib.util.spec_from_file_location("codex_launcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_reasoning_is_optional_and_explicit_xhigh_is_preserved(self) -> None:
        with mock.patch.dict(launcher.os.environ, {"FRONTIER_CODEX_MODEL": "test-model"}, clear=True):
            self.assertFalse(any("model_reasoning_effort" in item for item in launcher.command("http://127.0.0.1:8082/v1")))
            launcher.os.environ["FRONTIER_CODEX_REASONING_EFFORT"] = "xhigh"
            self.assertIn('model_reasoning_effort="xhigh"', launcher.command("http://127.0.0.1:8082/v1"))
            launcher.os.environ["FRONTIER_CODEX_REASONING_EFFORT"] = 'xhigh"\nunsafe=true'
            with self.assertRaises(launcher.LaunchError):
                launcher.command("http://127.0.0.1:8082/v1")

    def test_explicit_direct_mode_uses_env_key_not_proxy_header(self) -> None:
        with mock.patch.dict(launcher.os.environ, {"FRONTIER_CODEX_MODEL": "fixture-model"}, clear=True):
            command = launcher.command("https://model.example/v1", direct=True)
        self.assertIn('model_providers.frontier.env_key="CODEX_API_KEY"', command)
        self.assertFalse(any("X-Frontier-Agent-Token" in item for item in command))
        self.assertNotIn("fixture-key", str(command))

    def test_direct_url_is_https_and_cannot_embed_credentials(self) -> None:
        self.assertEqual(launcher.direct_model_url("https://model.example/v1/"), "https://model.example/v1")
        for value in ("http://model.example/v1", "https://user:secret@model.example/v1", "https://model.example/v1?token=secret", "https://model.example/v1\nsecret", "https://model.example:bad/v1"):
            with self.assertRaises(launcher.LaunchError):
                launcher.direct_model_url(value)

    def test_prompt_delegates_protocol_to_task_and_skill(self) -> None:
        self.assertIn("/frontier/task/README.md", launcher.PROMPT)
        self.assertIn("frontier-submit", launcher.PROMPT)
        self.assertIn("frontier-gpu", launcher.PROMPT)
        self.assertIn("exactly once", launcher.PROMPT)
        self.assertNotIn("submission.tar yourself", launcher.PROMPT)

    def test_command_runs_ephemeral_codex(self) -> None:
        with mock.patch.dict(launcher.os.environ, {"FRONTIER_CODEX_MODEL": "frontier-model"}, clear=True):
            command = launcher.command("http://127.0.0.1:8082/v1")
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn('model_providers.frontier.name="Frontier LiteLLM"', command)
        self.assertIn("model_providers.frontier.requires_openai_auth=false", command)
        self.assertFalse(any("model_providers.frontier.env_key" in value for value in command))
        self.assertEqual(command[-1], "-")

    def test_provider_retries_use_codex_supported_maximum_in_both_modes(self) -> None:
        with mock.patch.dict(launcher.os.environ, {"FRONTIER_CODEX_MODEL": "frontier-model"}, clear=True):
            for direct in (False, True):
                command = launcher.command("https://model.example/v1", direct=direct)
                self.assertIn("model_providers.frontier.request_max_retries=100", command)
                self.assertIn("model_providers.frontier.stream_max_retries=100", command)
                self.assertIn("allow_login_shell=false", command)

    def test_endpoint_rejects_nonlocal_shape(self) -> None:
        with mock.patch.dict(launcher.os.environ, {"BAD": "https://example.com:8080"}, clear=True):
            with self.assertRaises(launcher.LaunchError):
                launcher._endpoint("BAD", 8080, "")

    def test_final_receipt_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            receipt.write_text(json.dumps({"schema_version": "frontier.candidate-submit-result/v1", "operation": "final", "state": "accepted"}))
            with mock.patch.object(launcher, "FINAL_RECEIPT", receipt):
                launcher.validate_final_receipt()

    def test_nonzero_codex_exit_fails_even_if_receipt_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "final-submission.json"

            def primary(*_args, **_kwargs):
                receipt.write_text(
                    json.dumps(
                        {
                            "schema_version": "frontier.candidate-submit-result/v1",
                            "operation": "final",
                            "state": "accepted",
                        }
                    ),
                    encoding="utf-8",
                )
                return mock.Mock(returncode=1)

            with (
                mock.patch.object(launcher, "FINAL_RECEIPT", receipt),
                mock.patch.object(
                    launcher,
                    "prepare_environment",
                    return_value={
                        "FRONTIER_MODEL_API_BASE_URL": "http://127.0.0.1:8082/v1"
                    },
                ),
                mock.patch.object(launcher, "command", return_value=["codex"]),
                mock.patch.object(
                    launcher.subprocess, "run", side_effect=primary
                ) as run_codex,
                mock.patch.object(launcher, "cleanup"),
                mock.patch.dict(launcher.os.environ, {}, clear=True),
            ):
                self.assertEqual(launcher.main(), 2)
            run_codex.assert_called_once()

    def test_successful_codex_requires_accepted_final_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "final-submission.json"
            with (
                mock.patch.object(launcher, "FINAL_RECEIPT", receipt),
                mock.patch.object(
                    launcher,
                    "prepare_environment",
                    return_value={
                        "FRONTIER_MODEL_API_BASE_URL": "http://127.0.0.1:8082/v1"
                    },
                ),
                mock.patch.object(launcher, "command", return_value=["codex"]),
                mock.patch.object(
                    launcher.subprocess,
                    "run",
                    return_value=mock.Mock(returncode=0),
                ) as run_codex,
                mock.patch.object(launcher, "cleanup"),
                mock.patch.dict(launcher.os.environ, {}, clear=True),
            ):
                self.assertEqual(launcher.main(), 2)
            run_codex.assert_called_once()

    def test_environment_exposes_immutable_skill_and_submit_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            codex = root / "codex"
            task = root / "README.md"
            submit = root / "frontier-submit"
            skill_root = root / "skills"
            skill = skill_root / "frontier-submit" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            for path in (codex, task, submit, skill):
                path.write_text("x", encoding="utf-8")
            gpu_skill = skill_root / "frontier-gpu" / "SKILL.md"
            gpu_skill.parent.mkdir()
            for path in (gpu_skill, root / "frontier-gpu", root / "frontier-gpu-connect-proxy"):
                path.write_text("x", encoding="utf-8")
            runtime = {
                "CODEX_HOME": workspace / ".codex",
                "HOME": workspace / ".home",
                "TMPDIR": workspace / ".tmp",
            }
            environment = {
                "CODEX_API_KEY": "model-key",
                "FRONTIER_MODEL_PROXY_TOKEN": "proxy-token",
                "FRONTIER_DRIVER_API_URL": "http://127.0.0.1:8080",
                "FRONTIER_MODEL_API_BASE_URL": "http://127.0.0.1:8082/v1",
            }
            with mock.patch.multiple(
                launcher,
                WORKSPACE=workspace,
                CODEX=codex,
                TASK_README=task,
                SKILLS=skill_root,
                SUBMIT_BIN=submit.parent,
                RUNTIME_DIRS=runtime,
            ), mock.patch.dict(launcher.os.environ, environment, clear=True):
                child = launcher.prepare_environment()
                installed_skills = runtime["CODEX_HOME"] / "skills"
                self.assertTrue(installed_skills.is_dir())
                self.assertEqual((installed_skills / "frontier-gpu").resolve(), gpu_skill.parent)
                self.assertFalse(installed_skills.is_symlink())
                self.assertEqual(
                    (installed_skills / "frontier-submit").resolve(),
                    (skill_root / "frontier-submit").resolve(),
                )
                self.assertTrue(child["PATH"].startswith(str(submit.parent)))
                self.assertNotIn("CODEX_API_KEY", child)
                with mock.patch.dict(launcher.os.environ, {
                    "FRONTIER_MODEL_DIRECT_BASE_URL": "https://model.example/v1",
                    "FRONTIER_DRIVER_API_URL": "http://127.0.0.1:8080",
                    "CODEX_API_KEY": "fixture-direct-key",
                }, clear=True):
                    direct_child = launcher.prepare_environment()
                self.assertEqual(direct_child["CODEX_API_KEY"], "fixture-direct-key")
                self.assertNotIn("FRONTIER_MODEL_PROXY_TOKEN", direct_child)
                self.assertEqual(direct_child["FRONTIER_MODEL_DIRECT_BASE_URL"], "https://model.example/v1")


if __name__ == "__main__":
    unittest.main()
