#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("launcher.py")
SPEC = importlib.util.spec_from_file_location("kimi_code_launcher", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_prompt_and_command_use_native_skill_support(self) -> None:
        self.assertIn("/frontier/task/README.md", launcher.PROMPT)
        self.assertIn("frontier-submit", launcher.PROMPT)
        self.assertIn("frontier-gpu", launcher.PROMPT)
        self.assertIn("representative local check", launcher.PROMPT)
        self.assertIn("do not call TodoList", launcher.PROMPT)
        self.assertIn("brief completion message with no tool call", launcher.PROMPT)
        command = launcher.command("kimi-k2.7-code")
        self.assertEqual(command[:4], [
            str(launcher.KIMI_LOADER), "--library-path", str(launcher.KIMI_LIB),
            str(launcher.NODE),
        ])
        self.assertIn("--prompt", command)
        self.assertIn("--skills-dir", command)
        self.assertNotIn("--yolo", command)

    def test_config_uses_chat_completions_provider_and_disables_web(self) -> None:
        config = launcher.render_config(
            "http://10.0.0.7:8082/v1", "kimi-k2.7-code", "run-token"
        )
        self.assertIn('type = "openai"', config)
        self.assertIn('base_url = "http://10.0.0.7:8082/v1"', config)
        self.assertIn('model = "kimi-k2.7-code"', config)
        self.assertIn('api_key = "run-token"', config)
        self.assertIn('disabled = ["WebSearch", "FetchURL"]', config)
        self.assertIn("[loop_control]", config)
        self.assertIn("max_steps_per_turn = 0", config)
        self.assertIn("max_context_size = 262144", config)

    def test_context_budget_is_explicit_and_does_not_add_an_output_cap(self) -> None:
        with mock.patch.dict(launcher.os.environ, {}, clear=True):
            self.assertEqual(launcher.context_tokens(), 262144)
        with mock.patch.dict(launcher.os.environ,
                             {"FRONTIER_KIMI_MAX_CONTEXT_TOKENS": "1048576"}, clear=True):
            budget = launcher.context_tokens()
        config = launcher.render_config("http://10.0.0.7:8082/v1", "kimi-k3",
                                        "run-token", max_context_tokens=budget)
        self.assertIn("max_context_size = 1048576", config)
        self.assertIn('model = "kimi-k3"', config)
        self.assertNotIn("max_output_size", config)
        self.assertNotIn("max_tokens", config)
        for invalid in ("", "0", "-1", "1.5", "1e6", " 1048576", "2147483648"):
            with self.subTest(invalid=invalid), mock.patch.dict(
                launcher.os.environ, {"FRONTIER_KIMI_MAX_CONTEXT_TOKENS": invalid}, clear=True
            ), self.assertRaises(launcher.LaunchError):
                launcher.context_tokens()

    def test_endpoint_rejects_public_or_wrong_path(self) -> None:
        for value in ("https://example.com/v1", "http://127.0.0.1:8082/v2"):
            with self.subTest(value=value), mock.patch.dict(
                launcher.os.environ, {"BAD": value}, clear=True
            ):
                with self.assertRaises(launcher.LaunchError):
                    launcher._endpoint("BAD", 8082, "/v1")

    def test_model_effort_is_opt_in_and_uses_native_model_configuration(self) -> None:
        with mock.patch.dict(launcher.os.environ, {}, clear=True):
            self.assertIsNone(launcher.thinking_efforts())
        previous = launcher.render_config("http://127.0.0.1:8082/v1", "kimi-k2.7-code", "token")
        self.assertNotIn("effort", previous)
        with mock.patch.dict(launcher.os.environ, {
            "FRONTIER_KIMI_SUPPORT_EFFORTS": '["low","high","max"]',
            "FRONTIER_KIMI_DEFAULT_EFFORT": "high",
        }, clear=True):
            config = tomllib.loads(launcher.render_config(
                "http://127.0.0.1:8082/v1", "kimi-k3", "token",
                efforts=launcher.thinking_efforts(),
            ))
        self.assertEqual(config["models"]["frontier"]["support_efforts"], ["low", "high", "max"])
        self.assertEqual(config["models"]["frontier"]["default_effort"], "high")
        self.assertNotIn("thinking", config)

    def test_model_effort_rejects_missing_malformed_and_unsupported_values(self) -> None:
        for supported, default in (
            (None, "high"), ('["high"]', None), ("", "high"),
            ('"high"', "high"), ('{"high":true}', "high"), ("[]", "high"),
            ('[null]', "high"), ('[true]', "high"), ('[["high"]]', "high"),
            ('["high","high"]', "high"), ('["unknown"]', "unknown"),
            ('["low","high","max"]', "medium"), ('["high"]', "HIGH"),
            ('["high"]', " high"), ('["high"]', ""), (" " * 257, "high"),
        ):
            environment = {}
            if supported is not None:
                environment["FRONTIER_KIMI_SUPPORT_EFFORTS"] = supported
            if default is not None:
                environment["FRONTIER_KIMI_DEFAULT_EFFORT"] = default
            with self.subTest(supported=supported, default=default), mock.patch.dict(
                launcher.os.environ, environment, clear=True
            ), self.assertRaises(launcher.LaunchError):
                launcher.thinking_efforts()

    def test_environment_writes_private_config_and_scrubs_provider_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            node = root / "node"
            main = root / "main.mjs"
            task = root / "README.md"
            submit = root / "bin" / "frontier-submit"
            skill = root / "skills" / "frontier-submit" / "SKILL.md"
            library = root / "lib"
            loader = library / "ld-linux-x86-64.so.2"
            submit.parent.mkdir()
            skill.parent.mkdir(parents=True)
            library.mkdir()
            for path in (node, main, task, submit, skill, loader):
                path.write_text("x", encoding="utf-8")
            node.chmod(0o755)
            loader.chmod(0o755)
            submit.chmod(0o755)
            gpu_skill = skill.parents[1] / "frontier-gpu" / "SKILL.md"
            gpu_skill.parent.mkdir()
            for path in (gpu_skill, submit.parent / "frontier-gpu", submit.parent / "frontier-gpu-connect-proxy"):
                path.write_text("x", encoding="utf-8")
                path.chmod(0o755)
            runtime = {
                "KIMI_CODE_HOME": workspace / ".kimi",
                "HOME": workspace / ".home",
                "TMPDIR": workspace / ".tmp",
            }
            environment = {
                "CODEX_API_KEY": "legacy",
                "OPENAI_API_KEY": "provider",
                "ANTHROPIC_API_KEY": "provider",
                "ANTHROPIC_AUTH_TOKEN": "run-token",
                "ANTHROPIC_BASE_URL": "http://10.0.0.7:8082",
                "ANTHROPIC_MODEL": "kimi-k2.7-code",
                "ANTHROPIC_DEFAULT_MODEL": "kimi-k2.7-code",
                "ANTHROPIC_DEFAULT_FABLE_MODEL": "kimi-k2.7-code",
                "ANTHROPIC_DEFAULT_OPUS_MODEL": "kimi-k2.7-code",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-k2.7-code",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-k2.7-code",
                "KIMI_LOOP_MAX_STEPS_PER_TURN": "50",
                "LD_LIBRARY_PATH": "/task/base/libraries",
                "FRONTIER_MODEL_PROXY_TOKEN": "run-token",
                "FRONTIER_DRIVER_API_URL": "http://10.0.0.7:8080",
                "FRONTIER_MODEL_API_BASE_URL": "http://10.0.0.7:8082/v1",
                "FRONTIER_CODEX_MODEL": "kimi-k2.7-code",
                "FRONTIER_KIMI_MAX_CONTEXT_TOKENS": "1048576",
                "KIMI_MODEL_NAME": "unrelated-provider-model",
                "KIMI_MODEL_MAX_CONTEXT_SIZE": "2048",
                "KIMI_MODEL_MAX_TOKENS": "2048",
                "KIMI_MODEL_TEMPERATURE": "0.1",
                "KIMI_MODEL_THINKING_EFFORT": "medium",
                "FRONTIER_KIMI_SUPPORT_EFFORTS": '["low","high","max"]',
                "FRONTIER_KIMI_DEFAULT_EFFORT": "high",
            }
            with mock.patch.multiple(
                launcher,
                WORKSPACE=workspace,
                NODE=node,
                KIMI_MAIN=main,
                KIMI_LIB=library,
                KIMI_LOADER=loader,
                TASK_README=task,
                SKILLS=skill.parents[1],
                SUBMIT_BIN=submit.parent,
                RUNTIME_DIRS=runtime,
            ), mock.patch.dict(launcher.os.environ, environment, clear=True):
                child, model = launcher.prepare_environment()
            config = runtime["KIMI_CODE_HOME"] / "config.toml"
            self.assertEqual(model, "kimi-k2.7-code")
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            config_text = config.read_text()
            self.assertIn("max_context_size = 1048576", config_text)
            self.assertIn('model = "kimi-k2.7-code"', config_text)
            self.assertEqual(tomllib.loads(config_text)["models"]["frontier"]["default_effort"], "high")
            self.assertFalse(any(name.startswith("KIMI_MODEL_") for name in child))
            self.assertNotIn("OPENAI_API_KEY", child)
            self.assertNotIn("CODEX_API_KEY", child)
            for name in environment:
                if name.startswith("ANTHROPIC_"):
                    self.assertNotIn(name, child)
            self.assertNotIn("KIMI_LOOP_MAX_STEPS_PER_TURN", child)
            self.assertNotIn("LD_LIBRARY_PATH", child)
            self.assertTrue(child["PATH"].startswith(str(submit.parent)))

    def test_submission_skill_allows_long_gpu_train_and_terminal_final(self) -> None:
        skill = (MODULE_PATH.parent / "submit/SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("frontier-submit train /workspace/train-submission.tar", skill)
        self.assertIn("frontier-submit status <verification_id>", skill)
        self.assertIn("returns as soon as the platform accepts", skill)
        self.assertNotIn("--no-wait", skill)
        self.assertIn("Do not call TodoList", skill)
        self.assertNotIn("300-second", skill)
        self.assertNotIn("run_in_background", skill)

    def test_train_policy_matches_the_shared_codex_and_muse_skill(self) -> None:
        def train_section(path: Path) -> str:
            text = path.read_text(encoding="utf-8")
            start = text.index("Create an uncompressed tar at an absolute path")
            return text[start:text.index("Save a usable task-defined final tar early")]

        shared = train_section(MODULE_PATH.parents[1] / "codex/skills/frontier-submit/SKILL.md")
        kimi = train_section(MODULE_PATH.parent / "submit/SKILL.md")
        self.assertIn("you may keep working while it\nruns", shared)
        # Kimi adds Bash-specific guidance after the shared policy, never a different policy.
        self.assertTrue(kimi.startswith(shared.rstrip("\n")), "Kimi train policy diverged from Codex/Muse")

        dockerfile = (MODULE_PATH.parent / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("MAX_TIMEOUT_S$1 = 300", dockerfile)
        self.assertIn("MAX_TIMEOUT_S$1 = 3600", dockerfile)
        self.assertIn("grep -Fc", dockerfile)

    def test_only_private_accepted_final_receipt_recovers_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "final.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "frontier.candidate-submit-result/v1",
                        "operation": "final",
                        "state": "accepted",
                        "payload_id": "payload_01M1ES59XBTTB6CJFBTYDREH7C",
                        "verification_id": "verification_01M1ES59Z00000000000000000",
                    }
                ),
                encoding="utf-8",
            )
            receipt.chmod(0o600)
            with mock.patch.object(launcher, "FINAL_RECEIPT", receipt):
                self.assertTrue(launcher._accepted_final_receipt())
                receipt.chmod(0o644)
                self.assertFalse(launcher._accepted_final_receipt())

    def test_main_normalizes_post_final_nonzero_exit_only(self) -> None:
        child = {"PATH": "/usr/bin"}
        with mock.patch.object(
            launcher, "prepare_environment", return_value=(child, "kimi-k2.7-code")
        ), mock.patch.object(launcher.os, "chdir"), mock.patch.object(
            launcher.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=7),
        ), mock.patch.object(launcher, "_accepted_final_receipt", return_value=True):
            self.assertEqual(launcher.main(), 0)
        with mock.patch.object(
            launcher, "prepare_environment", return_value=(child, "kimi-k2.7-code")
        ), mock.patch.object(launcher.os, "chdir"), mock.patch.object(
            launcher.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=7),
        ), mock.patch.object(launcher, "_accepted_final_receipt", return_value=False):
            self.assertEqual(launcher.main(), 7)


if __name__ == "__main__":
    unittest.main()
