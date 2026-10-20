# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip
import importlib.util
import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "preview_kimi", ROOT / "candidates/kimi-code/launcher.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class KimiLauncherTests(unittest.TestCase):
    def test_failure_diagnostic_retains_only_bounded_structured_fields(self):
        tail = (
            '2026-10-03T00:00:00.000Z ERROR startup failed  operation="run prompt"\n'
            "  APIProviderRateLimitError: provider.rate_limit: secret-message\n"
            f"    at run ({launcher.KIMI_MAIN}:12:34)\n"
            "    at secret-path.js:2:3\n"
        ).encode()
        diagnostic = launcher._diagnostic_from_tail(tail, True, 2)
        self.assertEqual(diagnostic["capture"], "terminal_record")
        self.assertEqual(diagnostic["error_code"], "provider.rate_limit")
        self.assertEqual(diagnostic["frames"], [{"line": 12, "column": 34}])
        self.assertTrue(diagnostic["tail_truncated"])
        self.assertNotIn("secret", json.dumps(diagnostic))
        untrusted = launcher._diagnostic_from_tail(
            b"APIProviderRateLimitError: provider.rate_limit", False, 2
        )
        self.assertEqual(untrusted["capture"], "no_terminal_record")

    def test_failure_diagnostic_does_not_follow_log_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            logs = root / "outside"
            home.mkdir()
            logs.mkdir()
            (home / "logs").symlink_to(logs, target_is_directory=True)
            with mock.patch.object(launcher, "RUNTIME_DIRS", {"KIMI_CODE_HOME": home}):
                self.assertEqual(
                    launcher._failure_diagnostic(2)["capture"], "read_failed"
                )

    def test_main_keeps_exit_status_and_records_failure_only_without_final(self):
        for accepted in (False, True):
            with (
                self.subTest(accepted_final=accepted),
                mock.patch.object(
                    launcher, "prepare_environment", return_value=({}, "kimi")
                ),
                mock.patch.object(launcher.os, "chdir"),
                mock.patch.object(
                    launcher.subprocess, "run", return_value=mock.Mock(returncode=7)
                ),
                mock.patch.object(
                    launcher, "_accepted_final_receipt", return_value=accepted
                ),
                mock.patch.object(launcher, "_retain_failure_diagnostic") as retain,
            ):
                self.assertEqual(launcher.main(), 0 if accepted else 7)
                if accepted:
                    retain.assert_not_called()
                else:
                    retain.assert_called_once_with(7)

    def test_native_config_uses_run_proxy_and_selected_context(self):
        config = tomllib.loads(launcher.render_config(
            "http://10.0.0.7:8082/v1", "kimi-k2.7-code", "run-token",
            max_context_tokens=1048576,
        ))
        self.assertEqual(config["providers"]["frontier"], {
            "type": "openai", "base_url": "http://10.0.0.7:8082/v1",
            "api_key": "run-token",
        })
        self.assertEqual(config["models"]["frontier"]["max_context_size"], 1048576)
        self.assertEqual(config["loop_control"]["max_steps_per_turn"], 0)

    def test_prepare_writes_private_config_and_scrubs_provider_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            runtime = {name: workspace / name for name in launcher.RUNTIME_DIRS}
            replacements = {
                "WORKSPACE": workspace,
                "KIMI_LIB": root,
                "RUNTIME_DIRS": runtime,
            }
            env = {
                "FRONTIER_DRIVER_API_URL": "http://10.0.0.7:8080",
                "FRONTIER_MODEL_API_BASE_URL": "http://10.0.0.7:8082/v1",
                "FRONTIER_CODEX_MODEL": "kimi-k2.7-code",
                "KIMI_API_KEY": "provider-key",
                "KIMI_MODEL_NAME": "ambient",
                "CODEX_API_KEY": "ambient",
                "FRONTIER_MODEL_PROXY_TOKEN": "run-token",
            }
            with (
                mock.patch.multiple(launcher, **replacements),
                mock.patch.object(launcher, "_regular"),
                mock.patch.dict(os.environ, env, clear=True),
            ):
                child, alias = launcher.prepare_environment()
            config = runtime["KIMI_CODE_HOME"] / "config.toml"
            self.assertEqual(config.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                tomllib.loads(config.read_text())["providers"]["frontier"]["api_key"],
                "run-token",
            )
            for key in [
                "KIMI_API_KEY",
                "KIMI_MODEL_NAME",
                "CODEX_API_KEY",
            ]:
                self.assertNotIn(key, child)
            self.assertEqual(alias, "kimi-k2.7-code")
            self.assertTrue(child["PATH"].startswith(str(launcher.SUBMIT_BIN)))

    def test_final_receipt_recovery_rejects_public_or_incomplete_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "frontier.candidate-submit-result/v1",
                        "operation": "final",
                        "state": "accepted",
                        "payload_id": "payload_test",
                        "verification_id": "verification_test",
                    }
                )
            )
            with mock.patch.object(launcher, "FINAL_RECEIPT", path):
                path.chmod(0o644)
                self.assertFalse(launcher._accepted_final_receipt())
                path.chmod(0o600)
                self.assertTrue(launcher._accepted_final_receipt())
                path.write_text("{}")
                self.assertFalse(launcher._accepted_final_receipt())
