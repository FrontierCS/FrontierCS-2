#!/usr/bin/env python3
"""Byte-for-byte deployment behaviour fixtures; use --update only for reviewed changes."""

from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import argparse
import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/fixtures/golden"
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "scripts")]
from preview import config as site_config
from preview import publication
from render_task_definition import json_text
from test_frontier_deploy import Fixture
from test_frontier_deploy import frontier_deploy as deploy
from test_task_seeding import mirrored_images, resolved_catalog

GOLDEN_TASK_NAMES = (
    "batch_local_counting",
    "bounded_evm_path_prediction",
    "evm_state_checkpoints",
    "graph_mining_method_selection",
    "join_tree_rewriting",
    "sparse_video_attention_policy",
)


def capture() -> dict[str, bytes]:
    result = {}
    raw = deploy.load_json(ROOT / "config/site.example.json")
    for label, modal_enabled in [("modal", True), ("cpu", False)]:
        config = copy.deepcopy(raw)
        config["features"]["modal_gpu"] = modal_enabled
        if not modal_enabled:
            config.pop("modal")
        raw_config = copy.deepcopy(config)
        config = deploy.validate_config(config)
        outputs, images = Fixture.outputs(config), Fixture.images(config)
        result[f"{label}/bootstrap.json"] = deploy.render_bootstrap_manifest(
            config, outputs
        )
        for simulator in (False, True):
            result[f"{label}/runtime-simulator-{str(simulator).lower()}.json"] = (
                deploy.render_runtime_manifest(
                    config,
                    outputs,
                    images,
                    ["172.20.0.1/32"],
                    simulator_bootstrap=simulator,
                )
            )
        result[f"{label}/litellm-namespace.json"] = deploy.render_litellm_namespace(
            config
        )
        result[f"{label}/litellm.json"] = deploy.render_litellm_manifest(
            config, outputs, images
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "site.json"
            config_path.write_text(json_text(raw_config))
            # Only redirect output; the synthetic state path is rendered, never accessed.
            with (
                mock.patch.object(
                    site_config, "generated_directory", return_value=root / "generated"
                ),
                mock.patch.object(
                    site_config,
                    "state_directory",
                    return_value=Path("/frontier/golden-state/preview"),
                ),
            ):
                returned, output = deploy.render_bundle(config_path)
            result[f"{label}/bundle/config.json"] = json_text(returned).encode()
            for path in output.iterdir():
                # Selection is a separate artifact, outside the frozen bundle.
                if path.name == "task-selection.json":
                    continue
                result[f"{label}/bundle/{path.name}"] = path.read_bytes()
    definitions = {name: deploy.TASK_DEFINITIONS[name] for name in GOLDEN_TASK_NAMES}
    with mock.patch.object(
        publication, "load_task_definitions", return_value=definitions
    ):
        catalog = deploy.validate_task_catalog(
            resolved_catalog(GOLDEN_TASK_NAMES), require_resolved=True
        )
        mirrors = mirrored_images(catalog)
        for task in catalog["tasks"]:
            result[f"documents/task-{task['name']}.json"] = json_text(
                deploy.task_version_document(task, mirrors[task["name"]])
            ).encode()
        result["documents/candidate-codex.json"] = json_text(
            deploy.candidate_version_document(
                catalog["candidate"], mirrors["candidate"]["overlay_image"]
            )
        ).encode()
    result["documents/smoke.json"] = json_text(
        deploy.smoke_documents(Fixture.images(deploy.validate_config(raw)))
    ).encode()
    commands = next(
        action.choices
        for action in deploy.parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    with tempfile.TemporaryDirectory(prefix="frontier-golden-home-") as home:
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": home,
            "PYTHONPATH": os.environ["PYTHONPATH"],
            "TMPDIR": tempfile.gettempdir(),
            deployment_test_guard.STATE_PATHS_ENV: os.environ[
                deployment_test_guard.STATE_PATHS_ENV
            ],
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHON_COLORS": "0",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "COLUMNS": "80",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
        }
        for command in [None, *commands]:
            args = [] if command is None else [command]
            process = subprocess.run(
                [sys.executable, "-B", "scripts/frontier_deploy.py", *args, "--help"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                check=True,
            )
            assert not process.stderr, process.stderr
            result[f"cli/{command or 'root'}.txt"] = process.stdout
    return result


class GoldenTests(unittest.TestCase):
    def test_cli_capture_environment_is_minimal(self):
        environments = []

        def help_process(command, **kwargs):
            self.assertEqual(command[-1], "--help")
            env = kwargs["env"]
            self.assertTrue(Path(env["HOME"]).is_dir())
            self.assertNotEqual(env["HOME"], os.environ["HOME"])
            environments.append(env)
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

        with (
            mock.patch.dict(
                os.environ,
                {
                    "LANGUAGE": "fr",
                    "PYTHON_COLORS": "1",
                    "EXTRA_CAPTURE_VARIABLE": "must not leak",
                },
            ),
            mock.patch.object(subprocess, "run", side_effect=help_process),
        ):
            outputs = capture()
        self.assertEqual(
            len(environments), sum(name.startswith("cli/") for name in outputs)
        )
        self.assertGreater(len(environments), 1)
        for env in environments:
            self.assertEqual(env["PYTHON_COLORS"], "0")
            self.assertNotIn("LANGUAGE", env)
            self.assertEqual(
                set(env),
                {
                    "PATH",
                    "HOME",
                    "LC_ALL",
                    "LANG",
                    "PYTHON_COLORS",
                    "NO_COLOR",
                    "TERM",
                    "COLUMNS",
                    "PYTHONHASHSEED",
                    "PYTHONIOENCODING",
                    "PYTHONPATH",
                    "TMPDIR",
                    deployment_test_guard.STATE_PATHS_ENV,
                },
            )
            for key, value in {
                "LC_ALL": "C",
                "LANG": "C",
                "NO_COLOR": "1",
                "TERM": "dumb",
                "COLUMNS": "80",
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
            }.items():
                self.assertEqual(env[key], value)

    def test_deployment_outputs_match_reviewed_bytes(self):
        actual = capture()
        expected_paths = {
            path.relative_to(GOLDEN).as_posix(): path
            for path in GOLDEN.rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            set(actual),
            set(expected_paths),
            "Golden file membership changed; review the behaviour change before --update",
        )
        for name, body in actual.items():
            with self.subTest(fixture=name):
                self.assertEqual(
                    body,
                    expected_paths[name].read_bytes(),
                    f"{name} changed; review the behaviour change before --update",
                )


def update() -> None:
    outputs = capture()
    for name, body in outputs.items():
        path = GOLDEN / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    print(
        f"Updated {len(outputs)} golden fixtures; review every diff as a behaviour change."
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--update"]:
        update()
    else:
        unittest.main()
