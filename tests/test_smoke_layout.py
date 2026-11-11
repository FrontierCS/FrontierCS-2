"""The post-build Verifier smoke runs in the production layout on the production Verifier nodes
(scripts/build_task_images.py run_in_pod): a Pod from verifier_layout.pod_manifest with the
fixed NODE_SELECTOR in SMOKE_NAMESPACE, the staged inputs copied in and the outputs copied back."""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_task_images  # noqa: E402
import verifier_layout  # noqa: E402

IMAGE = "docker.io/example/verifier@sha256:" + "a" * 64
VERIFIER = {
    "command": ["/opt/frontier/verifier"],
    "cpu_millis": 4000,
    "memory_bytes": 8 << 30,
    "max_evidence_bytes": 1 << 20,
}


class RunInPodTests(unittest.TestCase):
    def test_the_command_runs_in_the_production_pod_and_files_are_copied(self):
        calls = []

        def kubectl(command, **kwargs):
            calls.append((command, kwargs))
            if command[3] == "cp" and command[4].endswith("/output/result.json"):
                Path(command[5]).write_text('{"status": "passed"}')
            return subprocess.CompletedProcess(command, 0, "out", "")

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(build_task_images.subprocess, "run", side_effect=kubectl),
        ):
            work = Path(directory) / "frontier"
            verifier_layout.stage_work(work, b"payload", b"dataset")
            (work / "input/fixture.json").write_text("{}")
            completed = build_task_images.run_in_pod(
                IMAGE, VERIFIER, "train", work, None, timeout=60, label="task-train"
            )
            self.assertEqual(completed.returncode, 0)
            self.assertEqual(
                json.loads((work / "output/result.json").read_text()),
                {"status": "passed"},
            )
        verbs = [command[3] for command, _ in calls]
        self.assertEqual(verbs[:3], ["apply", "wait", "exec"])
        self.assertEqual(verbs[-1], "delete")
        manifest = json.loads(calls[0][1]["input"])
        name = manifest["metadata"]["name"]
        self.assertEqual(
            manifest,
            verifier_layout.pod_manifest(
                name,
                IMAGE,
                VERIFIER,
                "train",
                node_selector=verifier_layout.NODE_SELECTOR,
                labels={"app": "frontier-verifier-smoke"},
            ),
        )
        self.assertEqual(
            manifest["spec"]["nodeSelector"],
            {"eks.amazonaws.com/nodegroup": "frontiercs2-preview-eval16"},
        )
        copied_in = sorted(
            command[5].rsplit("/", 1)[1]
            for command, _ in calls
            if command[3] == "cp" and command[5].startswith(name + ":")
        )
        self.assertEqual(copied_in, ["dataset", "fixture.json", "payload.tar"])
        executed = [command for command, _ in calls if command[3] == "exec"][-1]
        self.assertEqual(executed[4:], [name, "--", "/opt/frontier/verifier"])
        self.assertTrue(
            all(
                command[:3] == ["kubectl", "-n", verifier_layout.SMOKE_NAMESPACE]
                for command, _ in calls
            )
        )

    def test_a_pod_that_cannot_start_fails_and_is_deleted(self):
        calls = []

        def kubectl(command, **kwargs):
            calls.append(command)
            failed = command[3] == "wait"
            return subprocess.CompletedProcess(command, int(failed), "", "timed out")

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(build_task_images.subprocess, "run", side_effect=kubectl),
            self.assertRaisesRegex(build_task_images.BuildFailure, "did not start"),
        ):
            work = Path(directory) / "frontier"
            verifier_layout.stage_work(work, b"", b"")
            build_task_images.run_in_pod(
                IMAGE, VERIFIER, "train", work, None, timeout=60, label="x"
            )
        self.assertEqual(calls[-1][3], "delete")


if __name__ == "__main__":
    unittest.main()
