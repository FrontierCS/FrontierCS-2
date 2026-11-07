"""Command test sandboxes: task.toml [test_sandboxes.<name>] access = "command" and the shared
GPU adapter's run_sandboxes() (tasks/shared/verifier-gpu/fcs2_verifier_gpu.py).

The adapter cases reuse test_shared_gpu_adapter's fake Driver and fake SSH gateway (the remote
commands run locally with /workspace mapped to a per-runner directory), so the real upload,
step, timeout and collect commands run.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from test_shared_gpu_adapter import TEMPLATE, AdapterCase, task_manifest

SANDBOX = {"image": "agent_gpu", "workdir": "/workspace/sandbox",
           "resources": {"gpus": 1, "gpu_type": "A100-80GB"}, "max_concurrent": 2,
           "lease_seconds": {"train": 900, "eval": 900}}


class RunSandboxes(AdapterCase):
    def setUp(self):
        super().setUp()
        self.parameters.write_text(json.dumps({"entrypoint": "payload/solution.patch", "gpu": {
            "placement": "control", "max_gpu_instances": 2, "sandboxes": {"candidate": SANDBOX}}}))
        self.files = self.scratch / "files"
        (self.files / "work").mkdir(parents=True)
        (self.files / "work" / "answer.txt").write_text("42")
        self.into = self.scratch / "into"
        self.into.mkdir()

    def run_one(self, *steps, outputs=("out",), **extra):
        run = self.gpu.SandboxRun("candidate", self.files, list(steps), outputs, **extra)
        return self.gpu.run_sandboxes([run], into=self.into)[0]

    def step(self, script: str, seconds: int = 60, env=None):
        return self.gpu.SandboxStep(["/bin/sh", "-c", script], seconds, env)

    def test_only_the_files_go_up_steps_run_in_the_workdir_and_outputs_come_back(self):
        result = self.run_one(
            self.step('mkdir -p out/private && chmod 700 out/private && '
                      'cat work/answer.txt > out/answer && echo "$GREETING" > out/private/env && '
                      'chmod 600 out/private/env && echo built', env={"GREETING": "hi"}),
            self.step("echo second >> out/answer"), outputs=("out", "missing"))
        self.assertEqual([(r.exit_code, r.timed_out) for r in result.steps], [(0, False), (0, False)])
        self.assertEqual((result.outputs / "out/answer").read_text(), "42second\n")
        self.assertEqual((result.outputs / "out/private/env").read_text(), "hi\n")
        self.assertEqual((result.outputs / "out/private").stat().st_mode & 0o777, 0o700)
        self.assertEqual((result.outputs / "out/private/env").stat().st_mode & 0o777, 0o600)
        self.assertFalse((result.outputs / "work").exists())
        self.assertEqual(result.logs[0], b"built\n")
        self.assertFalse(result.outputs_exceeded)
        create = self.driver.creates[0]
        self.assertEqual((create["image"], create["gpu_type"], create["gpu_count"],
                          create["max_duration_seconds"]), ("agent_gpu", "A100-80GB", 1, 900))
        self.assertEqual(len(self.driver.deleted), 1)
        workspace = next((self.scratch / "runners").iterdir()) / "workspace"
        self.assertEqual(sorted(p.name for p in (workspace / "sandbox").iterdir()), ["out", "work"])

    def test_a_timeout_is_a_scored_fact_and_ends_the_run_but_outputs_still_come_back(self):
        result = self.run_one(self.step("mkdir out; echo partial > out/a; sleep 30", seconds=2),
                              self.step("echo never > out/b"))
        self.assertTrue(result.steps[0].timed_out)
        self.assertIsNone(result.steps[0].exit_code)
        self.assertIsNone(result.steps[1])
        self.assertEqual((result.outputs / "out/a").read_text(), "partial\n")
        self.assertFalse((result.outputs / "out/b").exists())
        self.assertEqual(len(self.driver.deleted), 1)

    def test_a_failing_step_is_reported_not_raised(self):
        result = self.run_one(self.step("echo oops; exit 3"), self.step("true"))
        self.assertEqual((result.steps[0].exit_code, result.steps[0].timed_out), (3, False))
        self.assertIsNone(result.steps[1])
        self.assertEqual(result.logs, [b"oops\n", None])

    def test_outputs_over_the_cap_are_not_returned(self):
        result = self.run_one(self.step("mkdir out && head -c 200000 /dev/zero > out/big"),
                              max_output_bytes=1000)
        self.assertTrue(result.outputs_exceeded)
        self.assertIsNone(result.outputs)

    def test_only_regular_files_and_directories_come_back(self):
        result = self.run_one(self.step("mkdir out && ln -s /etc/passwd out/link && echo x > out/f"))
        self.assertEqual(sorted(p.name for p in (result.outputs / "out").iterdir()), ["f"])

    def test_unreadable_outputs_are_reported_not_returned(self):
        result = self.run_one(self.step(
            "mkdir -p out/closed && echo x > out/ok && echo y > out/secret && chmod 000 out/secret "
            "&& echo z > out/closed/f && chmod 000 out/closed"))
        self.assertEqual(result.unreadable_outputs, ["out/closed", "out/secret"])
        self.assertEqual((result.outputs / "out/ok").read_text(), "x\n")
        self.assertFalse((result.outputs / "out/secret").exists())

    def test_two_runs_get_two_runners_and_their_own_outputs(self):
        runs = [self.gpu.SandboxRun("candidate", self.files,
                                    [self.step(f"mkdir out && echo {i} > out/i")], ["out"])
                for i in range(2)]
        results = self.gpu.run_sandboxes(runs, into=self.into)
        self.assertEqual([(r.outputs / "out/i").read_text() for r in results], ["0\n", "1\n"])
        self.assertEqual(len({create["key"] for create in self.driver.creates}), 2)
        self.assertEqual(len(self.driver.deleted), 2)

    def test_a_lost_step_session_is_a_fact_and_terminates_the_runner(self):
        # The step kills the wrapper shell (the parent of `timeout`): no status line.
        result = self.run_one(self.step("kill -9 $(ps -o ppid= -p $PPID); sleep 5"),
                              self.step("true"))
        self.assertTrue(result.steps[0].lost_session)
        self.assertEqual((result.steps[0].exit_code, result.steps[0].timed_out), (None, False))
        self.assertIsNone(result.steps[1])
        self.assertIsNone(result.outputs)
        self.assertEqual(len(self.driver.deleted), 1)

    def test_a_step_deadline_is_the_end_of_its_limit_on_the_runner_clock(self):
        result = self.run_one(self.step(
            'mkdir out && python3 -c "import os, time; '
            'print(float(os.environ[\'FRONTIER_VERIFIER_LOCAL_DEADLINE\']) - time.monotonic())" '
            '> out/left', seconds=100))
        self.assertTrue(90 < float((result.outputs / "out/left").read_text()) <= 100)

    def test_the_last_step_may_take_the_rest_of_the_grant(self):
        self.driver.granted = 500  # of the 900 s requested
        result = self.run_one(self.step("true"), self.step(
            'mkdir out && python3 -c "import os, time; '
            'print(float(os.environ[\'FRONTIER_VERIFIER_LOCAL_DEADLINE\']) - time.monotonic())" '
            '> out/left', seconds=None))
        self.assertEqual([r.exit_code for r in result.steps], [0, 0])
        # 500 s granted, less the first step's time, the step grace (65 s) and the collect reserve.
        left = float((result.outputs / "out/left").read_text())
        self.assertTrue(500 - 65 - 120 - 30 < left <= 500 - 65 - 120, left)

    def test_the_rest_of_the_grant_is_a_timeout_like_any_limit(self):
        self.driver.granted = 120 + 65 + 3  # the step's limit is then 3 s
        result = self.run_one(self.step("mkdir out; sleep 30", seconds=None))
        self.assertTrue(result.steps[0].timed_out)

    def test_a_trusted_sandbox_judge_may_upload_its_own_bundle(self):
        (self.files / "judge").mkdir()
        (self.files / "judge" / "probe.py").write_text("print('judge here')\n")
        result = self.run_one(self.step("python3 judge/probe.py"), outputs=())
        self.assertEqual(result.logs[0], b"judge here\n")

    def test_malformed_requests_lease_nothing(self):
        Run, Step = self.gpu.SandboxRun, self.gpu.SandboxStep
        bad = [
            [Run("nope", self.files, [self.step("true")])],
            [Run("candidate", self.files, [Step(["sh", "-c", "true"], 5)])],
            [Run("candidate", self.files, [self.step("true", seconds=10_000)])],
            [Run("candidate", self.files, [self.step("true")], ["../escape"])],
            [Run("candidate", self.files, [self.step("true", env={"A B": "x"})])],
            [Run("candidate", self.files, [self.step("true")])] * 3,  # max_concurrent 2
            [Run("candidate", self.files, [])],
            [Run("candidate", self.files, [self.step("true", seconds=None), self.step("true")])],
            [Run("candidate", self.files, [self.step("true", seconds=0)])],
        ]
        for runs in bad:
            with self.subTest(runs=runs), self.assertRaises(self.gpu.VerifierGpuError):
                self.gpu.run_sandboxes(runs, into=self.into)
        self.assertEqual(self.driver.creates, [])

    def test_other_placements_have_no_sandboxes(self):
        self.write_parameters("runner", {"train": 600, "eval": 600})
        with self.assertRaises(self.gpu.VerifierGpuError):
            self.run_one(self.step("true"))


COMMAND = '''
[gpu.agent]
allowed_types = ["A100-80GB"]
max_gpu_seconds = 100
max_instance_seconds = 100
max_instances = 1

[gpu.verifier]
max_gpu_instances = 2
max_gpu_seconds = 100

[test_sandboxes.candidate]
access = "command"
dockerfile = "docker/agent-gpu.Dockerfile"
workdir = "/workspace/sandbox"
resources = { gpus = 1, gpu_type = "A100-80GB" }
max_concurrent = 2
lease_seconds = { train = 600, eval = 600 }
'''


class Manifest(unittest.TestCase):
    def task(self, extra: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix="command-sandbox-")) / "Theory" / "template_task"
        self.addCleanup(shutil.rmtree, root.parents[1], True)
        shutil.copytree(TEMPLATE, root)
        (root / "task.toml").write_text((root / "task.toml").read_text() + "\n" + extra)
        return root

    def test_a_command_sandbox_puts_the_judge_on_the_control_image(self):
        task, origins = task_manifest.resolve(self.task(COMMAND))
        self.assertEqual(task_manifest.verifier_parameters(task)["gpu"], {
            "placement": "control", "max_gpu_instances": 2,
            "sandboxes": {"candidate": {
                "image": "agent_gpu", "workdir": "/workspace/sandbox",
                "resources": {"gpus": 1, "gpu_type": "A100-80GB"}, "max_concurrent": 2,
                "lease_seconds": {"train": 600, "eval": 600}}}})
        self.assertEqual(origins[("gpu", "verifier", "placement")],
                         'derived: [test_sandboxes] access = "command"')
        policy = task_manifest.policy_from_task(task)
        self.assertNotIn("sidecars", policy)
        self.assertEqual(policy["gpu"]["verifier"], {"max_gpu_instances": 2, "max_gpu_seconds": 100})

    def test_refusals(self):
        cases = {
            "an image of the judge for an untrusted sandbox":
                COMMAND.replace("agent-gpu.Dockerfile", "verifier-gpu.Dockerfile"),
            "an unpublished recipe": COMMAND.replace("agent-gpu.Dockerfile", "test-sandbox.Dockerfile"),
            "more GPUs than the verifier quota": COMMAND.replace("max_concurrent = 2", "max_concurrent = 3"),
            "a lease above the eval timeout": COMMAND.replace("train = 600", "train = 99999999"),
            "a placement (derived, never written)":
                COMMAND.replace("max_gpu_seconds = 100\n\n[test", 'max_gpu_seconds = 100\nplacement = "runner"\n\n[test'),
            "a [gpu.verifier] lease beside a command sandbox":
                COMMAND.replace("max_gpu_seconds = 100\n\n[test",
                                "max_gpu_seconds = 100\nlease_seconds = { train = 60, eval = 60 }\n\n[test"),
            "service keys": COMMAND.replace('access = "command"', 'access = "command"\nargs = ["x"]'),
            "a workdir outside /workspace": COMMAND.replace("/workspace/sandbox", "/tmp/sandbox"),
            "no agent GPU image": "[gpu.verifier]" + COMMAND.split("[gpu.verifier]")[1],
            "an unknown access": COMMAND.replace('"command"', '"telepathy"'),
            "CPU and memory on a GPU runner": COMMAND.replace(
                'gpu_type = "A100-80GB" }', 'gpu_type = "A100-80GB", cpu_millis = 8000 }'),
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(task_manifest.ManifestError):
                task_manifest.read_task(self.task(text))

    def test_a_cpu_machine_is_declared_but_not_provisioned_yet(self):
        text = COMMAND.replace('resources = { gpus = 1, gpu_type = "A100-80GB" }',
                               "resources = { cpu_millis = 16000, memory_bytes = 68719476736 }")
        with self.assertRaisesRegex(task_manifest.ManifestError, "CPU sandbox provisioning"):
            task_manifest.read_task(self.task(text))

    def test_a_trusted_sandbox_may_run_on_the_verifier_gpu_image(self):
        text = COMMAND.replace('access = "command"', 'access = "command"\ntrust = "trusted"').replace(
            "agent-gpu.Dockerfile", "verifier-gpu.Dockerfile")
        task = task_manifest.read_task(self.task(text))
        self.assertEqual(task_manifest.verifier_parameters(task)["gpu"]["sandboxes"]["candidate"]["image"],
                         "verifier_gpu")


if __name__ == "__main__":
    unittest.main()
