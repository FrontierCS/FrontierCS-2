"""Command test sandboxes: task.toml [test_sandboxes.<name>] access = "command" and the shared
adapter's run_sandboxes() (tasks/shared/verifier-gpu/fcs2_verifier_gpu.py), on GPU runners and on
CPU machines (the same protocol; only `resources` differs).

The adapter cases reuse test_shared_gpu_adapter's fake Driver and fake SSH gateway (the remote
commands run locally with /workspace mapped to a per-runner directory), so the real upload,
step, timeout and collect commands run.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from unittest import mock

from test_shared_gpu_adapter import REPO, TEMPLATE, AdapterCase, task_manifest

SANDBOX = {"image": "agent_gpu", "workdir": "/workspace/sandbox",
           "resources": {"gpus": 1, "gpu_type": "A100-80GB"}, "max_concurrent": 2,
           "lease_seconds": {"train": 900, "eval": 900}}
CPU_SANDBOX = {**SANDBOX, "image": "agent_cpu",
               "resources": {"cpu_millis": 4000, "memory_bytes": 8589934592}}


class RunSandboxes(AdapterCase):
    def setUp(self):
        super().setUp()
        self.parameters.write_text(json.dumps({
            "entrypoint": "payload/solution.patch",
            "gpu": {"placement": "control", "max_gpu_instances": 2},
            "sandboxes": {"candidate": SANDBOX}}))
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

    def test_a_create_the_driver_refused_is_reported_not_leased(self):
        # A 4xx refusal proves that no runner exists: the run failed, nothing to terminate.
        # A lost create (5xx), or an idempotency conflict / in-progress create, may hide one.
        for status, message, termination in (
                (429, "a command sandbox run failed", "not_leased"),
                (422, "a command sandbox run failed", "not_leased"),
                (500, "termination was not confirmed", "unconfirmed_no_identity"),
                (409, "termination was not confirmed", "unconfirmed_no_identity"),
                (410, "termination was not confirmed", "unconfirmed_no_identity"),
                (425, "termination was not confirmed", "unconfirmed_no_identity")):
            with self.subTest(status=status):
                self.driver.creates.clear()
                self.driver.fail_scope_suffix, self.driver.fail_status = 1, status
                into = Path(tempfile.mkdtemp(dir=self.scratch))
                with self.assertRaisesRegex(self.gpu.VerifierGpuError, message) as raised:
                    self.gpu.run_sandboxes([self.gpu.SandboxRun(
                        "candidate", self.files, [self.step("true")])], into=into)
                [run] = raised.exception.private_evidence["runs"]
                self.assertEqual((run["status"], run["termination"]), ("failed", termination))
        self.assertEqual(self.driver.deleted, [])

    # The Driver's statuses for a Verifier runner create (core driver/crates/driver-api
    # src/lib.rs, `impl From<DriverError> for ApiError` and the route's own refusals), and
    # whether each proves that no runner exists: raised before the provider is called. A status
    # the Driver adds later is in doubt until it is added here and to the adapter's allow-list.
    DRIVER_CREATE_STATUSES = {
        "bad_request (shape, headers, body)": (400, True),
        "unauthorized": (401, True),
        "verifier token scope": (403, True),
        "verification_not_found": (404, True),
        "idempotency_conflict": (409, False),       # an idempotency record exists
        "driver_cancelled": (410, False),           # mapped after the provider was called
        "invalid_dispatch / runner_not_declared": (422, True),
        "idempotency_in_progress": (425, False),    # a create under this key is running
        "quota_exhausted / gpu_type_not_allowed / cpu_machine_shape_not_declared": (429, True),
        "internal": (500, False),
        "gpu_backend_unavailable / verifier_start_failed": (502, False),
        "runner_capacity_refused / agent_not_ready": (503, False),
    }

    def test_only_refusals_before_the_provider_prove_that_nothing_was_leased(self):
        proven = {status for status, absent in self.DRIVER_CREATE_STATUSES.values() if absent}
        self.assertEqual(self.gpu.CREATE_REFUSALS_NOT_LEASED, frozenset(proven))
        for name, (status, absent) in self.DRIVER_CREATE_STATUSES.items():
            with self.subTest(name=name):
                self.assertEqual(self.gpu._not_leased(self.gpu._DriverRefusal(name, status)), absent)
        self.assertFalse(self.gpu._not_leased(self.gpu.VerifierGpuError("lost response")))

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
        # verifier.json has `sandboxes` only for a task that declares command test sandboxes
        # (task_manifest validated it); run_sandboxes on any other task fails on the missing key.
        self.write_parameters("runner", {"train": 600, "eval": 600})
        with self.assertRaisesRegex(KeyError, "sandboxes"):
            self.run_one(self.step("true"))


class RunCpuSandboxes(RunSandboxes):
    """Every RunSandboxes case again on a CPU machine: the same judge code, upload, steps and
    collect; only the create request carries the machine's exact shape instead of GPUs. A task
    without GPUs has no `gpu` table and no FRONTIER_VERIFIER_GPU_TYPE."""

    def setUp(self):
        super().setUp()
        os.environ.pop("FRONTIER_VERIFIER_GPU_TYPE")
        os.environ.pop("FRONTIER_VERIFIER_GPU_COUNT")
        self.parameters.write_text(json.dumps({"entrypoint": "payload/solution.patch",
                                               "sandboxes": {"candidate": CPU_SANDBOX}}))

    def test_only_the_files_go_up_steps_run_in_the_workdir_and_outputs_come_back(self):
        result = self.run_one(self.step("mkdir out && cat work/answer.txt > out/answer"))
        self.assertEqual((result.outputs / "out/answer").read_text(), "42")
        create = self.driver.creates[0]
        self.assertEqual({key: value for key, value in create.items() if key != "key"}, {
            "image": "agent_cpu", "cpu_millis": 4000, "memory_bytes": 8589934592,
            "max_duration_seconds": 900})
        self.assertEqual(len(self.driver.deleted), 1)

    def test_a_reply_of_another_shape_is_refused_as_infrastructure(self):
        original = self.driver.open

        def other_shape(request, timeout=None):
            response = original(request, timeout)
            if request.get_method() == "POST":
                body = json.loads(response._body)
                body["cpu_millis"] = 8000
                response._body = json.dumps(body).encode()
            return response

        with mock.patch.object(self.gpu, "_HTTP", mock.Mock(open=other_shape)):
            # Unscored, loudly: the reply is not this machine, so its identity is not trusted
            # for a delete either (the Driver reclaims it at the verification's end).
            with self.assertRaisesRegex(self.gpu.VerifierGpuError, "termination was not confirmed"):
                self.run_one(self.step("true"))

    def test_other_placements_have_no_sandboxes(self):
        self.parameters.write_text(json.dumps({"entrypoint": "payload/solution.patch"}))
        with self.assertRaisesRegex(KeyError, "sandboxes"):
            self.run_one(self.step("true"))

    def test_the_shared_verifier_keeps_the_driver_bearer_for_declared_sandboxes(self):
        verifier = self.verifier
        for sandboxes, kept in ((True, "bearer"), (False, None)):
            with mock.patch.dict(verifier._sealed, clear=True), \
                    mock.patch.dict(os.environ, {"FRONTIER_VERIFIER_TOKEN": "bearer"}), \
                    mock.patch.object(verifier, "_wipe_initial_environment", lambda names: None):
                verifier.seal_credentials(sandboxes=sandboxes)
                self.assertEqual(verifier.verifier_token(), kept)
                self.assertNotIn("FRONTIER_VERIFIER_TOKEN", os.environ)


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


CPU_COMMAND = '''
[test_sandboxes.candidate]
access = "command"
dockerfile = "docker/agent.Dockerfile"
workdir = "/workspace/sandbox"
resources = { cpu_millis = 4000, memory_bytes = 8589934592 }
max_concurrent = 2
lease_seconds = { train = 600, eval = 600 }

[test_sandboxes.timing]
access = "command"
trust = "trusted"
dockerfile = "docker/verifier.Dockerfile"
workdir = "/workspace/timing"
resources = { cpu_millis = 4000, memory_bytes = 8589934592 }
max_concurrent = 1
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
        parameters = task_manifest.verifier_parameters(task)
        self.assertEqual(parameters["gpu"], {"placement": "control", "max_gpu_instances": 2})
        self.assertEqual(parameters["sandboxes"], {"candidate": {
            "image": "agent_gpu", "workdir": "/workspace/sandbox",
            "resources": {"gpus": 1, "gpu_type": "A100-80GB"}, "max_concurrent": 2,
            "lease_seconds": {"train": 600, "eval": 600}}})
        self.assertNotIn("verifier_cpu", task)
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

    def test_a_cpu_machine_needs_only_its_resources(self):
        # A task without GPUs: the declaration alone makes its CPU machines and their quota.
        root = self.task(CPU_COMMAND)
        task, origins = task_manifest.resolve(root)
        self.assertNotIn("gpu", task)
        parameters = task_manifest.verifier_parameters(task)
        self.assertNotIn("gpu", parameters)
        self.assertEqual(parameters["sandboxes"], {
            "candidate": {"image": "agent_cpu",
                          "workdir": "/workspace/sandbox",
                          "resources": {"cpu_millis": 4000, "memory_bytes": 8589934592},
                          "max_concurrent": 2, "lease_seconds": {"train": 600, "eval": 600}},
            "timing": {"image": "verifier_cpu", "workdir": "/workspace/timing",
                       "resources": {"cpu_millis": 4000, "memory_bytes": 8589934592},
                       "max_concurrent": 1, "lease_seconds": {"train": 600, "eval": 600}}})
        # Three machines at once (2 + 1), each for a whole verification of each call.
        timeout, calls = task["quota"]["eval_timeout_seconds"], task["quota"]["max_train_calls"]
        self.assertEqual(task["verifier_cpu"], {
            "shapes": [{"cpu_millis": 4000, "memory_bytes": 8589934592}], "max_instances": 3,
            "max_seconds": calls * 3 * timeout, "final_seconds": 3 * timeout})
        self.assertEqual(task_manifest.policy_from_task(task)["verifier_cpu"], task["verifier_cpu"])
        self.assertTrue(origins[("verifier_cpu", "max_instances")].startswith("derived"))
        self.assertEqual(task_manifest.cpu_runner_roles(task),
                         ["agent-cpu-runner", "verifier-cpu-runner"])
        generated = task_manifest.generated_files(root)
        self.assertEqual(generated[".frontier-generated/runner/cpu-runner.Dockerfile"],
                         (REPO / "tasks/shared/images/cpu-runner.Dockerfile").read_bytes())
        self.assertIn(".frontier-generated/runner/task-runtime", generated)
        # A task without CPU machines generates no runner layer.
        self.assertNotIn(".frontier-generated/runner/task-runtime",
                         task_manifest.generated_files(self.task(COMMAND)))

    def test_a_cpu_machine_refusals(self):
        cases = {
            "fractional cores": (CPU_COMMAND.replace("cpu_millis = 4000", "cpu_millis = 3500"),
                                 "must be whole cores"),
            "memory that is not whole MiB": (
                CPU_COMMAND.replace("memory_bytes = 8589934592", "memory_bytes = 8589934593"), "whole MiB"),
            "more than 256 GiB": (
                CPU_COMMAND.replace("memory_bytes = 8589934592", "memory_bytes = 549755813888"), "256 GiB"),
            "GPUs and CPU together": (CPU_COMMAND.replace(
                "cpu_millis = 4000, memory_bytes", "gpus = 1, cpu_millis = 4000, memory_bytes"),
                "resources of a GPU runner are gpus and gpu_type"),
            "only cores": (CPU_COMMAND.replace(", memory_bytes = 8589934592 }", " }"),
                           "exactly cpu_millis and memory_bytes"),
            "the judge's image for the candidate": (
                CPU_COMMAND.replace('dockerfile = "docker/agent.Dockerfile"',
                                    'dockerfile = "docker/verifier.Dockerfile"'), "carries the judge"),
            "a GPU image for a CPU machine": (
                CPU_COMMAND.replace('dockerfile = "docker/agent.Dockerfile"',
                                    'dockerfile = "docker/agent-gpu.Dockerfile"'), "a CPU machine runs one of"),
            "a CPU image for a GPU runner": (COMMAND.replace("agent-gpu.Dockerfile", "agent.Dockerfile"),
                                             "a GPU machine runs one of"),
            "the quota written by hand": ("verifier_cpu = { max_instances = 3 }\n" + CPU_COMMAND, "verifier_cpu"),
        }
        for name, (text, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(task_manifest.ManifestError, message):
                task_manifest.read_task(self.task(text))

    def test_a_trusted_sandbox_may_run_on_the_verifier_gpu_image(self):
        text = COMMAND.replace('access = "command"', 'access = "command"\ntrust = "trusted"').replace(
            "agent-gpu.Dockerfile", "verifier-gpu.Dockerfile")
        task = task_manifest.read_task(self.task(text))
        self.assertEqual(task_manifest.verifier_parameters(task)["sandboxes"]["candidate"]["image"],
                         "verifier_gpu")


if __name__ == "__main__":
    unittest.main()



class VerifierChannelSize(unittest.TestCase):
    """The Verifier's SSH channel size (FRONTIER_MAX_SSH_BYTES_PER_DIRECTION, by which the
    adapter sizes upload chunks and collects to 7/8) is the Verifier image's own: a task's
    Dockerfile ENV, else the shared Verifier base's ENV (8 MiB); a legacy task's vendored adapter
    keeps its own default. The Driver never sets it. The task's
    network.audit.max_ssh_bytes_per_direction is a retired recording capacity; many tasks declare
    512 MiB, which the adapter would refuse. Every catalog task's effective size must be one its
    adapter accepts."""

    ENV = re.compile(r"FRONTIER_MAX_SSH_BYTES_PER_DIRECTION=(\d+)")

    def test_every_catalog_tasks_effective_channel_size_is_accepted_by_its_adapter(self):
        from test_shared_gpu_adapter import load_adapter
        import importlib.util
        import render_task_definition as renderer
        shared = load_adapter()
        catalog = json.loads((REPO / "config/task-catalog.json").read_text())["tasks"]
        checked = 0
        for task in catalog:
            root = renderer.problem_directory(renderer.TASK_ROOT / "problems", task["name"])
            vendored = root / "verifier/fcs2_verifier_gpu.py"
            if vendored.is_file():
                spec = importlib.util.spec_from_file_location(f"vendored_{task['name']}", vendored)
                adapter = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(adapter)
                values = {match for dockerfile in (root / "verifier").glob("*Dockerfile*")
                          for match in self.ENV.findall(dockerfile.read_text(encoding="utf-8"))}
            elif task_manifest.is_native(root):
                parameters = task_manifest.verifier_parameters(task_manifest.read_task(root))
                if "gpu" not in parameters and "sandboxes" not in parameters:
                    continue
                adapter = shared
                # Each image that carries the shared adapter inherits the Verifier base's ENV (its
                # final stage is FROM ${VERIFIER_BASE}) or sets the size itself.
                for dockerfile in (root / "docker").glob("*.Dockerfile"):
                    text = dockerfile.read_text(encoding="utf-8")
                    if "fcs2_verifier_gpu.py" not in text:
                        continue
                    stages = re.findall(r"^FROM\s+(\S+)", text, re.MULTILINE)
                    with self.subTest(task=task["name"], dockerfile=dockerfile.name):
                        self.assertTrue(stages[-1] == "${VERIFIER_BASE}" or self.ENV.search(text),
                                        stages)
                values = {task_manifest.verifier_channel_bytes(root)}
            else:
                continue
            self.assertLessEqual(len(values), 1, task["name"])
            environment = {"FRONTIER_MAX_SSH_BYTES_PER_DIRECTION": values.pop()} if values else {}
            with self.subTest(task=task["name"]):
                self.assertGreater(adapter._upload_chunk_bytes(environment), 0)
            checked += 1
        self.assertGreaterEqual(checked, 15)
        print(f"{checked} catalog tasks use a GPU adapter; each one accepts its channel size", file=sys.stderr)
        # What the Driver must never pass: the retired cap many tasks declare.
        with self.assertRaises(shared.VerifierGpuError):
            shared._upload_chunk_bytes({"FRONTIER_MAX_SSH_BYTES_PER_DIRECTION": str(512 * 1024 * 1024)})
        # The shared adapter has no built-in size: an image without the ENV fails loudly.
        with self.assertRaisesRegex(shared.VerifierGpuError, "FRONTIER_MAX_SSH_BYTES_PER_DIRECTION"):
            shared._upload_chunk_bytes({})

    def test_effective_sizes_are_unchanged_by_the_explicit_base_value(self):
        # The base ENV is the adapter's former built-in 8 MiB.
        import render_task_definition as renderer
        self.assertEqual(task_manifest.verifier_channel_bytes(None), str(8 * 1024 * 1024))
        polygon = renderer.problem_directory(renderer.TASK_ROOT / "problems", "polygon")
        self.assertEqual(task_manifest.verifier_channel_bytes(polygon), str(8 * 1024 * 1024))
