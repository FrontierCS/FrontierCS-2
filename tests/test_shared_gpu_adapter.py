"""The shared GPU adapter (tasks/shared/verifier-gpu/fcs2_verifier_gpu.py, the gpu-tools image).

A fake Driver (the adapter's HTTP opener) and a fake SSH gateway (a local process that runs the
remote command with /workspace mapped to a per-runner directory) exercise the real adapter code:
the one-runner idempotency key, the per-mode lease, the judge deadline, GPU errors that a judge's
`except RuntimeError` cannot swallow, the [gpu.verifier] manifest keys and the shared verifier's
placement hook. The command test sandboxes (run_sandboxes) are tests/test_command_sandboxes.py.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import hashlib
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import task_manifest  # noqa: E402

ADAPTER = REPO / "tasks/shared/verifier-gpu/fcs2_verifier_gpu.py"
VERIFIER_DIR = REPO / "tasks/shared/verifier"
TEMPLATE = REPO / "tasks/template/Theory/template_task"
VERIFICATION = "verification_01ARZ3NDEKTSV4RRFFQ69G5FAV"
DRIVER = "http://driver.internal:8080"
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

FAKE_SSH = r'''
import os, subprocess, sys
root, command = sys.argv[1], sys.argv[-1]
os.makedirs(os.path.join(root, "workspace"), exist_ok=True)
command = command.replace("/workspace", os.path.join(root, "workspace"))
sys.exit(subprocess.call(["/bin/sh", "-c", command]))
'''


def load_verifier():
    """A fresh shared verifier module (the adapter's sealed-credential handle)."""
    for name in ("fcs2_verifier", "beat_spec_feedback", "edit_scope"):
        sys.modules.pop(name, None)
    if str(VERIFIER_DIR) not in sys.path:
        sys.path.insert(0, str(VERIFIER_DIR))
    import fcs2_verifier
    return fcs2_verifier


def load_adapter():
    sys.modules.pop("fcs2_verifier_gpu", None)
    spec = importlib.util.spec_from_file_location("fcs2_verifier_gpu", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fcs2_verifier_gpu"] = module
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, status: int, url: str, body: dict):
        self.status, self._url, self._body = status, url, json.dumps(body).encode()

    def geturl(self):
        return self._url

    def read(self, limit=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    """The Driver's Verifier GPU API: idempotent create, delete, per-request failure hooks."""

    def __init__(self, granted: int | None = None):
        self.lock = threading.Lock()
        self.creates: list[dict] = []
        self.runners: dict[str, dict] = {}  # Idempotency-Key -> runner
        self.deleted: list[str] = []
        self.granted = granted
        self.fail_scope_suffix = None

    def open(self, request, timeout=None):
        url = request.full_url
        base = f"{DRIVER}/internal/v1/verifications/{VERIFICATION}/gpu-runners"
        with self.lock:
            if request.get_method() == "POST":
                body = json.loads(request.data)
                key = request.get_header("Idempotency-key")
                self.creates.append({"key": key, **body})
                if self.fail_scope_suffix is not None and len(self.creates) == self.fail_scope_suffix:
                    raise urllib.error.HTTPError(url, 500, "boom", {}, io.BytesIO(b"{}"))
                if key not in self.runners:
                    number = len(self.runners)
                    runner_id = "gpurunner_0" + "".join(
                        CROCKFORD[(number >> (5 * i)) & 31] for i in range(25))
                    shape = ("cpu_millis", "memory_bytes") if "cpu_millis" in body \
                        else ("gpu_type", "gpu_count")
                    self.runners[key] = {
                        "runner_id": runner_id, "state": "ready",
                        **{name: body[name] for name in shape},
                        "image": body.get("image", "verifier_gpu"),
                        "granted_seconds": self.granted or body["max_duration_seconds"],
                        "proxy_endpoint": f"{base}/{runner_id}/connect",
                        "ssh_gateway": {"schema_version": "frontier.ssh-gateway/v1",
                                        "username": "frontier",
                                        "server_host_public_key": "ssh-ed25519 AAAA",
                                        "server_host_key_sha256": "SHA256:x",
                                        "authentication": "connect_bearer", "max_channels": 1}}
                return Response(202, url, self.runners[key])
            runner_id = url.rsplit("/", 1)[1]
            self.deleted.append(runner_id)
            return Response(200, url, {"runner_id": runner_id, "state": "terminated"})


class AdapterCase(unittest.TestCase):
    def setUp(self):
        self.gpu = load_adapter()
        self.scratch = Path(tempfile.mkdtemp(prefix="shared-gpu-adapter-"))
        self.parameters = self.scratch / "verifier.json"
        self.write_parameters("runner", {"train": 600, "eval": 900})
        fake_ssh = self.scratch / "fake_ssh.py"
        fake_ssh.write_text(FAKE_SSH)
        self.driver = FakeDriver()
        environment = {
            "FRONTIER_DRIVER_INTERNAL_URL": DRIVER, "FRONTIER_VERIFICATION_ID": VERIFICATION,
            "FRONTIER_VERIFIER_GPU_TYPE": "H100",
            "FRONTIER_VERIFIER_GPU_COUNT": "1", "FRONTIER_VERIFICATION_MODE": "train",
            "FRONTIER_MAX_EVIDENCE_BYTES": "4194304"}
        for name in ("FRONTIER_VERIFIER_GPU_SECONDS", "FRONTIER_VERIFIER_DEADLINE_UNIX_MS",
                     "FRONTIER_MAX_SSH_BYTES_PER_DIRECTION"):
            environment[name] = ""
        self.verifier = load_verifier()
        self.real_ssh_argv = self.gpu._ssh_argv
        patches = [
            mock.patch.dict(os.environ, environment),
            # As sealed by fcs2_verifier.main(); the test process itself stays dumpable.
            mock.patch.dict(self.verifier._sealed, {"FRONTIER_VERIFIER_TOKEN": "token"}),
            mock.patch.object(self.verifier, "_make_undumpable", lambda: None),
            mock.patch.object(self.gpu, "PARAMETERS", self.parameters),
            mock.patch.object(self.gpu, "_HTTP", self.driver),
            mock.patch.object(self.gpu, "_ssh_argv", lambda runner, scratch: [
                sys.executable, str(fake_ssh), str(self.scratch / "runners" / runner["runner_id"])]),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        for name in ("FRONTIER_VERIFIER_GPU_SECONDS", "FRONTIER_VERIFIER_DEADLINE_UNIX_MS",
                     "FRONTIER_MAX_SSH_BYTES_PER_DIRECTION"):
            os.environ.pop(name)

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def write_parameters(self, placement: str, lease: dict, instances: int = 8):
        self.parameters.write_text(json.dumps({"entrypoint": "payload/solution.patch", "gpu": {
            "placement": placement, "lease_seconds": lease, "max_gpu_instances": instances}}))


class Errors(AdapterCase):
    def test_the_one_runner_placement_keeps_its_idempotency_key(self):
        self.gpu._create_runner(DRIVER, VERIFICATION, "token", "H100", 600, 4)
        self.assertEqual(self.driver.creates[0]["key"], hashlib.sha256(
            f"frontiersmith-verifier-gpu-v1\0{VERIFICATION}".encode()).hexdigest())

    def test_gpu_errors_are_not_runtime_errors(self):
        self.assertFalse(issubclass(self.gpu.VerifierGpuError, RuntimeError))


class LeaseAndDeadline(AdapterCase):
    def test_the_runner_command_exports_the_deadline_before_verify(self):
        command = self.gpu._remote_verifier_command("eval", 4, remaining_seconds=50.0)
        self.assertLess(command.index("FRONTIER_VERIFIER_LOCAL_DEADLINE"), command.index("verify.py"))
        runtime = self.gpu._remote_runtime(50.0)
        shown = subprocess.run(
            ["/bin/sh", "-c", runtime + 'printf %s "$FRONTIER_VERIFIER_LOCAL_DEADLINE"'],
            capture_output=True, text=True, check=True).stdout
        self.assertAlmostEqual(float(shown) - time.monotonic(), 50.0, delta=10)
        with self.assertRaises(self.gpu.VerifierGpuError):
            self.gpu._remote_runtime(0)

    def test_the_lease_follows_the_verification_mode(self):
        self.write_parameters("runner", {"train": 7200, "eval": 10800})
        requested = []

        def create(*args, **kwargs):
            requested.append(args[4])
            raise self.gpu.VerifierGpuError("stop here")
        payload, dataset = self.scratch / "payload.tar", self.scratch / "dataset"
        payload.write_bytes(b"x")
        dataset.write_bytes(b"y")
        with mock.patch.object(self.gpu, "_create_runner", create):
            for mode in ("train", "eval"):
                with self.assertRaises(self.gpu.VerifierGpuError):
                    self.gpu.run(payload=payload, dataset=dataset, mode=mode,
                                 result_path=self.scratch / "r", evidence_path=self.scratch / "e")
        self.assertEqual(requested, [7200, 10800])

    def test_the_retired_lease_variable_is_refused(self):
        os.environ["FRONTIER_VERIFIER_GPU_SECONDS"] = "12600"
        with self.assertRaises(self.gpu.VerifierGpuError):
            self.gpu.lease_seconds("eval")

    def test_malformed_parameters_are_refused(self):
        for gpu in ({"placement": "both", "lease_seconds": {"train": 1, "eval": 1}, "max_gpu_instances": 1},
                    {"placement": "runner", "lease_seconds": {"train": 1}, "max_gpu_instances": 1},
                    {"placement": "runner", "lease_seconds": {"train": 0, "eval": 1}, "max_gpu_instances": 1}):
            self.parameters.write_text(json.dumps({"gpu": gpu}))
            with self.subTest(gpu=gpu), self.assertRaises(self.gpu.VerifierGpuError):
                self.gpu.placement()


class SharedVerifierPlacement(AdapterCase):
    """fcs2_verifier.main: "runner" ships the verification; "control" judges on the control Pod."""

    def setUp(self):
        super().setUp()
        self.bundle = self.scratch / "task"
        shutil.copytree(task_manifest.judge_bundle(TEMPLATE), self.bundle)
        (self.bundle / "beat_spec.json").unlink(missing_ok=True)
        self.out = self.scratch / "out"
        self.out.mkdir()
        os.environ.update({"FRONTIER_RESULT_PATH": str(self.out / "result.json"),
                           "FRONTIER_EVIDENCE_PATH": str(self.out / "evidence.tar.zst"),
                           "FRONTIER_DATASET_PATH": str(TEMPLATE / "datasets/train.json"),
                           "FRONTIER_PAYLOAD_PATH": str(self.payload())})

    def payload(self) -> Path:
        body = (TEMPLATE / "judge/reference.patch").read_bytes()
        path = self.scratch / "payload.tar"
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
            folder = tarfile.TarInfo("payload")
            folder.type, folder.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(folder)
            entry = tarfile.TarInfo("payload/solution.patch")
            entry.size = len(body)
            archive.addfile(entry, io.BytesIO(body))
        return path

    def evaluator(self, body: str):
        (self.bundle / "evaluator.py").chmod(0o644)
        (self.bundle / "evaluator.py").write_text(textwrap.dedent(body))

    def result(self) -> dict:
        return json.loads((self.out / "result.json").read_text())

    def test_runner_placement_ships_the_whole_verification(self):
        self.write_parameters("runner", {"train": 600, "eval": 900})
        with mock.patch.object(self.gpu, "run") as run:
            self.assertEqual(self.verifier.main(self.bundle, "payload/solution.patch", sandboxes=False), 0)
        self.assertEqual(run.call_args.kwargs["mode"], "train")

    def control(self):
        self.parameters.write_text(json.dumps({
            "entrypoint": "payload/solution.patch",
            "gpu": {"placement": "control", "max_gpu_instances": 2},
            "sandboxes": {"shard": {
                "image": "verifier_gpu", "workdir": "/workspace/shard",
                "resources": {"gpus": 1, "gpu_type": "H100"}, "max_concurrent": 2,
                "lease_seconds": {"train": 600, "eval": 600}}}}))

    def test_control_placement_runs_evaluate_here_with_the_applied_tree(self):
        self.control()
        self.evaluator('''
            import json, os, tempfile
            from fcs2_verifier_gpu import SandboxRun, SandboxStep, run_sandboxes
            def evaluate(tree):
                into = tempfile.mkdtemp()
                step = SandboxStep(["/bin/sh", "-c", "mkdir out && ls > out/listing"], 60)
                results = run_sandboxes([SandboxRun("shard", tree, [step], ["out"])] * 2, into=into)
                rows = [{"workload": str(i), "metrics": {"ok": r.steps[0].exit_code == 0}}
                        for i, r in enumerate(results)]
                return 50.0, 50.0, "ok", {"beat_spec_workload_passes": rows, "beats_reference": 1,
                                          "reference_score": 50.0, "margin": 0.0}
        ''')
        with mock.patch.object(self.gpu, "run") as run:
            self.assertEqual(self.verifier.main(self.bundle, "payload/solution.patch", sandboxes=False), 0)
        run.assert_not_called()
        self.assertEqual(self.result()["status"], "passed", self.result())
        self.assertEqual(len(self.driver.deleted), 2)
        self.assertEqual(sorted(self.driver.deleted),
                         sorted(r["runner_id"] for r in self.driver.runners.values()))

    def test_a_sandbox_failure_is_an_unscored_gpu_failure(self):
        self.control()
        self.driver.fail_scope_suffix = 1
        self.evaluator('''
            import tempfile
            from fcs2_verifier_gpu import SandboxRun, SandboxStep, run_sandboxes
            def evaluate(tree):
                run_sandboxes([SandboxRun("shard", tree, [SandboxStep(["/bin/true"], 60)])],
                              into=tempfile.mkdtemp())
        ''')
        self.assertEqual(self.verifier.main(self.bundle, "payload/solution.patch", sandboxes=False), 0)
        result = self.result()
        self.assertEqual((result["status"], result["failure"]["code"]), ("failed", "verifier_gpu_error"))
        self.assertNotIn("raw_score", result)

    def test_a_judge_catching_runtime_errors_cannot_score_a_lost_runner(self):
        self.control()
        self.driver.fail_scope_suffix = 1
        self.evaluator('''
            import tempfile
            from fcs2_verifier_gpu import SandboxRun, SandboxStep, run_sandboxes
            def evaluate(tree):
                try:
                    run_sandboxes([SandboxRun("shard", tree, [SandboxStep(["/bin/true"], 60)])],
                                  into=tempfile.mkdtemp())
                except (RuntimeError, ValueError, OSError) as exc:  # the submission's failure
                    return 0.0, 0.0, str(exc), {}
        ''')
        self.assertEqual(self.verifier.main(self.bundle, "payload/solution.patch", sandboxes=False), 0)
        result = self.result()
        self.assertEqual((result["status"], result["failure"]["code"]), ("failed", "verifier_gpu_error"))
        self.assertNotIn("raw_score", result)

    def test_a_broken_gpu_configuration_is_a_gpu_failure(self):
        self.parameters.write_text("{}")
        self.assertEqual(self.verifier.main(self.bundle, "payload/solution.patch", sandboxes=False), 0)
        self.assertEqual(self.result()["failure"]["code"], "verifier_gpu_error")


class ConnectBroker(AdapterCase):
    """ssh's ProxyCommand child reaches the Driver CONNECT through this process; it never holds
    the bearer (environment, argv) and the Driver still sees it on the CONNECT."""

    RUNNER = {"runner_id": "gpurunner_0" + "A" * 25,
              "ssh_gateway": {"server_host_public_key": "ssh-ed25519 AAAA", "username": "frontier"}}

    def setUp(self):
        super().setUp()
        self.server = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(self.server.close)
        self.seen: list[bytes] = []
        self.refuse = False
        self.verifier._sealed["FRONTIER_VERIFIER_TOKEN"] = "sealed-bearer-7f3a"
        threading.Thread(target=self.serve, daemon=True).start()
        os.environ["FRONTIER_DRIVER_INTERNAL_URL"] = f"http://127.0.0.1:{self.server.getsockname()[1]}"

    def serve(self):
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            with connection:
                header = b""
                while not header.endswith(b"\r\n\r\n"):
                    header += connection.recv(1)
                self.seen.append(header)
                if self.refuse:
                    connection.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                    continue
                connection.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                data = b""
                while chunk := connection.recv(65536):
                    data += chunk
                connection.sendall(b"pong-" + data)

    def proxy(self, environment_extra=None):
        os.environ["FRONTIER_VERIFIER_TOKEN"] = "leaked-if-forwarded"  # a careless caller
        argv = self.real_ssh_argv(self.RUNNER, self.scratch)
        proxy = next(a for a in argv if a.startswith("ProxyCommand="))[len("ProxyCommand="):]
        self.assertNotIn("sealed-bearer-7f3a", proxy)
        command = proxy.replace("/opt/frontier/fcs2_verifier_gpu.py", str(ADAPTER))
        environment = self.gpu._ssh_environment(self.scratch)
        self.assertFalse([k for k, v in environment.items()
                          if "sealed-bearer" in v or "leaked-if-forwarded" in v or k == "FRONTIER_VERIFIER_TOKEN"])
        return subprocess.run(["/bin/sh", "-c", command], input=b"ping", capture_output=True,
                              env=environment, timeout=30)

    def test_the_proxy_child_tunnels_without_the_bearer(self):
        completed = self.proxy()
        self.assertEqual((completed.returncode, completed.stdout), (0, b"pong-ping"), completed.stderr)
        request = self.seen[0].decode()
        self.assertTrue(request.startswith(
            f"CONNECT /internal/v1/verifications/{VERIFICATION}/gpu-runners/{self.RUNNER['runner_id']}/connect "))
        self.assertIn("Authorization: Bearer sealed-bearer-7f3a\r\n", request)

    def test_a_refused_connect_keeps_its_diagnostic(self):
        self.refuse = True
        completed = self.proxy()
        self.assertEqual(completed.returncode, 1)
        self.assertIn(b"the Driver refused the Verifier GPU CONNECT", completed.stderr)

    def test_termination_closes_the_broker(self):
        self.real_ssh_argv(self.RUNNER, self.scratch)
        self.assertTrue((self.scratch / "connect.sock").exists())
        self.gpu._terminate_runner(DRIVER, VERIFICATION, "token", self.RUNNER["runner_id"])
        self.assertFalse((self.scratch / "connect.sock").exists())
        self.assertNotIn(self.RUNNER["runner_id"], self.gpu._BROKERS)

    def test_no_sealed_bearer_is_a_gpu_error(self):
        self.verifier._sealed.clear()
        with self.assertRaisesRegex(self.gpu.VerifierGpuError, "no Driver credential"):
            self.gpu._config()


class Manifest(unittest.TestCase):
    """task.toml [gpu.verifier]: adapter keys, defaults, never in the platform definition."""

    def task(self, gpu: str) -> Path:
        root = Path(tempfile.mkdtemp(prefix="gpu-manifest-")) / "Theory" / "template_task"
        self.addCleanup(shutil.rmtree, root.parents[1], True)
        shutil.copytree(TEMPLATE, root)
        (root / "task.toml").write_text((root / "task.toml").read_text() + "\n" + gpu)
        return root

    def test_defaults_are_the_runner_and_the_eval_timeout(self):
        root = self.task("[gpu.verifier]\nmax_gpu_instances = 4\nmax_gpu_seconds = 100\n")
        task, origins = task_manifest.resolve(root)
        timeout = task["quota"]["eval_timeout_seconds"]
        self.assertEqual(task_manifest.verifier_parameters(task)["gpu"], {
            "placement": "runner", "lease_seconds": {"train": timeout, "eval": timeout},
            "max_gpu_instances": 4})
        self.assertEqual(origins[("gpu", "verifier", "placement")], "default")
        self.assertEqual(task_manifest.policy_from_task(task)["gpu"],
                         {"verifier": {"max_gpu_instances": 4, "max_gpu_seconds": 100}})

    def test_per_mode_lease(self):
        root = self.task('[gpu.verifier]\nmax_gpu_instances = 8\nmax_gpu_seconds = 86400\n'
                         'lease_seconds = { train = 5400, eval = 14400 }\n')
        task = task_manifest.read_task(root)
        self.assertEqual(task_manifest.verifier_parameters(task)["gpu"], {
            "placement": "runner", "lease_seconds": {"train": 5400, "eval": 14400},
            "max_gpu_instances": 8})
        self.assertNotIn("lease_seconds", task_manifest.policy_from_task(task)["gpu"]["verifier"])
        self.assertIn("lease_seconds", task_manifest.effective_toml(root))

    def test_bad_keys_are_refused(self):
        for table in ('placement = "runner"', 'placement = "fanout"', "lease_seconds = { train = 5 }",
                      "lease_seconds = { train = 0, eval = 5 }", "lease = 5"):
            root = self.task(f"[gpu.verifier]\nmax_gpu_instances = 1\nmax_gpu_seconds = 1\n{table}\n")
            with self.subTest(table=table), self.assertRaises(task_manifest.ManifestError):
                task_manifest.read_task(root)


if __name__ == "__main__":
    unittest.main()
