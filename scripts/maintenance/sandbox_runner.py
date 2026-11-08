"""A task's judge with its command test sandboxes on Modal (authoring only; not in CI).

Runs `evaluate(tree)` of a task.toml task here, as the CPU control image does, with the shared
adapter's Driver lease and SSH gateway replaced by Modal Sandboxes of the published runner images
(the declared GPUs, or a CPU machine of exactly the declared cores and memory; no network) and
an exec as the runner's SSH user (uid 10001). Each sandbox's runner image kind (agent_gpu,
verifier_gpu, agent_cpu, verifier_cpu) maps to a published image; a verifier_gpu image gets this
checkout's judge bundle at /opt/frontier/task, where the task's Verifier GPU image holds it (a
verifier_cpu runner image already holds it, as the Verifier image does). Everything above that
transport is the real code: run_sandboxes() builds the upload, step and collect commands, the
judge classifies the results. No Driver, Management or site call; it costs the task's GPUs.

  set -a; . ~/.config/fcs2/modal.env; set +a
  uv run --no-project --with modal --with zstandard python scripts/maintenance/sandbox_runner.py \
    --task <task dir> --image agent_gpu=<image@sha256:...> [--image verifier_gpu=<...>] \
    [--image agent_cpu=<agent CPU runner image> --image verifier_cpu=<...>] \
    --candidate reference|<patch> [--mode train|eval] [--bundle <staged judge bundle>]

The judge and the adapter come from this checkout; --bundle replaces the staged judge/ +
pristine/ when the control image builds more into it (e.g. nano_world_model's public_data/).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2]
RUNNER_UID = 10001
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
VERIFICATION = "verification_01ARZ3NDEKTSV4RRFFQ69G5FAV"


def remote_exec(sandbox_id: str, command: str) -> int:
    """The `ssh` stand-in: run `command` in the sandbox as the runner user with sshd's cleared
    environment; local stdin goes up, remote stdout/stderr come back."""
    sandbox = modal.Sandbox.from_id(sandbox_id)
    body = sys.stdin.buffer.read()
    process = sandbox.exec(
        "setpriv", f"--reuid={RUNNER_UID}", f"--regid={RUNNER_UID}", "--clear-groups",
        "env", "-i", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME=/workspace", "/bin/sh", "-c", command, text=False)
    if body:
        process.stdin.write(body)
    process.stdin.write_eof()
    process.stdin.drain()

    def pump(source, target):
        for chunk in source:
            target.write(chunk)
            target.flush()

    errors = threading.Thread(target=pump, args=(process.stderr, sys.stderr.buffer), daemon=True)
    errors.start()
    pump(process.stdout, sys.stdout.buffer)
    errors.join()
    return process.wait()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--image", action="append", required=True, metavar="KIND=REF",
                        help="the published image of a runner image kind (agent_gpu, "
                             "verifier_gpu, agent_cpu, verifier_cpu)")
    parser.add_argument("--bundle", type=Path,
                        help="a staged judge bundle (default: task_manifest.judge_bundle)")
    parser.add_argument("--candidate", required=True,
                        help="'reference' (judge/reference.patch) or a patch file")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    options = parser.parse_args()

    sys.path.insert(0, str(REPO / "scripts"))
    import task_manifest

    root = options.task.resolve()
    task = task_manifest.read_task(root)
    scratch = Path(tempfile.mkdtemp(prefix="sandbox-runner-"))
    parameters = scratch / "verifier.json"
    parameters.write_text(json.dumps(task_manifest.verifier_parameters(task)))
    gpu = load("fcs2_verifier_gpu", REPO / "tasks/shared/verifier-gpu/fcs2_verifier_gpu.py")
    gpu.PARAMETERS = parameters

    bundle = options.bundle.resolve() if options.bundle else task_manifest.judge_bundle(root)
    app = modal.App.lookup("sandbox-runner-check", create_if_missing=True)
    references = dict(item.split("=", 1) for item in options.image)
    images = {kind: modal.Image.from_registry(reference) for kind, reference in references.items()}
    if "verifier_gpu" in images:
        images["verifier_gpu"] = images["verifier_gpu"].add_local_dir(
            bundle, "/opt/frontier/task", copy=True)
    sandboxes: dict[str, modal.Sandbox] = {}
    lock = threading.Lock()

    def create_runner(driver_url, verification_id, token, gpu_type, seconds, gpu_count=1, *,
                      window=None, idempotency_scope=None, image=None, machine=None):
        started = time.monotonic()
        if machine is not None:
            # A CPU machine as the Driver makes it (core driver-modal sandbox_resources): request
            # and limit are the declared cores and memory.
            cores, memory = machine["cpu_millis"] / 1000, machine["memory_bytes"] // (1024 * 1024)
            shape = {"cpu": (cores, cores), "memory": (memory, memory)}
        else:
            # The Driver's Modal GPU name (core driver-modal modal_gpu_type): H100 is the
            # exact-H100 selector "H100!"; a plain "H100" may run on an H200.
            modal_gpu = "H100!" if gpu_type.upper() == "H100" else gpu_type.upper()
            shape = {"gpu": modal_gpu if gpu_count == 1 else f"{modal_gpu}:{gpu_count}",
                     "cpu": 8, "memory": 64 * 1024}
        sandbox = modal.Sandbox.create(app=app, image=images[image], timeout=seconds + 900,
                                       block_network=True, **shape)
        with lock:
            number = len(sandboxes)
            runner_id = "gpurunner_0" + "".join(
                CROCKFORD[(number >> (5 * i)) & 31] for i in range(25))
            sandboxes[runner_id] = sandbox
        print(f"### runner {runner_id} ({idempotency_scope}) = {sandbox.object_id}, "
              f"up after {time.monotonic() - started:.0f} s", flush=True)
        return {"runner_id": runner_id, "state": "ready",
                **(dict(machine) if machine is not None else {"gpu_count": gpu_count}),
                "granted_seconds": seconds, "image": image}

    def terminate_runner(driver_url, verification_id, token, runner_id, *, window=None):
        with lock:
            sandbox = sandboxes.pop(runner_id)
        sandbox.terminate()
        return True

    gpu._create_runner = create_runner
    gpu._terminate_runner = terminate_runner
    gpu._ssh_argv = lambda runner, directory: [
        sys.executable, str(Path(__file__).resolve()), "_exec",
        sandboxes[runner["runner_id"]].object_id]
    gpu._ssh_environment = lambda directory: dict(os.environ)
    gpu._driver_config = lambda: ("http://driver.invalid", VERIFICATION, "unused")

    # As in the task's Verifier image (its ENV, or the shared Verifier base's).
    os.environ.update({
        "FRONTIER_MAX_SSH_BYTES_PER_DIRECTION": task_manifest.verifier_channel_bytes(root),
        "FRONTIER_VERIFICATION_MODE": options.mode,
        "FRONTIER_SUBMISSION_ROLE": "final" if options.mode == "eval" else "development"})
    sys.path.insert(0, str(bundle))
    evaluator = load("evaluator", bundle / "evaluator.py")

    tree = scratch / "tree"
    shutil.copytree(bundle / "pristine", tree, symlinks=True)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    patch = bundle / "reference.patch" if options.candidate == "reference" else Path(options.candidate)
    subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch.resolve())], cwd=tree,
                   check=True, env={**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch)})
    started = time.monotonic()
    try:
        score, raw, message, metrics = evaluator.evaluate(str(tree))
    finally:
        for runner_id in list(sandboxes):
            terminate_runner(None, None, None, runner_id)
    summary = {"seconds": round(time.monotonic() - started, 1), "score": score, "raw": raw,
               "message": message,
               "metrics": {key: value for key, value in metrics.items()
                           if key != "private_execution"}}
    print("SANDBOX_RUNNER_RESULT " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "_exec":
        raise SystemExit(remote_exec(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
