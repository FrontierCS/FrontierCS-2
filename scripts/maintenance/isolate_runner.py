"""The shared isolate launcher's suite on a Verifier GPU runner (authoring only; not in CI).

tests/test_isolate.py runs each test once per backend the host can build. EKS nodes and
developer hosts build `landlock`; the GPU runners (Modal, gVisor: no Landlock) need `userns`,
which only a runner can show. This starts one Modal Sandbox the way the platform starts a
Verifier GPU runner (a published Verifier GPU image, one GPU, no network) and runs the suite as
the runner's SSH user (uid 10001) with TMPDIR=/workspace, as the GPU adapter sets it, with the
image's judge interpreter (/opt/frontier/runtime-python, the one /opt/frontier/verifier runs),
then prints the backend report and the escape/limit/worker/CUDA test results. It also starts a
worker the way a judge does (the interpreter on a file outside /opt/frontier, plain and `-I`)
and imports isolate there with no sys.path edit: the image's frontier.pth must provide it.
No Driver, Management or site call; it costs one GPU for a few minutes.

  MODAL_PROFILE=frontier-cs-2 modal run scripts/maintenance/isolate_runner.py \
    --image <a task's verifier-gpu image@sha256:...> [--gpu H100]

The module and the tests come from this checkout.
"""

import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parents[2]
CHECK = "/opt/isolate-check"
RUNNER_UID = 10001

REPORT = r"""
import json, os, sys
sys.path.insert(0, "tasks/shared/lib")
import isolate
print(json.dumps({"uid": os.getuid(), "landlock_abi": isolate.landlock_abi(),
                  "landlock": isolate.backend_available("landlock"),
                  "userns": isolate.backend_available("userns"),
                  "auto": isolate.choose_backend(),
                  "self_test": isolate._probe(isolate.choose_backend())}, indent=1))
"""

# A judge's worker: the judge interpreter on a file in another directory (sys.path[0] is that
# directory, not /opt/frontier), so only the image's default import path can provide the module.
WORKER = "import isolate; print(isolate.__file__)"

app = modal.App("isolate-runner-check")


@app.local_entrypoint()
def main(image: str, gpu: str = "H100", timeout: int = 1800):
    built = (
        modal.Image.from_registry(image)
        .add_local_file(
            str(REPO / "tasks/shared/lib/isolate.py"),
            f"{CHECK}/tasks/shared/lib/isolate.py",
            copy=True,
        )
        .add_local_file(
            str(REPO / "tests/test_isolate.py"),
            f"{CHECK}/tests/test_isolate.py",
            copy=True,
        )
        .add_local_file(
            str(REPO / "tests/isolate_layers.py"),
            f"{CHECK}/tests/isolate_layers.py",
            copy=True,
        )
        .add_local_file(
            str(REPO / "tests/deployment_test_guard.py"),
            f"{CHECK}/tests/deployment_test_guard.py",
            copy=True,
        )
        .run_commands(
            f"install -d {CHECK}/worker",
            f"printf '%s\\n' '{WORKER}' > {CHECK}/worker/evaluator.py",
            f"chmod -R a+rX {CHECK}",
            f"install -d -o {RUNNER_UID} -g {RUNNER_UID} /workspace",
        )
    )
    sandbox = modal.Sandbox.create(
        app=app,
        image=built,
        gpu=gpu,
        cpu=8,
        memory=64 * 1024,
        timeout=timeout,
        block_network=True,
    )
    as_runner = (
        f"setpriv --reuid={RUNNER_UID} --regid={RUNNER_UID} --clear-groups "
        "env -i PATH=/opt/frontier/python/bin:/usr/local/bin:/usr/bin:/bin "
        "HOME=/workspace TMPDIR=/workspace CUDA_VISIBLE_DEVICES=0 "
    )
    python = "/opt/frontier/runtime-python"
    script = (
        f"set -u; cd {CHECK}; uname -a; (dmesg 2>/dev/null | head -n 3) || true; "
        "nvidia-smi -L || true; ls -l /dev/nvidia* || true; "
        f"{as_runner}{python} -I -c '{REPORT}'; "
        f"(cd / && {as_runner}{python} {CHECK}/worker/evaluator.py && "
        f"{as_runner}{python} -I {CHECK}/worker/evaluator.py) 2>&1; "
        "echo WORKER_IMPORT_EXIT=$?; "
        f"(cd tests && {as_runner}{python} -m unittest -v test_isolate) 2>&1; "
        "echo ISOLATE_SUITE_EXIT=$?"
    )
    try:
        process = sandbox.exec("bash", "-c", script, timeout=timeout)
        for line in process.stdout:
            print(line, end="")
        print(process.stderr.read(), end="")
        process.wait()
    finally:
        sandbox.terminate()
