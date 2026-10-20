"""Fan a verification out to several single-H100 Verifier GPU instances.

The CPU verifier (this image) keeps the judge, applies and checks the patch,
and gates. Measurement runs on N Driver-owned single-GPU runners created in
parallel with the shared broker in fcs2_verifier_gpu.py. Each runner receives
only the submission patch and runs `evaluator.py --shard-run SHARD N MODE`,
which measures its deterministic share of the articles with the unchanged
single-GPU harness (worker isolation, probes and scrub as before) and returns
one JSON document. Every runner is terminated before this returns.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time

import fcs2_verifier_gpu as broker

MAX_SHARD_BYTES = 4 * 1024 * 1024
# Kept back from the runner's remaining grant for writing and returning the
# shard result and terminating the runner. The shard's harness budget is the
# rest, so a slow submission fails inside the harness (a scored failure) before
# the runner's grant ends (which would be an unscored infrastructure failure).
RESULT_MARGIN_SECONDS = 240
SHARD_OUTPUT = "/workspace/frontier-verifier/output/shard.json"


class FanoutInfrastructureError(RuntimeError):
    """A runner could not be created, reached or terminated: no score."""


class ShardSubmissionError(RuntimeError):
    """A shard reported that the submission failed: a scored zero."""


def shard_command(shard: int, shards: int, mode: str, budget: int) -> str:
    """The broker's remote environment setup, running one shard instead of verify.py."""
    base = broker._remote_verifier_command(mode, 1)
    run = "\"$verifier_python\" /opt/frontier/verify.py "
    if run not in base or "result.json evidence.tar.zst" not in base:
        raise FanoutInfrastructureError("the shared remote command changed shape")
    shard_run = (f"\"$verifier_python\" /opt/frontier/task/evaluator.py --shard-run {int(shard)} {int(shards)} "
                 f"{'final' if mode == 'eval' else 'train'} /workspace/frontier-verifier/input/payload.tar "
                 f"{SHARD_OUTPUT} {int(budget)} ")
    return base.replace(run, shard_run).replace("result.json evidence.tar.zst", "shard.json")


def _payload_tar(patch: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo("payload")
        info.type, info.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(info)
        archive.add(patch, arcname="payload/solution.patch", recursive=False)


def _read_shard(archive_path: Path) -> dict:
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != ["shard.json"] or not members[0].isfile():
            raise FanoutInfrastructureError("a runner returned an unexpected output archive")
        if members[0].size > MAX_SHARD_BYTES:
            raise FanoutInfrastructureError("a runner returned an oversized shard result")
        document = json.loads(archive.extractfile(members[0]).read().decode("utf-8"))
        if isinstance(document, dict) and document.get('status') == 'infrastructure':
            print('fastkv shard infrastructure: ' + str(document.get('message'))[-2000:], file=sys.stderr)
        return document


def _one_shard(shard: int, shards: int, mode: str, patch: Path, seconds: int, config) -> dict:
    driver_url, verification_id, token, gpu_type, _seconds, _count = config
    scratch = Path(tempfile.mkdtemp(prefix=f"frontier-fanout-{shard}-",
                                    dir="/frontier" if os.access("/frontier", os.W_OK) else None))
    runner = None
    try:
        payload, dataset = scratch / "payload.tar", scratch / "dataset"
        _payload_tar(patch, payload)
        dataset.write_text(mode)
        inputs, outputs = scratch / "inputs.tar", scratch / "outputs.tar"
        broker._input_archive(inputs, payload, dataset)
        anchor = time.monotonic()
        runner = broker._create_runner(
            driver_url, verification_id, token, gpu_type, seconds, 1,
            idempotency_scope=f"fastkv-shard-v1:{mode}:{shards}:{shard}")
        deadline = broker._runner_deadline(runner, seconds, anchor=anchor)
        ssh = broker._ssh_argv(runner, scratch)
        environment = broker._ssh_environment(scratch)
        broker._upload_inputs(ssh, inputs, broker._remaining(deadline), environment, runner_deadline=deadline)
        budget = int(broker._remaining(deadline) - RESULT_MARGIN_SECONDS)
        if budget <= 0:
            raise FanoutInfrastructureError(f"shard {shard}: no time left after runner setup")
        returncode, stalled = broker._run_remote_verifier(
            ssh, shard_command(shard, shards, mode, budget), outputs, broker._remaining(deadline), environment,
            runner_deadline=deadline)
        try:
            return _read_shard(outputs)
        except (OSError, tarfile.TarError, ValueError) as error:
            raise FanoutInfrastructureError(
                f"shard {shard} returned no result (exit {returncode}, stalled {stalled})") from error
    except broker.VerifierGpuError as error:
        raise FanoutInfrastructureError(f"shard {shard}: {error}") from error
    finally:
        terminated = runner is None or broker._terminate_runner(
            driver_url, verification_id, token, str(runner["runner_id"]))
        shutil.rmtree(scratch, ignore_errors=True)
        if not terminated:
            raise FanoutInfrastructureError(f"shard {shard}: runner termination was not confirmed")


def run_shards(patch: Path, mode: str, shards: int, seconds: int) -> dict[int, dict]:
    """All shards in parallel, one single-GPU runner each; {shard: shard result}."""
    config = broker._config()
    with ThreadPoolExecutor(max_workers=shards) as pool:
        futures = {shard: pool.submit(_one_shard, shard, shards, mode, Path(patch), seconds, config)
                   for shard in range(shards)}
        results, errors = {}, []
        for shard, future in futures.items():
            try:
                results[shard] = future.result()
            except Exception as error:  # noqa: BLE001 - classified by the caller
                # The generic evaluator adapter hides exception messages from the agent.
                # Preserve the bounded cause in operator-only verifier stderr/evidence.
                print(f"fastkv fanout shard {shard}: {type(error).__name__}: {str(error)[-2000:]}", file=sys.stderr)
                errors.append(error)
    if errors:
        raise next((error for error in errors if isinstance(error, FanoutInfrastructureError)), errors[0])
    return results
