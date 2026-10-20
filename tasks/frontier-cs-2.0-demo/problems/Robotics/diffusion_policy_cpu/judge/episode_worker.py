"""Trusted child process; submitted code is loaded only inside SamplerBridge."""

import json
import os
from pathlib import Path
import resource
import sys

sys.dont_write_bytecode = True
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
os.environ.update(CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", SDL_VIDEODRIVER="dummy",
                  SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1",
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def main(request_path):
    import yaml
    from task_guards import InvalidSubmission, TaskIntegrityError
    from trusted_workload import execute_episode, write_json
    from private_context import load_context, validate_request
    request = json.loads(Path(request_path).read_text())
    context = load_context(root=request["private_context"],
                           mode=request["mode"] if request["profile"] == "evaluation" else None)
    validate_request(context, request)
    try:
        execute_episode(request["case"]["run"], request["checkpoint"], request["output"],
                        request["runtime_prefix"], submission_path=request["submission_path"],
                        sampler_seconds=request["sampler_seconds"],
                        execution_kind=request["case"]["side"],
                        expected_ema_sha256=context["seal"]["runtime"]["ema_sha256"])
    except BaseException as error:
        output = Path(request["output"])
        if output.is_dir():
            write_json(output / "failure.json", {
                "category": "invalid_submission" if isinstance(error, InvalidSubmission) else "measurement_unavailable",
                "type": type(error).__name__, "private_error": str(error)})
        raise


if __name__ == "__main__":
    main(sys.argv[1])
