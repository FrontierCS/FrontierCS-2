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
    from task_guards import InvalidSubmission, TaskIntegrityError  # noqa: F401
    from trusted_workload import execute_episode, write_json
    request = json.loads(Path(request_path).read_text())
    try:
        execute_episode(request["case"]["run"], request["checkpoint"], request["output"],
                        request["runtime_prefix"], submission_path=request["submission_path"],
                        sampler_seconds=request["sampler_seconds"],
                        execution_kind=request["case"]["side"])
    except BaseException as error:
        output = Path(request["output"])
        if output.is_dir():
            write_json(output / "failure.json", {
                "category": ("invalid_submission" if isinstance(error, InvalidSubmission)
                             else "candidate_timeout" if isinstance(error, TimeoutError)
                             and request["case"]["side"] == "candidate"
                             else "measurement_unavailable"),
                "type": type(error).__name__, "private_error": str(error)})
        raise


if __name__ == "__main__":
    main(sys.argv[1])
