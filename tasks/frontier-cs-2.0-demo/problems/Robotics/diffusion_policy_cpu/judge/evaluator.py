"""Receipt-gated private evaluation. No public execution before reference acceptance.

The preserved execution target is diffusion_policy/env_runner/pusht_keypoints_runner.py
and its PushTKeypointsRunner.run method. Numeric calibration and a verified
score definition and final suite bindings are mandatory before evaluation.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
from submission_patch import apply_submission_patch
from task_guards import InvalidSubmission, MAX_PATCH_BYTES, TaskIntegrityError, load_lock, validate_tree


class CalibrationRequired(RuntimeError):
    """Authoring prerequisite failure, never a solver score of zero."""


@contextmanager
def prepared_submission(solution_path: str | Path, config: dict):
    patch = Path(solution_path)
    if (patch.is_symlink() or not patch.is_file()
            or patch.stat().st_size > MAX_PATCH_BYTES):
        raise InvalidSubmission("Provide one bounded regular patch file.")
    result = apply_submission_patch(patch, TASK, config)
    if not result.ok or result.workdir is None:
        if result.candidate_rejection:
            raise InvalidSubmission("The patch does not apply to the permitted source surface.")
        raise TaskIntegrityError("Submission preparation is unavailable.")
    try:
        evidence = validate_tree(result.workdir, TASK / "harbor/app", load_lock(TASK))
        yield result.workdir, evidence
    finally:
        if result.cleanup_root is not None:
            shutil.rmtree(result.cleanup_root)


def evaluate(solution_path: str):
    import os
    import yaml
    from measurement_runtime import measure_suite
    from native_data import materialize_dataset, native_mode
    from option_b import assemble_scores
    from private_context import accepted_calibrations, authorize, binding, load_context
    from task_guards import TaskIntegrityError
    try:
        mode = native_mode(os.environ)
        dataset = os.environ.get("FRONTIER_DATASET_PATH")
        if not dataset:
            raise TaskIntegrityError("The selected private dataset is missing")
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
        with materialize_dataset(dataset, mode, TASK.parent) as private_root:
            context = load_context(root=private_root, mode=mode)
            authorize(context, "evaluation")
            calibrations = accepted_calibrations(context)
            config = yaml.safe_load((TASK / "config.yaml").read_text())
            runtime = context["seal"]["runtime"]
            from runtime_layout import validate_layout
            validate_layout(runtime, config["runtime"]["measurement"])
            output = private_root / "measurement"
            try:
                measurement = measure_suite(config, solution_path, output, runtime["prefix"],
                                            runtime["checkpoint"], mode, context=context)
            except InvalidSubmission:
                # Only a typed candidate refusal after trusted prerequisites is scored.
                from candidate_failure import diagnostic
                return 0.0, 0.0, "Submission rejected by source policy.", {
                    "candidate_diagnostic": diagnostic("source_policy"),
                }
            result = assemble_scores(measurement, baseline=calibrations[mode],
                expected_binding=binding(context, mode), suite=context["suites"][mode], execution_context=context)
            if result["status"] != "complete":
                raise TaskIntegrityError("Complete bound scoring is unavailable")
            return result["native_result"]
    except InvalidSubmission:
        raise
    except (OSError, ValueError, KeyError, TypeError, TaskIntegrityError) as error:
        raise CalibrationRequired("The selected dataset, runtime and exact-version acceptance must be valid") from error


def measure_for_review(solution_path, *, config, output, runtime_prefix, checkpoint, environ):
    """Trusted author-only entrypoint; execution remains separately disabled."""
    from measurement_runtime import measure_suite
    from task_guards import suite_mode
    return measure_suite(config, solution_path, output, runtime_prefix, checkpoint, suite_mode(environ))


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: evaluator.py SOLUTION.patch", file=sys.stderr)
        return 2
    try:
        result = evaluate(sys.argv[1])
        print(json.dumps(result, allow_nan=False))
    except CalibrationRequired:
        print(json.dumps({"status": "unavailable", "score": None,
                          "message": "This draft is not enabled for evaluation."}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
