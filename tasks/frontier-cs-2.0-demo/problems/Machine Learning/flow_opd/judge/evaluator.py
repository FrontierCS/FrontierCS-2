"""Source-only draft evaluator. No authoritative run before GPU qualification.

Never replace the qualification error with a CPU/synthetic score. The metric
arithmetic is unit-tested separately; that is not a reference Evaluation.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

TASK_DIR = Path(__file__).resolve().parent
# importlib-by-path callers do not automatically add the evaluator directory.
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from submission_patch import apply_submission_patch, validate_patch
from judge.policy import check_surface
from judge.scoring import QualificationError, TrustedEvaluationFailure, failure_metrics

# config.yaml uses JSON syntax, a valid YAML subset.
CONFIG = json.loads((TASK_DIR/'config.yaml').read_text())
ENTRYPOINT = 'scripts/single_node/sd3_opd_example.sh'


# The exact private metric schema is part of the evaluator's public API.
BEAT_METRICS = {"ocr_reward"}


def checked_result(result):
    score, unbounded, message, metrics = result
    if (type(score) not in (int,float) or not math.isfinite(score) or not 0 <= score <= 100
            or type(unbounded) not in (int,float) or not math.isfinite(unbounded)):
        raise QualificationError('Invalid continuous score.')
    rows = metrics["beat_spec_workload_passes"]
    identifiers = [row["workload"] for row in rows]
    if len(set(identifiers)) != len(identifiers) or any(not isinstance(i, str) or not i for i in identifiers):
        raise QualificationError('Invalid private workload audit.')
    for row in rows:
        if set(row["metrics"]) != BEAT_METRICS or any(type(v) is not bool for v in row["metrics"].values()):
            raise QualificationError('Invalid OCR comparison audit.')
    # aggregate() compares unrounded C_w > R_w, including strict failure on ties.
    expected = int(bool(rows) and all(row["metrics"]["ocr_reward"] for row in rows))
    if type(metrics["beats_reference"]) is not int or metrics["beats_reference"] != expected:
        raise QualificationError('Incorrect strict conjunction.')
    if "reference_score" not in metrics or "margin" not in metrics:
        raise QualificationError('Incomplete reference accounting.')
    return score, unbounded, message, metrics


def offline_model_root():
    """Resolve only baked snapshots; never accept a remote model repository ID."""
    value = os.environ.get('FRONTIER_MODEL_ROOT')
    if not value or not Path(value).is_absolute():
        raise QualificationError('Offline model root is unavailable.')
    try:
        root = Path(value).resolve(strict=True)
        for model in CONFIG['runtime']['huggingface_models']:
            target = (root/model['target']).resolve(strict=True)
            if not target.is_relative_to(root) or not target.is_dir():
                raise QualificationError('Offline model snapshot is unavailable.')
    except OSError as exc:
        raise QualificationError('Offline model snapshot is unavailable.') from exc
    return root


def launch_workload(tree, profile, output, deadline, model_root):
    """Collect and execute the unchanged shell launcher on the patched real tree.

    The shim supplies only offline/bounded runtime adaptation. Training still
    runs scripts/train_sd3.py with the original single-OCR kl_only computation.
    """
    import pytest
    from judge.shell_workload import ShellWorkloadPlugin, collection_context

    remaining = min(profile['max_runner_seconds'], deadline-time.monotonic())
    if remaining <= 0:
        raise TimeoutError('Total evaluation deadline exhausted.')
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output/'run-profile.json'
    profile_path.write_text(json.dumps(profile))
    adapter = TASK_DIR/'harbor/app'
    env = {k:v for k,v in os.environ.items() if k in
           {'PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','TMPDIR','LANG'}}
    env.update(FRONTIER_MODEL_ROOT=str(model_root), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1',
               FRONTIER_RUNTIME_ADAPTER=str(adapter), FRONTIER_RUN_PROFILE=str(profile_path.resolve()),
               FRONTIER_RUN_OUTPUT=str(output.resolve()))
    env['PATH'] = str(adapter/'frontier_shim') + os.pathsep + env.get('PATH','/usr/bin:/bin')
    workload = ShellWorkloadPlugin(tree, ENTRYPOINT, env, output,
                                   min(deadline, time.monotonic()+remaining))
    with collection_context(tree, output):
        exit_code = pytest.main(
            ['-q', '-s', '-p', 'no:cacheprovider', 'scripts/single_node/sd3_opd_example.sh'],
            plugins=[workload])
    if exit_code == 0 and (workload.executions != 1 or workload.returncode != 0):
        raise RuntimeError('The original upstream entrypoint was not executed.')
    progress_path = output/'progress.json'
    if not progress_path.is_file():
        raise TrustedEvaluationFailure('workload_report_missing')
    if progress_path.stat().st_size > 1024*1024:
        raise ValueError('Oversized result.')
    result = json.loads(progress_path.read_text())
    if exit_code:
        result['complete'] = False
    return result


def qualification():
    q = json.loads((TASK_DIR/'judge/qualification.json').read_text())
    if q.get('status')!='qualified':
        raise QualificationError('GPU runtime qualification is incomplete.')
    required = ('profile','suite_manifest','reference_cache','measured_reference_scores',
                'peak_gpu_memory_mb','measured_runner_seconds')
    if any(q.get(k) is None for k in required) or q.get('reference_repetitions',0)<1:
        raise QualificationError('Qualification evidence incomplete.')
    # Baseline difficulty, solver evidence and control-memory profiling belong
    # to release review; they cannot gate the reference runs that measure them.
    if not q.get('development_quota_qualified'):
        raise QualificationError('Resource qualification incomplete.')
    if CONFIG['evaluation']['max_dev_submissions']<1:
        raise QualificationError('Development quota is not qualified.')
    return q


def safe_patch_reason(message):
    text = message.lower()
    categories = (("empty", "Empty patch."), ("allow", "Patch exceeds the allowed source surface."),
                  ("outside", "Patch exceeds the allowed source surface."), ("binary", "Binary patches are forbidden."),
                  ("size", "Patch exceeds the size limit."), ("unsafe", "Unsafe patch path."),
                  ("does not apply", "Patch does not apply to the pristine source."),
                  ("could not read", "Submission file is missing or unreadable."))
    return next((public for token, public in categories if token in text), "Malformed patch submission.")


def prepare():
    try:
        qualification()
    except QualificationError:
        return {'ready':False, 'status':'pending_authorized_gpu_qualification',
                'authoritative':False}
    return {'ready':True, 'status':'qualified'}


def evaluate(solution_path: str) -> tuple[float,float,str,dict]:
    final = os.environ.get('FRONTIER_SUBMISSION_ROLE')=='final'
    result = None
    deadline = time.monotonic()+CONFIG['runtime']['timeout_seconds']
    try:
        q = qualification()
        from judge.runner import comparison_context, run_evaluation
        _, _, _, ids, reference, feedback_reference = comparison_context(q, final=final)
        metrics = failure_metrics(ids, reference=reference, final=final,
                                  feedback_reference=feedback_reference)
        # A broken trusted scaffold must not be attributed to the candidate.
        pristine = TASK_DIR/'harbor/app'
        check_surface(pristine, pristine)
        # Validate source packaging without importing or executing submitted Python.
        try:
            patch_text = Path(solution_path).read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return 0.0, 0.0, 'Invalid submission: patch is not UTF-8.', metrics
        valid_patch, patch_message, _ = validate_patch(patch_text, CONFIG)
        if not valid_patch:
            return 0.0, 0.0, 'Invalid submission: ' + safe_patch_reason(patch_message), metrics
        result = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not result.ok:
            raise TrustedEvaluationFailure('trusted_runtime_unclassified')
        try:
            check_surface(result.workdir, TASK_DIR/'harbor/app')
        except (ValueError, SyntaxError):
            return 0.0, 0.0, 'Invalid submission: source surface rejected.', metrics
        model_root = offline_model_root()
        runs = run_evaluation(Path(result.workdir), q, final=final,
                              deadline=deadline, model_root=model_root)
        # The evaluator owns workload dispatch; the runner provides frozen run
        # profiles and consumes the actual process results for private scoring.
        try:
            profile, output = next(runs)
            while True:
                try:
                    measured = launch_workload(Path(result.workdir), profile, output,
                                               deadline, model_root)
                except (TimeoutError, ValueError, RuntimeError) as exc:
                    profile, output = runs.throw(exc)
                else:
                    profile, output = runs.send(measured)
        except StopIteration as completed:
            return checked_result(completed.value)
        finally:
            runs.close()
    except TrustedEvaluationFailure:
        raise
    except QualificationError as error:
        raise TrustedEvaluationFailure('reference_or_qualification_invalid') from error
    except Exception as error:
        raise TrustedEvaluationFailure('trusted_runtime_unclassified') from error
    finally:
        if result is not None and result.cleanup_root:
            shutil.rmtree(result.cleanup_root,ignore_errors=True)

if __name__ == '__main__':
    if len(sys.argv)!=2:
        raise TrustedEvaluationFailure('evaluator_invocation_invalid')
    else:
        print(json.dumps(evaluate(sys.argv[1]),allow_nan=False))
