"""GPU evaluator for flow_opd on the shared verifier (#757).

The shared verifier applies the patch with the task's edit-scope rules, runs check_tree (only the
two computational bodies may change; UTF-8 files) and calls evaluate(tree): the starter (bundle
pristine/, also the runtime adapter) with only the allowed files overlaid. Never replace the
qualification error with a CPU/synthetic score.

Each workload run (the unchanged shell launcher on the patched tree) executes in a fresh instance
of the shared isolate launcher (/opt/frontier/isolate.py): it sees the tree, the runtime adapter,
the transition guard, its run profile and corpus, the offline models and the GPU, and no other
judge file or platform variable.

Scoring. The retained launcher trains the student for 8 verified optimizer updates on the real
single-OCR-teacher kl_only path, then generates one 512x512 image per prompt-and-seed workload.
  raw_score = score_unbounded = 100 * mean ocr_reward over the scored suite (development in train,
  final in eval); 0 if any required workload is missing or invalid.
Metric (per workload, beat_spec_workload_passes):
  ocr_reward  the scorer's normalized, capped Levenshtein OCR reward in [0, 1], higher is better;
              passes if C > R (reference R < 1), or C >= R when R == 1 (ceiling).
beats_reference = 1 iff the graded suite (train-grading in train, final in eval) is complete and
every workload passes. reference_score is the frozen reference mean (x100), margin = score - it.
Train metrics: dev_eval_reward_ocr, dev_verified_optimizer_updates. Qualification, runtime and
missing-report faults raise TrustedEvaluationFailure (an unscored verification).
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

from judge.policy import SYMBOLS, check_surface
from judge.scoring import QualificationError, TrustedEvaluationFailure

# config.yaml (generated from task.toml) uses JSON syntax, a valid YAML subset.
CONFIG = json.loads((TASK_DIR/'config.yaml').read_text())
PRISTINE = TASK_DIR/'pristine'  # the starter repo the shared verifier applies the patch to
# The runtime adapter (frontier_runtime.py and frontier_shim/, from the starter) and the judge's
# transition guard are staged into each run's isolate instance as {in}/adapter/app and
# {in}/judge: frontier_runtime.py finds the guard at parents[2]/judge/transition_checks.py.
JUDGE_INPUTS = {'transition_checks.py': TASK_DIR/'judge'/'transition_checks.py'}
INSTANCE_ENTRY = TASK_DIR/'judge'/'instance_entry.py'
ENTRYPOINT = 'scripts/single_node/sd3_opd_example.sh'
# The baked offline snapshots below FRONTIER_MODEL_ROOT (docker/*-gpu.Dockerfile base).
MODEL_TARGETS = ('stable-diffusion-3.5-medium', 'SD3.5M-FlowGRPO-Text')


# The exact private metric schema is part of the evaluator's public API.
BEAT_METRICS = {"ocr_reward"}
# The train metrics the Agent receives (task.toml [feedback]); judge/judge/scoring.py fills them.
FEEDBACK_METRICS = ("dev_eval_reward_ocr", "dev_verified_optimizer_updates")


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
    if any(name not in metrics for name in FEEDBACK_METRICS):
        raise QualificationError('Incomplete development feedback.')
    return score, unbounded, message, metrics


def offline_model_root():
    """Resolve only baked snapshots; never accept a remote model repository ID."""
    value = os.environ.get('FRONTIER_MODEL_ROOT')
    if not value or not Path(value).is_absolute():
        raise QualificationError('Offline model root is unavailable.')
    try:
        root = Path(value).resolve(strict=True)
        for name in MODEL_TARGETS:
            target = (root/name).resolve(strict=True)
            if not target.is_relative_to(root) or not target.is_dir():
                raise QualificationError('Offline model snapshot is unavailable.')
    except OSError as exc:
        raise QualificationError('Offline model snapshot is unavailable.') from exc
    return root


def instance_inputs(output, profile_path):
    """The read-only inputs of a run's instance, staged below the run's private output: the
    adapter, the transition guard, the run profile and its corpus, the instance's first command."""
    staging = output/'instance-inputs'
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(PRISTINE, staging/'adapter'/'app', symlinks=False)
    (staging/'judge').mkdir()
    for name, path in JUDGE_INPUTS.items():
        shutil.copyfile(path, staging/'judge'/name)
    return {'adapter': staging/'adapter', 'judge': staging/'judge',
            'corpus': Path(json.loads(profile_path.read_text())['dataset']),
            'run-profile.json': profile_path, 'instance_entry.py': INSTANCE_ENTRY}


# The run's isolate launcher: the GPU, the offline models (read in place), a private loopback,
# AF_UNIX sockets and /dev/shm (DataLoader workers); its HOME holds the run's output
# (checkpoints and full state), bounded by SANDBOX_DISK_BYTES. Memory is bounded by the runner.
SANDBOX_DISK_BYTES = 128 << 30
SANDBOX_OUTPUT_BYTES = 256 << 20   # stdout and stderr, each (the private runtime log)


def sandbox_options(model_root):
    path = '{in}/adapter/app/frontier_shim' + os.pathsep + os.environ.get('PATH', '/usr/bin:/bin')
    env = {'PATH': path, 'FRONTIER_MODEL_ROOT': str(model_root), 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled', 'PYTHONDONTWRITEBYTECODE': '1',
           'FRONTIER_RUNTIME_ADAPTER': '{in}/adapter/app'}
    if 'LD_LIBRARY_PATH' in os.environ:
        env['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
    return {'env': env, 'read_paths': [str(model_root)], 'devices': ['nvidia'],
            'network': 'loopback', 'unix_sockets': True, 'dev_shm': True,
            'disk_bytes': SANDBOX_DISK_BYTES, 'output_bytes': SANDBOX_OUTPUT_BYTES}


def launch_workload(tree, profile, output, deadline, model_root):
    """Collect and execute the unchanged shell launcher on the patched real tree.

    The run executes in a fresh isolate instance (judge/judge/shell_workload.py). The shim
    supplies only offline/bounded runtime adaptation. Training still runs scripts/train_sd3.py
    with the original single-OCR kl_only computation.
    """
    import pytest
    from judge.shell_workload import ShellWorkloadPlugin, collection_context

    remaining = min(profile['max_runner_seconds'], deadline-time.monotonic())
    if remaining <= 0:
        raise TimeoutError('Total evaluation deadline exhausted.')
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output/'run-profile.json'
    profile_path.write_text(json.dumps(profile))
    workload = ShellWorkloadPlugin(tree, ENTRYPOINT, instance_inputs(output, profile_path),
                                   sandbox_options(model_root), output,
                                   min(deadline, time.monotonic()+remaining))
    with collection_context(tree, output):
        exit_code = pytest.main(
            ['-q', '-s', '-p', 'no:cacheprovider', 'scripts/single_node/sd3_opd_example.sh'],
            plugins=[workload])
    if workload.infrastructure_error is not None:   # a boundary that could not be built
        raise TrustedEvaluationFailure('isolate_unavailable') from workload.infrastructure_error
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


def prepare():
    try:
        qualification()
    except QualificationError:
        return {'ready':False, 'status':'pending_authorized_gpu_qualification',
                'authoritative':False}
    return {'ready':True, 'status':'qualified'}


def check_tree(tree, pristine, changes):
    """Shared-verifier hook: in the two retained modules only the bodies of
    pipeline_with_logprob and sde_step_with_logprob may change (imports, decorators,
    signatures and module-level wiring stay as they are), and every changed file is UTF-8."""
    tree, pristine = Path(tree), Path(pristine)
    undecodable = []
    for change in changes:
        if change.status == 'D':
            continue
        try:
            (tree/change.path).read_bytes().decode('utf-8')
        except UnicodeDecodeError:
            undecodable.append(change.path)
    if undecodable:
        return 'malformed', undecodable
    # A broken trusted scaffold must not be attributed to the candidate.
    check_surface(pristine, pristine)
    try:
        check_surface(tree, pristine)
    except (ValueError, SyntaxError, OSError):
        return 'out_of_scope', [path for path in SYMBOLS
                                if any(c.path == path for c in changes)] or sorted(SYMBOLS)
    return None


def evaluate(tree: str) -> tuple[float,float,str,dict]:
    final = os.environ.get('FRONTIER_SUBMISSION_ROLE')=='final'
    deadline = time.monotonic()+CONFIG['runtime']['timeout_seconds']
    try:
        q = qualification()
        from judge.runner import run_evaluation
        model_root = offline_model_root()
        runs = run_evaluation(Path(tree), q, final=final,
                              deadline=deadline, model_root=model_root)
        # The evaluator owns workload dispatch; the runner provides frozen run
        # profiles and consumes the actual process results for private scoring.
        try:
            profile, output = next(runs)
            while True:
                try:
                    measured = launch_workload(Path(tree), profile, output,
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
