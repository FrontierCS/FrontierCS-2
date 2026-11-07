"""Closed-loop Push-T judge: the candidate's conditional sampler in the frozen diffusion policy.

The shared verifier applies the patch (task.toml [submission]: the policy file and Python
helpers under diffusion_policy/sampler_extensions/), runs check_tree (the source rules
task.toml cannot express: only the conditional_sample body may change, helpers are bounded in
count and bytes and parse) and then evaluate(tree). The dataset object is a composite of
hardware-sealed private datasets; hardware_dataset selects the one whose hardware class admits
this machine.

Score: per workload w (one initial-condition group of complete episodes), Q = mean
episode-maximum reward, L = mean predict_action seconds, T = median episode-process seconds;
with the fixed, pre-measured naive (20-step DDPM) constants Qn, Ln, Tn:
gain = max(0, (Q - Qn) / (1 - Qn)), U_w = gain * min(Ln / L, Tn / T); U = mean over the
workloads; raw_score = 100 U / (1 + U), score_unbounded = U. Each sampler (candidate and
restored reference alike) runs as a worker under the shared isolate launcher (sampler_bridge.py).

Metrics, one row per workload in beat_spec_workload_passes, against the reference measured in
the same evaluation:
  mean_episode_max_reward         higher is better; passes when candidate >= reference.
  episode_success_rate            higher is better; passes when candidate >= reference.
  mean_predict_action_seconds     lower is better; passes when candidate < reference.
  complete_workload_wall_seconds  lower is better; passes when candidate < reference.
beats_reference = 1 iff every workload passes all four. Train also returns the four dev_*
candidate summaries (DEV_FEEDBACK). A candidate that breaks the sampler protocol or the
model-use budget during measurement is rejected as content_policy; a candidate sampler over its
130 s lifetime, a candidate episode process over 150 s or an exhausted suite budget is a scored
zero (the reason in the private message). Judge faults (a reference timeout, bad data) raise.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
from task_guards import (DEV_FIELDS, CandidateTimeout, InvalidSubmission,  # noqa: E402
                         TaskIntegrityError, check_source)

PRISTINE = TASK / "pristine"  # the starter repo; the shared verifier overlays the submission on a copy
DEV_FEEDBACK = ("dev_mean_episode_max_reward", "dev_episode_success_rate",
                "dev_mean_predict_action_seconds", "dev_total_workload_wall_seconds")
assert DEV_FEEDBACK == DEV_FIELDS

# check_tree categories for task_guards.check_source's rejection messages.
CATEGORIES = {
    "The helper file or byte allowance was exceeded.": "too_large",
    "The sampler source byte allowance was exceeded.": "too_large",
    "The required sampler interface must remain valid Python.": "out_of_scope",
    "Only the conditional_sample body may be modified.": "out_of_scope",
    "Source files must be UTF-8 Python.": "out_of_scope",
    "A helper is not valid Python.": "content_policy",
}


class CalibrationRequired(RuntimeError):
    """Authoring prerequisite failure, never a solver score of zero."""


def check_tree(tree: str, pristine: str, changes: list):
    """Shared-verifier hook: the source rules of task_guards.check_source."""
    try:
        check_source(Path(tree), Path(pristine))
    except InvalidSubmission as error:
        category = CATEGORIES.get(str(error))
        if category is None:
            raise
        return category, [c.path for c in changes]
    return None


def _writable_copy(source: Path, scratch: Path) -> Path:
    tree = scratch / "tree"
    shutil.copytree(source, tree, symlinks=True)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o600))
    return tree


@contextmanager
def prepared_source(tree):
    """A private, writable copy of a judged tree for one episode's sampler."""
    scratch = Path(tempfile.mkdtemp(prefix="candidate-source-"))
    try:
        yield _writable_copy(Path(tree), scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@contextmanager
def reference_source():
    """The pristine starter with reference.patch applied (trusted, judge-owned)."""
    scratch = Path(tempfile.mkdtemp(prefix="reference-source-"))
    try:
        tree = _writable_copy(PRISTINE, scratch)
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch.resolve())}
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(TASK / "reference.patch")],
                       cwd=tree, check=True, capture_output=True, timeout=60, env=env)
        yield tree
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def evaluate(tree: str):
    import yaml
    from hardware_dataset import actual_identity, select_dataset
    from measurement_runtime import measure_suite
    from native_data import materialize_dataset, native_mode, read_dataset
    from option_b import assemble_scores
    from private_context import accepted_calibrations, binding, load_context
    from runtime_layout import validate_layout
    try:
        mode = native_mode(os.environ)
        dataset = os.environ.get("FRONTIER_DATASET_PATH")
        if not dataset:
            raise TaskIntegrityError("The selected private dataset is missing")
        # As the release's inline dataset-selection command did before handing over: private
        # files owner-only, and this process (and every measurement process) pinned to two CPUs.
        os.umask(0o077)
        identity = actual_identity()
        scratch = Path(tempfile.mkdtemp(prefix="hardware-dataset-"))
        try:
            selected = select_dataset(dataset, "eval" if mode == "final" else "train", identity,
                                      scratch, read_dataset)
            with materialize_dataset(selected, mode, TASK.parent) as private_root:
                context = load_context(root=private_root, mode=mode)
                calibrations = accepted_calibrations(context)
                config = yaml.safe_load((TASK / "config.yaml").read_text())
                runtime = context["seal"]["runtime"]
                validate_layout(runtime, config["runtime"]["measurement"])
                output = private_root / "measurement"
                try:
                    measurement = measure_suite(config, tree, output, runtime["prefix"],
                                                runtime["checkpoint"], mode, context=context)
                except InvalidSubmission:
                    # A candidate that broke the sampler protocol or its model-use budget.
                    return 0.0, 0.0, "Submission rejected by the sampler protocol.", {
                        "rejected": "content_policy"}
                except CandidateTimeout as error:
                    # The candidate's own sampler, episode or the suite budget ran out: a scored
                    # zero, the reason in the (private) message.
                    return 0.0, 0.0, f"Candidate timeout: {error}", {}
                result = assemble_scores(measurement, baseline=calibrations[mode],
                                         expected_binding=binding(context, mode),
                                         suite=context["suites"][mode])
                if result["status"] != "complete":
                    raise TaskIntegrityError("Complete bound scoring is unavailable")
                return result["native_result"]
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except InvalidSubmission:
        raise
    except (OSError, ValueError, KeyError, TypeError, TaskIntegrityError) as error:
        raise CalibrationRequired("The selected dataset, runtime and exact-version acceptance must be valid") from error


def measure_for_review(solution_path, *, config, output, runtime_prefix, checkpoint, environ):
    """Trusted author-only entrypoint; execution remains separately disabled."""
    from measurement_runtime import measure_suite
    from task_guards import suite_mode
    return measure_suite(config, solution_path, output, runtime_prefix, checkpoint, suite_mode(environ))
