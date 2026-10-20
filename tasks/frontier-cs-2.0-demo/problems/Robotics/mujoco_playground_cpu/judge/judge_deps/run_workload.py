"""Train both sides on the verifier GPU and measure both in the pristine env.

One training per (side, train seed); its policy artifact is evaluated under each
condition the suite pairs with that seed. See ``runtime.py`` for the trust
model. Exit status: 0 with a JSON payload on stdout (either measured workloads or
a ``candidate_failure``); 75 for a setup deadline; anything else is a trusted
(reference or infrastructure) failure recorded in the private journal.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import sysconfig
import tempfile
import time

import yaml

from runtime import (HERE, ChildRunner, PhaseFailed, PhaseTimeout, SetupDeadlineExpired,
                     admit_pristine, admit_trusted_children, harden_process,
                     load_source_manifest, sha256)
from private_evidence import Journal, report_bytes

TASK = HERE.parent
GUARDS = ("gpu", "source_lock", "pristine_evaluation", "train_within_budget",
          "policy_interface", "finite_actions", "equal_evaluation")


STAGE_NAMES = {"import": "importing the submission", "train": "training"}


class CandidateFailure(Exception):
    """The submission, not the judge, failed. Becomes a scored zero."""


def log_tail(log):
    if not log.is_file():
        return ""
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 2048))
        return stream.read(2048).decode("utf-8", errors="replace")


def execute(repo, suite):
    journal = Journal(os.environ.get("FRONTIER_GO1_EVIDENCE_PATH"))
    subreaper = harden_process()
    recipe = json.loads((HERE / "workloads.json").read_text())
    manifest = load_source_manifest()
    guards = recipe["guards"]
    immutable_runtime = os.geteuid() != 0 and not os.access(sysconfig.get_paths()["purelib"], os.W_OK)
    journal.record("suite_started", suite=suite, runner_uid=os.getuid(), subreaper=subreaper,
                   immutable_runtime=immutable_runtime)
    if guards["require_subreaper"] and not subreaper:
        raise RuntimeError("descendant reaping is unavailable on this host")
    if guards["require_immutable_runtime"] and not immutable_runtime:
        raise RuntimeError("the Python runtime is writable by submitted code")
    admit_trusted_children(manifest)
    config = yaml.safe_load((TASK / "config.yaml").read_text())
    local_deadline = os.environ.get("FRONTIER_VERIFIER_LOCAL_DEADLINE")
    suite_seconds = (config["runtime"].get("final_timeout_seconds", config["runtime"]["timeout_seconds"])
                     if suite == "final" else config["runtime"]["timeout_seconds"])
    runner = ChildRunner(suite_seconds - 90,
                         envelope_deadline=float(local_deadline) if local_deadline is not None else None,
                         on_scientific_start=lambda: journal.record("scientific_started"))

    def interrupted(signum, frame):
        runner.stop()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)

    workloads = recipe["suites"][suite]
    seeds = list(dict.fromkeys(row["train_seed"] for row in workloads))
    metrics = (*recipe["episode_metrics"], *recipe["step_metrics"])
    with tempfile.TemporaryDirectory(prefix="go1-workloads-") as temp:
        scratch = Path(temp)
        scratch.chmod(0o711)
        pristine = scratch / "pristine"
        shutil.copytree(TASK / "harbor/app", pristine)
        admit_pristine(pristine, manifest)
        # No provider or Driver environment is forwarded to scientific code.
        base_env = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")
                    if key in os.environ}
        base_env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MUJOCO_GL="egl",
                        XLA_PYTHON_CLIENT_PREALLOCATE="false")
        if not guards["require_gpu"]:
            base_env["JAX_PLATFORMS"] = "cpu"
        python = [sys.executable, "-I", "-B"]
        pinned = {}

        def phase(name, side, argv, limit, *, stages=(), **fields):
            """Run one child in a fresh directory; classify its failure by side."""
            work = scratch / f"{name}-{side}"
            work.mkdir(mode=0o700)
            env = dict(base_env, HOME=str(work), TMPDIR=str(work))
            log = scratch / f"{name}-{side}.log"
            began = time.monotonic()
            try:
                spent = runner.run([part.replace("{work}", str(work)) for part in argv], cwd=work,
                                   env=env, log=log, limit=limit, stages=stages)
            except (PhaseTimeout, PhaseFailed) as error:
                stage = str(error)
                journal.record("phase_failed", phase=name, side=side, stage=stage,
                               elapsed_seconds=time.monotonic() - began,
                               error_type=type(error).__name__, untrusted_log_tail=log_tail(log),
                               **fields)
                # Before the first marker nothing submitted has run: that stage
                # failing is the judge's problem on either side.
                if side == "candidate" and stage != "startup":
                    what = ("exporting the policy" if (fields["kind"], stage) == ("training", "exit")
                            else STAGE_NAMES.get(stage, fields["kind"]))
                    raise CandidateFailure(
                        f"{what} exceeded its time limit" if isinstance(error, PhaseTimeout)
                        else f"{what} failed") from error
                raise
            except BaseException as error:
                journal.record("phase_failed", phase=name, side=side,
                               elapsed_seconds=time.monotonic() - began,
                               error_type=type(error).__name__, **fields)
                raise
            return work, spent

        candidate_ran = False

        def trusted_inputs():
            # Submitted code may have run since the last check; nothing of it is
            # alive now (ChildRunner.stop), so these bytes cannot change again.
            try:
                admit_trusted_children(manifest)
                admit_pristine(pristine, manifest)
            except ValueError as error:
                journal.record("phase_failed", phase="trusted-inputs",
                               side="candidate" if candidate_ran else "trusted",
                               error_type="TrustedInputChanged", detail=str(error)[:200])
                if candidate_ran:
                    # Only submitted code has run with write access since the
                    # last successful check: the submission is invalid.
                    raise CandidateFailure("submitted code modified judge-owned files") from error
                raise

        def train(seed, side):
            nonlocal candidate_ran
            trusted_inputs()
            tree = pristine if side == "reference" else repo
            candidate_ran = candidate_ran or side == "candidate"
            work, spent = phase(
                f"train-{seed}", side,
                [*python, str(HERE / "train_entry.py"), "--repo", str(tree), "--seed", str(seed),
                 "--budget-seconds", str(recipe["train_budget_seconds"]),
                 "--envs", str(recipe["eval_envs"]),
                 "--observation-size", str(recipe["observation_size"]),
                 "--action-size", str(recipe["action_size"]),
                 "--artifact", "{work}/policy.bin", "--ready-fd", "{ready_fd}"],
                recipe["train_export_seconds"],
                stages=((b"I", recipe["train_startup_seconds"], "startup"),
                        (b"T", recipe["train_import_seconds"], "import"),
                        (b"D", recipe["train_budget_seconds"], "train")),
                kind="training", train_seed=seed)
            source = work / "policy.bin"
            artifact = scratch / f"policy-{seed}-{side}.bin"
            problem = None
            if source.is_symlink() or not source.is_file():
                problem = "training produced no policy artifact"
            elif source.stat().st_size > recipe["max_policy_artifact_bytes"]:
                problem = "policy artifact exceeds its size limit"
            if problem:
                journal.record("phase_failed", phase=f"train-{seed}", side=side,
                               error_type="ArtifactRejected", kind="training", train_seed=seed)
                if side == "candidate":
                    raise CandidateFailure(problem)
                raise RuntimeError(problem)
            shutil.copyfile(source, artifact)
            artifact.chmod(0o444)
            pinned[seed, side] = (artifact, sha256(artifact))
            shutil.rmtree(work, ignore_errors=True)
            journal.record("phase_completed", phase=f"train-{seed}", side=side, kind="training",
                           train_seed=seed, budget_seconds_used=spent["train"],
                           stage_seconds=spent, artifact_bytes=artifact.stat().st_size)

        def evaluate(workload, side):
            trusted_inputs()
            artifact, digest = pinned[workload["train_seed"], side]
            if artifact.is_symlink() or sha256(artifact) != digest:
                journal.record("phase_failed", phase="trusted-inputs", side="candidate",
                               error_type="TrustedInputChanged", detail="policy artifact")
                raise CandidateFailure("submitted code modified judge-owned files")
            condition = recipe["conditions"][workload["condition"]]
            work, spent = phase(
                f"eval-{workload['id']}", side,
                [*python, str(HERE / "evaluate_policy.py"), "--repo", str(pristine),
                 "--artifact", str(artifact), "--env-name", condition["env_name"],
                 "--overrides", json.dumps(condition["overrides"], sort_keys=True),
                 "--eval-seed", str(workload["eval_seed"]),
                 "--envs", str(recipe["eval_envs"]), "--steps", str(recipe["eval_steps"]),
                 "--observation-size", str(recipe["observation_size"]),
                 "--action-size", str(recipe["action_size"]),
                 "--max-artifact-bytes", str(recipe["max_policy_artifact_bytes"]),
                 "--episode-metrics", ",".join(recipe["episode_metrics"]),
                 "--step-metrics", ",".join(recipe["step_metrics"]),
                 "--report", "{work}/report.json"],
                recipe["eval_timeout_seconds"], kind="policy evaluation", workload=workload["id"])
            report = json.loads(report_bytes(work / "report.json"))
            shutil.rmtree(work, ignore_errors=True)
            if "invalid" in report:
                journal.record("phase_failed", phase=f"eval-{workload['id']}", side=side,
                               error_type="PolicyRejected", kind="policy evaluation",
                               workload=workload["id"], report=report)
                if side == "candidate":
                    raise CandidateFailure("policy artifact was rejected by the evaluation")
                raise RuntimeError("trusted reference policy was rejected")
            values = report["metrics"]
            if (set(values) != set(metrics) or report["finite_actions"] is not True
                    or (report["env_name"], report["eval_seed"], report["envs"], report["steps"])
                    != (condition["env_name"], workload["eval_seed"], recipe["eval_envs"],
                        recipe["eval_steps"])
                    or any(not isinstance(v, float) or not math.isfinite(v) for v in values.values())
                    or (guards["require_gpu"] and report["gpu"] is not True)):
                journal.record("phase_failed", phase=f"eval-{workload['id']}", side="trusted",
                               error_type="ReportRejected", kind="policy evaluation",
                               workload=workload["id"])
                raise RuntimeError("trusted evaluation report failed validation")
            journal.record("phase_completed", phase=f"eval-{workload['id']}", side=side,
                           kind="policy evaluation", workload=workload["id"],
                           elapsed_seconds=spent["exit"], report=report)
            return values

        try:
            rows = []
            for seed in seeds:
                # Reference first: a candidate failure then follows a trusted
                # success on the same GPU and is attributable to the candidate.
                for side in ("reference", "candidate"):
                    train(seed, side)
                for workload in (row for row in workloads if row["train_seed"] == seed):
                    row = {"id": workload["id"]}
                    for side in ("reference", "candidate"):
                        row[side] = evaluate(workload, side)
                    row["validity"] = {key: True for key in GUARDS}
                    rows.append(row)
        except CandidateFailure as failure:
            journal.record("suite_completed", suite=suite, candidate_failure=str(failure))
            return {"candidate_failure": str(failure)}
        finally:
            runner.stop()
    order = {row["id"]: index for index, row in enumerate(workloads)}
    rows.sort(key=lambda row: order[row["id"]])
    journal.record("suite_completed", suite=suite)
    return {"workloads": rows, "admission": "exported_policy_pristine_evaluation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--suite", choices=("development", "final"), required=True)
    opts = parser.parse_args()
    try:
        print(json.dumps(execute(opts.repo.resolve(), opts.suite)))
    except SetupDeadlineExpired:
        print(json.dumps({"verifier_failure": "verifier_setup_timeout"}))
        raise SystemExit(75)
