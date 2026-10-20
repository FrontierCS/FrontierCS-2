"""Private raw-metric workload execution; this module does not assign scores."""

from __future__ import annotations

import contextlib
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

TASK = Path(__file__).resolve().parent
CHECKPOINT_SHA256 = "fa302d345467ef5927d60f3fd7cdd222e1ee8fda52216ca64323a08e2ce700a8"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_manifest(path):
    value = json.loads(Path(path).read_text())
    if set(value) != {"schema_version", "purpose", "groups", "runs"}:
        raise ValueError("Unknown manifest fields")
    if value["schema_version"] != 1 or value["purpose"] != "provisional_metric_calibration":
        raise ValueError("This loader only supports provisional raw calibration")
    groups, runs = value["groups"], value["runs"]
    if not isinstance(groups, list) or not 1 <= len(groups) <= 4:
        raise ValueError("Expected one to four provisional groups")
    if len(runs) != len(groups) * 6 or len(runs) > 24:
        raise ValueError("Exactly three paired repetitions per group required")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in groups):
        raise ValueError("Invalid environment seeds")
    if len(set(groups)) != len(groups):
        raise ValueError("Duplicate initial-condition groups")
    seen = set()
    pairs = {}
    for index, run in enumerate(runs):
        if set(run) != {"index", "group", "repetition", "env_seed", "diffusion_seed", "steps"}:
            raise ValueError("Unknown run fields")
        if any(type(x) is not int for x in run.values()):
            raise ValueError("Run fields must be integers")
        group, repetition, steps = run["group"], run["repetition"], run["steps"]
        if run["index"] != index or not 0 <= group < len(groups):
            raise ValueError("Run order/group mismatch")
        if repetition not in (0, 1, 2) or steps not in (20, 100):
            raise ValueError("Unprescribed repetition or baseline")
        if run["env_seed"] != groups[group] or not 0 <= run["diffusion_seed"] < 2**32:
            raise ValueError("Run seeds mismatch")
        key = (group, repetition, steps)
        if key in seen:
            raise ValueError("Repeated episode")
        seen.add(key)
        pairs.setdefault((group, repetition), set()).add(run["diffusion_seed"])
    if any(len(seeds) != 1 for seeds in pairs.values()):
        raise ValueError("Reference/baseline RNG seeds must be paired")
    if len({next(iter(seeds)) for seeds in pairs.values()}) != len(pairs):
        raise ValueError("Repetitions must use distinct diffusion seeds")
    return value


def validate_measurements(result):
    rewards = result["rewards"]
    dones = result["original_done"]
    timings = result["prediction_seconds"]
    if not rewards or len(rewards) != len(dones) or len(rewards) > 300:
        raise ValueError("Empty or malformed complete trajectory")
    if any(type(done) is not bool for done in dones):
        raise ValueError("Success must come from original boolean termination")
    if not any(dones) and len(rewards) != 300:
        raise ValueError("Incomplete trajectory without original success")
    if any(not math.isfinite(r) or not 0 <= r <= 1 for r in rewards):
        raise ValueError("Invalid original reward")
    if not timings or len(timings) > 38 or any(not math.isfinite(t) or t <= 0 for t in timings):
        raise ValueError("Invalid prediction timing")
    metrics = result["runner_metrics"]
    if not metrics or any(not math.isfinite(v) for v in metrics.values()):
        raise ValueError("Empty or nonfinite runner metrics")
    expected_key = "test/sim_max_reward_" + str(result["run"]["env_seed"])
    if metrics.get(expected_key) != max(rewards) or metrics.get("test/mean_score") != max(rewards):
        raise ValueError("Original runner quality metric mismatch")
    if result.get("execution_kind", "reference") in {"reference", "baseline"}:
        if result["bridge_model_calls"] != len(timings) * result["run"]["steps"]:
            raise ValueError("Reference did not execute the prescribed denoising schedule")
    elif result.get("execution_kind") == "candidate":
        counts = result.get("prediction_model_calls", [])
        if (len(counts) != len(timings) or any(type(n) is not int or not 1 <= n <= 200 for n in counts)
                or sum(counts) != result["bridge_model_calls"]):
            raise ValueError("Candidate must use the frozen model in every prediction within its budget")
    else:
        raise ValueError("Unknown execution role")
    if result["bridge_worker_terminated"] is not True:
        raise ValueError("Sampler worker did not terminate")
    return {
        "episode_max_reward": max(rewards),
        "original_success": any(dones),
        "prediction_count": len(timings),
        "mean_predict_action_seconds": sum(timings) / len(timings),
    }


def module_provenance(restored):
    sources = {}
    for name, module in list(sys.modules.items()):
        if name == "diffusion_policy" or name.startswith("diffusion_policy."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                relative = path.relative_to(restored.resolve())
                sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sources



def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def execute_episode(run, checkpoint, output, runtime_prefix, *, submission_path=None,
                    sampler_seconds=130, execution_kind=None, expected_ema_sha256=None):
    execution_kind = execution_kind or ("reference" if submission_path is None else "candidate")
    if execution_kind not in {"reference", "baseline", "candidate"} or ((submission_path is not None) != (execution_kind == "candidate")):
        raise ValueError("Unbound execution kind")
    if execution_kind == "baseline" and run["steps"] != 20:
        raise ValueError("Baseline must use natural 20-step DDPM")
    import dill
    import hydra
    import numpy as np
    import torch
    import yaml
    from evaluator import prepared_submission
    from sampler_bridge import SamplerBridge
    from frozen_model_service import FrozenModelService, connect_sampler
    from bridge_protocol import public_configuration
    from bridge_protocol import ProtocolError
    from task_guards import InvalidSubmission

    output = Path(output)
    output.mkdir(exist_ok=False)
    checkpoint = Path(checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Official checkpoint hash mismatch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = yaml.safe_load((TASK / "config.yaml").read_text())
    with contextlib.ExitStack() as stack:
        restored, source_evidence = stack.enter_context(prepared_submission(TASK / "reference.patch", config))
        sampler_source = restored
        candidate_evidence = None
        if submission_path is not None:
            sampler_source, candidate_evidence = stack.enter_context(prepared_submission(submission_path, config))
        sys.path.insert(0, str(restored))
        # Keep namespace-package imports inside the retained source closure.
        import diffusion_policy
        diffusion_policy.__path__ = [str(restored / "diffusion_policy")]
        from diffusion_policy.env.pusht.pusht_env import PushTEnv
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
        cfg = payload["cfg"]
        if not cfg.training.use_ema:
            raise ValueError("Expected the official EMA policy")
        ema_sha256 = tensor_state_sha256(payload["state_dicts"]["ema_model"])
        if expected_ema_sha256 is not None and ema_sha256 != expected_ema_sha256:
            raise ValueError("Frozen EMA identity mismatch")
        policy = hydra.utils.instantiate(cfg.policy)
        policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
        policy.to("cpu").eval().requires_grad_(False)
        if (policy.horizon, policy.n_obs_steps, policy.n_action_steps, policy.num_inference_steps) != (10, 2, 8, 100):
            raise ValueError("Checkpoint horizon/schedule mismatch")
        if not policy.obs_as_cond or policy.kwargs:
            raise ValueError("Unsupported checkpoint interface")
        public = {key: getattr(policy, key) for key in (
            "horizon", "obs_dim", "action_dim", "n_action_steps", "n_obs_steps",
            "num_inference_steps", "obs_as_cond", "pred_action_steps_only")}
        public["scheduler"] = {key: getattr(policy.noise_scheduler.config, key) for key in (
            "num_train_timesteps", "beta_start", "beta_end", "beta_schedule",
            "variance_type", "clip_sample", "prediction_type")}
        public["num_inference_steps"] = run["steps"]
        public_configuration(public)
        original_reset, original_step = PushTEnv.reset, PushTEnv.step

        def observed_reset(env):
            env.calibration_dones = []
            return original_reset(env)

        def observed_step(env, action):
            value = original_step(env, action)
            env.calibration_dones.append(bool(value[2]))
            return value

        PushTEnv.reset, PushTEnv.step = observed_reset, observed_step
        predictions, timings, model_counts = [], [], []
        original_predict = policy.predict_action

        def observed_prediction(obs):
            prior_calls = bridge.model_calls
            before = time.monotonic()
            try:
                value = original_predict(obs)
            except ProtocolError as error:
                if submission_path is not None:
                    raise InvalidSubmission("Candidate sampling violated the isolated tensor protocol") from error
                raise
            timings.append(time.monotonic() - before)
            model_counts.append(bridge.model_calls - prior_calls)
            if not 1 <= model_counts[-1] <= 200:
                if submission_path is not None:
                    raise InvalidSubmission("Prediction exceeded its frozen-model use budget")
                raise ValueError("Reference exceeded its frozen-model use budget")
            predictions.append({key: FrozenModelService.pack(tensor) for key, tensor in value.items()})
            return value

        policy.predict_action = observed_prediction
        horizon = policy.n_action_steps if policy.pred_action_steps_only else policy.horizon
        service = FrozenModelService(policy.model, torch.Generator().manual_seed(run["diffusion_seed"]),
            horizon=horizon, action_dim=policy.action_dim, obs_dim=policy.obs_dim,
            n_obs_steps=policy.n_obs_steps, num_train_timesteps=100, max_batch=1)
        bridge = stack.enter_context(SamplerBridge(sampler_source, TASK / "sampler_worker.py",
            lifetime_seconds=sampler_seconds, max_model_calls=38 * 200,
            max_random_calls=38 * 1000, model_forward=service.forward,
            random_normal=service.normal, runtime={"prefix": str(runtime_prefix),
                                                  "base": sys.base_prefix, "version": "3.10"}))
        bridge.configure(public)
        connect_sampler(policy, bridge)
        random.seed(run["diffusion_seed"])
        np.random.seed(run["diffusion_seed"])
        torch.manual_seed(run["diffusion_seed"])
        runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output),
            n_train=0, n_train_vis=0, n_test=1, n_test_vis=0, n_envs=1,
            test_start_seed=run["env_seed"], max_steps=300)
        try:
            module_provenance(restored)
            runner_started = time.monotonic()
            metrics = runner.run(policy)
            runner_seconds = time.monotonic() - runner_started
            rewards = runner.env.call("get_attr", "reward")[0]
            dones = runner.env.call("get_attr", "calibration_dones")[0]
            provenance = module_provenance(restored)
        finally:
            runner.env.close()
    result = {
        "run": run, "predictions": predictions, "prediction_seconds": timings,
        "execution_kind": execution_kind, "ema_sha256": ema_sha256,
        "prediction_model_calls": model_counts,
        "rewards": [float(r) for r in rewards], "original_done": [bool(d) for d in dones],
        "runner_metrics": {key: float(value) for key, value in metrics.items()},
        "runner_seconds": runner_seconds, "checkpoint_sha256": CHECKPOINT_SHA256,
        "public_configuration": public, "source_evidence": asdict(source_evidence),
        "candidate_source_evidence": None if candidate_evidence is None else asdict(candidate_evidence),
        "trusted_module_sources": provenance,
        "bridge_model_calls": bridge.model_calls, "bridge_random_calls": bridge.random_calls,
        "bridge_lifetime_seconds": bridge.finished_at - bridge.started_at,
        "bridge_worker_terminated": bridge.process.poll() is not None,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "self_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        "children_cpu_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
    }
    write_json(output / "trajectory.json", result)
    result["raw_metrics"] = validate_measurements(result)
    write_json(output / "trajectory.json", result)
    return result
