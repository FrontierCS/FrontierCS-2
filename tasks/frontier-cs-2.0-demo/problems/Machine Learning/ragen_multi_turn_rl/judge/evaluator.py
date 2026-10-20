"""Judge for multi-turn trajectory construction in the authentic RAGEN PPO path."""
from __future__ import annotations

import ast
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from candidate_failure import CandidateSourceError, MAX_DIAGNOSTIC_BYTES, public_diagnostic, structure_diagnostic, diagnostic as provenance_diagnostic
from private_execution import ExecutionEvidenceError, SCHEMA as EXECUTION_SCHEMA
from private_execution import collect as collect_execution, install as install_execution

try:
    from submission_patch import apply_submission_patch
except ModuleNotFoundError:
    # Keep local import-by-path smoke tests self-contained; the platform injects
    # the same-named trusted helper beside the evaluator.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from submission_patch import apply_submission_patch

TASK_DIR = Path(__file__).resolve().parent
CONFIG = {
    "submission": {
        "allow_paths": ["ragen/llm_agent/ctx_manager.py", "ragen/llm_agent/solver_extensions/"],
        "max_changed_files": 20,
        "max_patch_bytes": 262144,
    }
}
SPEC_METRICS = ("success_rate", "average_episodic_reward")
REFERENCE_PATCH = TASK_DIR / "reference.patch"
AUTHENTIC_ENTRYPOINT = "train.py"
GROUP_SIZE = 4
GROUP_COUNT = 6
TRAIN_OVERRIDES = (
    "--config-name", "_2_sokoban",
    "trainer.total_training_steps=1",
    "trainer.validation_steps=1",
    "trainer.val_before_train=false",
    "trainer.test_freq=-1",
    "trainer.save_freq=-1",
    "trainer.logger=[console]",
    "trainer.generations_to_log_to_wandb.train=0",
    "trainer.generations_to_log_to_wandb.val=0",
    "es_manager.train.env_groups=6",
    "es_manager.train.group_size=4",
    "es_manager.train.env_configs.n_groups=[6]",
    "actor_rollout_ref.rollout.response_length=96",
    "actor_rollout_ref.rollout.max_model_len=1024",
    "actor_rollout_ref.rollout.max_num_batched_tokens=4096",
    "actor_rollout_ref.rollout.temperature=0.7",
    "actor_rollout_ref.actor.use_ref=false",
    "actor_rollout_ref.actor.use_torch_compile=false",
    "ppo_mini_batch_size=4",
    "micro_batch_size_per_gpu=4",
    "agent_proxy.max_turn=3",
)


class InfrastructureError(RuntimeError):
    """The fixed evaluation environment or exact reference could not run."""


def _require_verifier_cuda() -> dict:
    diagnostic = {
        "uid": os.getuid(),
        "cuda_visible_devices_present": "CUDA_VISIBLE_DEVICES" in os.environ,
        "cuda_visible_devices_nonempty": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
        "nvidia0_exists": None,
        "torch_version": None,
        "cuda_runtime": None,
        "cuda_available": None,
        "device_count": None,
        "devices": [],
    }
    try:
        diagnostic["nvidia0_exists"] = Path("/dev/nvidia0").exists()
    except OSError as error:
        diagnostic["device_node_query_error"] = type(error).__name__
    failure = None
    try:
        import torch

        diagnostic["torch_version"] = str(torch.__version__)[:128]
        diagnostic["cuda_runtime"] = str(torch.version.cuda)[:128]
        diagnostic["cuda_available"] = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        diagnostic["device_count"] = count
        if diagnostic["cuda_available"]:
            torch.cuda.init()
            for index in range(min(count, 4)):
                properties = torch.cuda.get_device_properties(index)
                diagnostic["devices"].append({
                    "index": index, "name": str(properties.name)[:256],
                    "total_memory_bytes": int(properties.total_memory),
                })
        if not diagnostic["cuda_available"]:
            raise RuntimeError("Torch reports CUDA unavailable")
        if count != 1:
            raise RuntimeError(f"expected one CUDA device, observed {count}")
        device = diagnostic["devices"][0]
        # CUDA reports usable memory slightly below the marketed 80 GB capacity.
        if "A100" not in device["name"].upper() or device["total_memory_bytes"] < 75 * 1024**3:
            raise RuntimeError("observed CUDA device does not match the declared A100-80GB")
    except Exception as error:
        failure = error
        diagnostic["query_error"] = {"type": type(error).__name__, "message": str(error)[:512]}
    print("frontier private CUDA runtime: " + json.dumps(diagnostic, sort_keys=True)[:4096],
          file=sys.stderr)
    if failure is not None:
        raise ExecutionEvidenceError({"schema": EXECUTION_SCHEMA, "device": diagnostic,
                                      "candidate": None, "reference": None}) from failure
    return diagnostic


def _zero(reason: str) -> tuple[float, float, str, dict]:
    return 0.0, 0.0, reason, {}


def _source_valid(workdir: Path) -> tuple[bool, str]:
    path = workdir / "ragen/llm_agent/ctx_manager.py"
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return False, "ctx_manager.py is unreadable"
    try:
        compile(source, str(path), "exec")
        tree = ast.parse(source)
    except SyntaxError:
        return False, "ctx_manager.py does not compile"
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "ContextManager"]
    methods = {node.name: node for cls in classes for node in cls.body
               if isinstance(node, ast.FunctionDef)}
    required = [(functions, "get_masks_and_scores")] + [
        (methods, name) for name in ("_parse_response", "_normalize_score_tensor",
                                    "get_lm_inputs", "get_env_inputs", "formulate_rollouts")]
    stub_messages = {
        "get_masks_and_scores": "trajectory mask and reward alignment is part of the task",
        "_parse_response": "response exchange is part of the task",
        "_normalize_score_tensor": "trajectory reward construction is part of the task",
        "get_lm_inputs": "trajectory packing and DataProto assembly is part of the task",
        "get_env_inputs": "observation/action exchange is part of the task",
        "formulate_rollouts": "training trajectory assembly is part of the task",
    }
    for namespace, name in required:
        node = namespace.get(name)
        if node is None:
            return False, "a required public trajectory-construction signature is missing"
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        if (not node.decorator_list and len(body) == 1 and isinstance(body[0], ast.Raise)
                and body[0].cause is None
                and isinstance(body[0].exc, ast.Call)
                and isinstance(body[0].exc.func, ast.Name)
                and body[0].exc.func.id == "NotImplementedError"
                and len(body[0].exc.args) == 1 and not body[0].exc.keywords
                and isinstance(body[0].exc.args[0], ast.Constant)
                and body[0].exc.args[0].value == stub_messages[name]):
            return False, "trajectory construction is still stubbed"
    return True, "ok"


def _install_instrumentation(workdir: Path, output_path: Path, seed: int) -> None:
    install_execution(workdir, TASK_DIR / "private_execution.py", workdir / ".frontier-private-execution")
    es_path = workdir / "ragen/llm_agent/es_manager.py"
    es_path.write_text(es_path.read_text(encoding="utf-8") + f'''\n\n# Frontier evaluator runtime instrumentation.\n_frontier_reset = EnvStateManager.reset\ndef _frontier_seeded_reset(self, seed=None):\n    if self.mode == "train":\n        seed = {seed}\n    return _frontier_reset(self, seed=seed)\nEnvStateManager.reset = _frontier_seeded_reset\n''', encoding="utf-8")
    ctx_path = workdir / "ragen/llm_agent/ctx_manager.py"
    quoted_output = repr(str(output_path))
    ctx_path.write_text(ctx_path.read_text(encoding="utf-8") + f'''\n\n# Frontier evaluator runtime instrumentation.\n_frontier_formulate = ContextManager.formulate_rollouts\ndef _frontier_checked_formulate(self, env_outputs):\n    import hashlib as _hashlib, json as _json, math as _math\n    import copy as _copy\n    trusted_outputs = _copy.deepcopy(env_outputs)\n    result = _frontier_formulate(self, env_outputs)\n    return _frontier_validate_result(self, trusted_outputs, result)\ndef _frontier_validate_result(self, env_outputs, result):\n    import hashlib as _hashlib, json as _json, math as _math\n    required = ("input_ids", "attention_mask", "position_ids", "responses", "loss_mask", "rm_scores", "original_rm_scores")\n    if result.batch is None or any(k not in result.batch.keys() for k in required):\n        raise _frontier_private_module.CandidateStructureError("required_fields")\n    b = result.batch\n    _frontier_private_module.validate_trajectory_layout(\n        {{key: b[key].tolist() for key in required}},\n        result.non_tensor_batch["env_ids"].tolist(),\n        [row["env_id"] for row in env_outputs], self.tokenizer.pad_token_id)\n    n = len(env_outputs)\n    if n == 0 or b["input_ids"].ndim != 2 or b["input_ids"].shape[0] != n:\n        raise _frontier_private_module.CandidateStructureError("row_count")\n    if b["responses"].shape != b["loss_mask"].shape or b["responses"].shape != b["rm_scores"].shape:\n        raise _frontier_private_module.CandidateStructureError("tensor_shapes")\n    if b["responses"].shape[1] + 1 != b["input_ids"].shape[1] or not torch.equal(b["responses"], b["input_ids"][:, 1:]):\n        raise _frontier_private_module.CandidateStructureError("shifted_responses")\n    if any(not torch.isfinite(b[k].float()).all().item() for k in required):\n        raise _frontier_private_module.CandidateStructureError("finite_tensors")\n    attn = b["attention_mask"]\n    if not (((attn == 0) | (attn == 1)).all().item() and (attn[:, 1:] >= attn[:, :-1]).all().item()):\n        raise _frontier_private_module.CandidateStructureError("left_padding")\n    if not b["loss_mask"].bool().any(dim=-1).all().item():\n        raise _frontier_private_module.CandidateStructureError("trainable_tokens")\n    gids = [int(x) for x in result.non_tensor_batch["group_ids"].tolist()]\n    expected_gids = [int(x["group_id"]) for x in env_outputs]\n    if gids != expected_gids:\n        raise _frontier_private_module.CandidateStructureError("group_identity")\n    raw = []\n    groups = {{}}\n    initial_hashes = {{}}\n    for row, env_output in enumerate(env_outputs):\n        rewards = [float(turn.get("reward", 0.0)) for turn in env_output["history"] if "reward" in turn]\n        episode_reward = sum(rewards)\n        raw.append(episode_reward)\n        actions = [a for turn in env_output["history"] for a in turn.get("actions", [])]\n        if any(a not in (1, 2, 3, 4) for a in actions):\n            raise _frontier_private_module.CandidateStructureError("executed_action")\n        gid = int(env_output["group_id"])\n        state = str(env_output["history"][0].get("state", ""))\n        initial_hashes.setdefault(gid, set()).add(_hashlib.sha256(state.encode()).hexdigest())\n        success = float(env_output["metrics"]["SimpleSokoban/success"])\n        groups.setdefault(gid, []).append((success, episode_reward))\n    expected_rewards = torch.tensor(raw, dtype=b["original_rm_scores"].dtype, device=b["original_rm_scores"].device)\n    penalties = torch.tensor([float(row["penalty"]) for row in env_outputs], dtype=expected_rewards.dtype, device=expected_rewards.device)\n    training_rewards = expected_rewards + penalties\n    if not torch.allclose(b["rm_scores"].sum(-1), training_rewards, atol=1e-6, rtol=0):\n        raise _frontier_private_module.CandidateStructureError("training_reward")\n    recorded_rewards = b["original_rm_scores"].sum(-1)\n    if not (torch.allclose(recorded_rewards, expected_rewards, atol=1e-6, rtol=0) or torch.allclose(recorded_rewards, training_rewards, atol=1e-6, rtol=0)):\n        raise _frontier_private_module.CandidateStructureError("original_reward")\n    if b["original_rm_scores"].shape[1] > 1 and torch.count_nonzero(b["original_rm_scores"][:, :-1]).item():\n        raise _frontier_private_module.CandidateStructureError("terminal_reward")\n    if any(len(v) != 1 for v in initial_hashes.values()) or any(len(v) != {GROUP_SIZE} for v in groups.values()):\n        raise _frontier_private_module.CandidateStructureError("initial_state_group")\n    payload = {{"groups": {{str(g): {{"success_rate": sum(x[0] for x in rows)/len(rows), "average_episodic_reward": sum(x[1] for x in rows)/len(rows)}} for g, rows in groups.items()}}, "structural_ok": True, "initial_state_hashes": {{str(g): next(iter(values)) for g, values in initial_hashes.items()}}}}\n    with open({quoted_output}, "w", encoding="utf-8") as handle:\n        _json.dump(payload, handle, sort_keys=True)\n    return result\n_frontier_validate_result = _frontier_private_module.observe_validation(_frontier_validate_result, {str(workdir / '.frontier-private-execution')!r})\nContextManager.formulate_rollouts = _frontier_checked_formulate\n''', encoding="utf-8")


def _run_gpu(workdir: Path, seed: int, private_execution: dict, role: str) -> dict[str, Any]:
    output = workdir / ".frontier_rollouts.json"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["FRONTIER_TRAJECTORY_OUTPUT"] = str(output)
    env["FRONTIER_REPO_ROOT"] = str(workdir)
    env["PYTHONPATH"] = os.pathsep.join((str(workdir), str(workdir / "verl")))
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    model_root = Path(os.environ.get("FRONTIER_MODEL_ROOT", "/opt/frontier/models"))
    model_path = model_root / "qwen2.5-0.5b-instruct"
    if not model_path.is_dir():
        raise InfrastructureError("the baked offline model snapshot is unavailable") from FileNotFoundError(str(model_path))
    pristine = TASK_DIR / "harbor" / "app"
    # The isolated inspector compiles/analyzes bytes only, before submitted code
    # can run. Its result is consumed now and is never read from a child sidecar.
    build_deadline = time.monotonic() + 60
    inspection = subprocess.run(
        [sys.executable, "-I", str(TASK_DIR / "candidate_failure.py"),
         "--inspect", str(workdir), str(pristine)],
        cwd=pristine, capture_output=True, text=True,
        timeout=max(0.0, build_deadline - time.monotonic()),
    )
    if inspection.returncode != 0:
        print((inspection.stdout + inspection.stderr)[-8192:], file=sys.stderr)
        raise InfrastructureError("the source inspector failed") from subprocess.CalledProcessError(
            inspection.returncode, inspection.args, output=inspection.stdout, stderr=inspection.stderr)
    if len(inspection.stdout.encode()) > MAX_DIAGNOSTIC_BYTES:
        raise InfrastructureError("the source inspector returned oversized output")
    try:
        value = json.loads(inspection.stdout)
        diagnostic = None if value is None else public_diagnostic(value)
    except (ValueError, TypeError) as cause:
        raise InfrastructureError("the source inspector returned invalid output") from cause
    _install_instrumentation(workdir, output, seed)
    # Invalid syntax cannot be built. Check the fixed build/dependency closure
    # before returning any source zero, using only the pristine tree in that case.
    build_root = pristine if diagnostic is not None else workdir
    build_env = dict(env)
    build_env["FRONTIER_REPO_ROOT"] = str(build_root)
    build_env["PYTHONPATH"] = os.pathsep.join((str(build_root), str(build_root / "verl")))
    build = subprocess.run(
        ["bash", str(build_root / "frontier-build.sh")], cwd=build_root, env=build_env,
        capture_output=True, text=True, timeout=max(0.0, build_deadline - time.monotonic()),
    )
    if build.returncode != 0:
        print((build.stdout + build.stderr)[-8192:], file=sys.stderr)
        raise InfrastructureError("the submitted tree could not be built") from subprocess.CalledProcessError(
            build.returncode, build.args, output=build.stdout, stderr=build.stderr)
    # A separate import-only process checks the untouched public dependency
    # closure; it cannot populate the candidate process's module cache or RNG.
    # Both children share the original 750-second execution deadline.
    deadline = time.monotonic() + 750
    probe_env = dict(env)
    probe_env["PYTHONPATH"] = os.pathsep.join((str(pristine), str(pristine / "verl")))
    probe = subprocess.run(
        [sys.executable, str(TASK_DIR / "candidate_failure.py"), "--preflight", str(pristine)],
        cwd=pristine, env=probe_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=max(0.0, deadline - time.monotonic()),
    )
    if probe.returncode != 0:
        print(probe.stdout[-8192:], file=sys.stderr)
        raise InfrastructureError("the untouched public dependency imports failed") from subprocess.CalledProcessError(
            probe.returncode, probe.args, output=probe.stdout)
    if diagnostic is not None:
        raise CandidateSourceError(diagnostic)
    command = [sys.executable, str(workdir / AUTHENTIC_ENTRYPOINT),
               *TRAIN_OVERRIDES, f"model_path={model_path}"]
    try:
        proc = subprocess.run(
            command, cwd=workdir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=max(0.0, deadline - time.monotonic()),
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        observed = collect_execution(workdir / ".frontier-private-execution")
        observed.update(valid=False, process_returncode=None)
        observed["issues"].append("training_process_timeout" if isinstance(error, subprocess.TimeoutExpired) else "training_process_launch_failed")
        private_execution[role] = observed
        raise ExecutionEvidenceError(private_execution) from error
    rejection = workdir / ".frontier-private-execution/provenance-rejection.json"
    if rejection.is_file() and rejection.stat().st_size <= 256:
        try:
            record = json.loads(rejection.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            record = None
        if (isinstance(record, dict) and set(record) == {"schema", "reason"}
                and record["schema"] == "ragen.action-provenance/v1"
                and record["reason"] in {"action_provenance", "prompt_provenance", "assistant_provenance"}):
            raise CandidateSourceError(provenance_diagnostic(record["reason"], "ragen/llm_agent/ctx_manager.py", 1))
    observed = collect_execution(workdir / ".frontier-private-execution")
    observed["process_returncode"] = proc.returncode
    private_execution[role] = observed
    if proc.returncode != 0:
        print(proc.stdout[-8192:], file=sys.stderr)
        observed["valid"] = False
        observed["issues"].append("training_process_failed")
        raise ExecutionEvidenceError(private_execution) from subprocess.CalledProcessError(
            proc.returncode, proc.args, output=proc.stdout)
    if observed["valid"] is not True:
        raise ExecutionEvidenceError(private_execution)
    if not output.is_file():
        print(proc.stdout[-8192:], file=sys.stderr)
        observed["valid"] = False
        observed["issues"].append("rollout_evidence_missing")
        raise ExecutionEvidenceError(private_execution) from FileNotFoundError(str(output))
    payload = json.loads(output.read_text(encoding="utf-8"))
    groups = payload.get("groups")
    if (payload.get("structural_ok") is not True or not isinstance(groups, dict)
            or set(groups) != {str(i) for i in range(GROUP_COUNT)}
            or any(not isinstance(row, dict) or any(
                type(row.get(metric)) not in (int, float) or not math.isfinite(row[metric])
                for metric in SPEC_METRICS) for row in groups.values())):
        raise RuntimeError("the PPO run did not produce all structurally valid finite rollout groups")
    return payload


def _score(groups: dict[str, dict[str, float]]) -> float:
    successes = [float(v["success_rate"]) for v in groups.values()]
    rewards = [float(v["average_episodic_reward"]) for v in groups.values()]
    reward_quality = sum(max(0.0, min(1.0, (r + 1.0) / 12.0)) for r in rewards) / len(rewards)
    return max(0.0, min(100.0, 10.0 + 60.0 * sum(successes) / len(successes) + 30.0 * reward_quality))


def _candidate_structure_reason(error):
    evidence = error.private_execution_evidence
    candidate = evidence.get("candidate") if isinstance(evidence, dict) else None
    if not isinstance(candidate, dict) or candidate.get("process_returncode") != 1:
        return None
    diagnostic = candidate.get("diagnostics")
    if not isinstance(diagnostic, dict) or diagnostic.get("dropped_events") != 0:
        return None
    from private_execution import validate_diagnostics
    validate_diagnostics(diagnostic)
    reasons = [row["reason"] for row in diagnostic["events"]
               if row["event"] == "candidate_rejection"]
    return reasons[0] if len(reasons) == 1 else None


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    candidate_result = None
    reference_result = None
    private_execution = {"schema": EXECUTION_SCHEMA, "device": None, "candidate": None, "reference": None}
    try:
        candidate_result = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
        if not candidate_result.ok:
            return _zero(candidate_result.message)
        candidate_dir = Path(candidate_result.workdir)
        valid, reason = _source_valid(candidate_dir)
        if not valid:
            return _zero(reason)
        reference_result = apply_submission_patch(str(REFERENCE_PATCH), TASK_DIR, CONFIG)
        if not reference_result.ok:
            raise InfrastructureError("judge reference could not be prepared") from RuntimeError(reference_result.message)
        reference_dir = Path(reference_result.workdir)
        private_execution["device"] = _require_verifier_cuda()
        seed = 314159 if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else 271828
        candidate_rejection = None
        try:
            candidate = _run_gpu(candidate_dir, seed, private_execution, "candidate")
        except ExecutionEvidenceError as error:
            candidate_rejection = _candidate_structure_reason(error)
            if candidate_rejection is None:
                raise
        except CandidateSourceError as error:
            return 0.0, 0.0, ("candidate structure rejected" if error.diagnostic["category"].endswith("_provenance")
                              else "candidate source preflight rejected before execution"), {
                "candidate_diagnostic": error.diagnostic,
            }
        try:
            reference = _run_gpu(reference_dir, seed, private_execution, "reference")
        except ExecutionEvidenceError:
            raise
        except Exception as cause:
            failure = InfrastructureError("the exact reference execution failed")
            failure.private_execution_evidence = private_execution
            raise failure from cause
        if candidate_rejection is not None:
            return 0.0, 0.0, "candidate training DataProto violates the task contract", {
                "candidate_rejection": True, "beats_reference": 0,
                "candidate_diagnostic": structure_diagnostic(candidate_rejection),
                "private_execution": private_execution,
            }
        if candidate["initial_state_hashes"] != reference["initial_state_hashes"]:
            return _zero("candidate and reference initial states differ")
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        selected = [1, 3, 5] if final else [0, 2, 4]
        candidate_groups = {f"group-{j}": candidate["groups"][str(i)] for j, i in enumerate(selected)}
        reference_groups = {f"group-{j}": reference["groups"][str(i)] for j, i in enumerate(selected)}
        for is_reference, rows in ((False, candidate_groups), (True, reference_groups)):
            if any(not all(math.isfinite(float(row[k])) for k in SPEC_METRICS) for row in rows.values()):
                if is_reference:
                    raise InfrastructureError("the exact reference produced a non-finite rollout metric")
                return _zero("non-finite rollout metric")
            if any(not 0.0 <= float(row["success_rate"]) <= 1.0 for row in rows.values()):
                if is_reference:
                    raise InfrastructureError("the exact reference produced an invalid success rate")
                return _zero("invalid success rate")
        passes = []
        for workload in candidate_groups:
            c, r = candidate_groups[workload], reference_groups[workload]
            passes.append({"workload": workload, "metrics": {
                "success_rate": bool(c["success_rate"] > r["success_rate"]),
                "average_episodic_reward": bool(c["average_episodic_reward"] > r["average_episodic_reward"]),
            }})
        beats = int(all(value for row in passes for value in row["metrics"].values()))
        score = _score(candidate_groups)
        reference_score = _score(reference_groups)
        metrics = {
            "private_execution": private_execution,
            "success_rate": sum(float(v["success_rate"]) for v in candidate_groups.values()) / len(candidate_groups),
            "average_episodic_reward": sum(float(v["average_episodic_reward"]) for v in candidate_groups.values()) / len(candidate_groups),
            "per_workload_candidate_metrics": candidate_groups,
            "initial_state_hashes": candidate["initial_state_hashes"],
            "reference_success_rate": {k: float(v["success_rate"]) for k, v in reference_groups.items()},
            "reference_average_episodic_reward": {k: float(v["average_episodic_reward"]) for k, v in reference_groups.items()},
            "beat_spec_workload_passes": passes,
            "beats_reference": beats,
            "reference_score": reference_score,
            "margin": score - reference_score,
        }
        return score, score, "scored three structurally valid rollout groups", metrics
    finally:
        for result in (candidate_result, reference_result):
            if result is not None and result.workdir:
                shutil.rmtree(getattr(result, "cleanup_root", None) or result.workdir,
                              ignore_errors=True)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(REFERENCE_PATCH)
    print(json.dumps(evaluate(path), sort_keys=True))
