"""Judge of ragen_multi_turn_rl: multi-turn RAGEN PPO trajectory batches (#757 scoring note).

Runs on the CPU control image. The PPO runs of the submission and of the exact reference go to
two [test_sandboxes.candidate] runners (one A100-80GB each, the Agent GPU image) at the same
time; each machine receives only its instrumented tree, the starter and the observation hooks
the PPO process loads, and returns the device diagnostic, the observation records and the
rollout record, judged here as before, except that each group's success_rate and
average_episodic_reward are recomputed here (judge/replay.py replays the recorded episodes in
the starter's SimpleSokoban from the judge's seed); those are the scoring inputs.

Score (raw_score = score_unbounded, 0..100): over the three graded rollout groups w, with s_w the
success fraction and r_w the average undiscounted episodic reward,
    score = clip(10 + 60 * mean_w(s_w) + 30 * mean_w(clip((r_w + 1) / 12, 0, 1)), 0, 100).
Scored zeros: a compile/stub, source-preflight or provenance rejection (before the reference
run), and a DataProto-contract rejection, a crash or 750 s overrun of the submission's PPO run,
differing initial states or an invalid rollout metric (once the exact reference run succeeded);
a missing public signature is rejected by check_tree.

Metrics, per rollout group (means over groups are the train feedback):
- success_rate: fraction of trajectories ending with every box on a target; higher is better;
  passes when strictly above the reference's.
- average_episodic_reward: mean undiscounted sum of genuine environment rewards; higher is
  better; passes when strictly above the reference's.

beats_reference = 1 iff both metrics pass on every graded group.
The archived beat spec is docs/task-authoring/ragen_multi_turn_rl/provenance/beat_spec.json.
"""
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
from pathlib import Path, PurePosixPath
from typing import Any

from source_preflight import CandidateSourceError, MAX_DIAGNOSTIC_BYTES, public_diagnostic, diagnostic as provenance_diagnostic
from private_execution import ExecutionEvidenceError, SCHEMA as EXECUTION_SCHEMA
from private_execution import collect as collect_execution, install as install_execution
from fcs2_verifier_gpu import SandboxResult, SandboxRun, SandboxStep, run_sandboxes, sandbox_workdir

TASK_DIR = Path(__file__).resolve().parent
PRISTINE = TASK_DIR / "pristine"  # the starter repo; the shared verifier overlays the submission
SPEC_METRICS = ("success_rate", "average_episodic_reward")
REFERENCE_PATCH = TASK_DIR / "reference.patch"
AUTHENTIC_ENTRYPOINT = "train.py"
# The PPO runs on [test_sandboxes.candidate] runners of the Agent GPU image: its interpreter and
# baked offline model snapshot. What comes back: the device diagnostic, the observation records
# and the validated rollout metrics.
PYTHON = "/opt/frontier/runtime-python"
MODEL_PATH = "/opt/frontier/models/qwen2.5-0.5b-instruct"
OUTPUTS = ("out", "work/.frontier-private-execution", "work/.frontier_rollouts.json")
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


# The device check, run as the first step of each sandbox (argv[1]: its JSON output, argv[2]:
# the model snapshot). It exits 1 when the runner does not have the one declared A100-80GB and
# 2 when the baked offline model snapshot is missing.
CUDA_CHECK = r"""
import json, os, sys
from pathlib import Path
diagnostic = {"uid": os.getuid(),
              "cuda_visible_devices_present": "CUDA_VISIBLE_DEVICES" in os.environ,
              "cuda_visible_devices_nonempty": bool(os.environ.get("CUDA_VISIBLE_DEVICES")),
              "nvidia0_exists": None, "torch_version": None, "cuda_runtime": None,
              "cuda_available": None, "device_count": None, "devices": []}
try:
    diagnostic["nvidia0_exists"] = Path("/dev/nvidia0").exists()
except OSError as error:
    diagnostic["device_node_query_error"] = type(error).__name__
failed = False
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
            diagnostic["devices"].append({"index": index, "name": str(properties.name)[:256],
                                          "total_memory_bytes": int(properties.total_memory)})
    if not diagnostic["cuda_available"]:
        raise RuntimeError("Torch reports CUDA unavailable")
    if count != 1:
        raise RuntimeError(f"expected one CUDA device, observed {count}")
    device = diagnostic["devices"][0]
    # CUDA reports usable memory slightly below the marketed 80 GB capacity.
    if "A100" not in device["name"].upper() or device["total_memory_bytes"] < 75 * 1024**3:
        raise RuntimeError("observed CUDA device does not match the declared A100-80GB")
except Exception as error:
    failed = True
    diagnostic["query_error"] = {"type": type(error).__name__, "message": str(error)[:512]}
diagnostic["model_snapshot"] = Path(sys.argv[2]).is_dir()
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text(json.dumps(diagnostic, sort_keys=True)[:4096])
sys.exit(1 if failed else 0 if diagnostic["model_snapshot"] else 2)
"""


def _zero(reason: str) -> tuple[float, float, str, dict]:
    return 0.0, 0.0, reason, {}


REQUIRED_SIGNATURES = ("get_masks_and_scores", "_parse_response", "_normalize_score_tensor",
                       "get_lm_inputs", "get_env_inputs", "formulate_rollouts")


def _public_signatures(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """get_masks_and_scores at module level and the ContextManager methods, by name."""
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "ContextManager"]
    methods = {node.name: node for cls in classes for node in cls.body
               if isinstance(node, ast.FunctionDef)}
    found = {name: functions[name] for name in REQUIRED_SIGNATURES[:1] if name in functions}
    found.update({name: methods[name] for name in REQUIRED_SIGNATURES[1:] if name in methods})
    return found


def check_tree(tree: str, pristine: str, changes: list) -> tuple[str, list[str]] | None:
    """Shared-verifier hook: the public trajectory-construction signatures of ctx_manager.py
    are part of the file that must stay. Source that does not parse is left to evaluate(),
    which scores it zero as before."""
    path = Path(tree) / "ragen/llm_agent/ctx_manager.py"
    try:
        parsed = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return None
    if set(_public_signatures(parsed)) != set(REQUIRED_SIGNATURES):
        return "out_of_scope", ["ragen/llm_agent/ctx_manager.py"]
    return None


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


def _install_instrumentation(workdir: Path, remote: PurePosixPath, observer: PurePosixPath,
                             seed: int) -> None:
    """Instrument the tree at `workdir` (here) for its run at `remote` in the sandbox, with the
    observation hooks at `observer` there."""
    (workdir / ".frontier-private-execution").mkdir(mode=0o700)
    install_execution(workdir, observer / "private_execution.py",
                      remote / ".frontier-private-execution")
    es_path = workdir / "ragen/llm_agent/es_manager.py"
    es_path.write_text(es_path.read_text(encoding="utf-8") + f'''\n\n# Frontier evaluator runtime instrumentation.\n_frontier_reset = EnvStateManager.reset\ndef _frontier_seeded_reset(self, seed=None):\n    if self.mode == "train":\n        seed = {seed}\n    return _frontier_reset(self, seed=seed)\nEnvStateManager.reset = _frontier_seeded_reset\n''', encoding="utf-8")
    ctx_path = workdir / "ragen/llm_agent/ctx_manager.py"
    quoted_output = repr(str(remote / ".frontier_rollouts.json"))
    ctx_path.write_text(ctx_path.read_text(encoding="utf-8") + f'''\n\n# Frontier evaluator runtime instrumentation.\n_frontier_formulate = ContextManager.formulate_rollouts\ndef _frontier_checked_formulate(self, env_outputs):\n    import hashlib as _hashlib, json as _json, math as _math\n    import copy as _copy\n    trusted_outputs = _copy.deepcopy(env_outputs)\n    result = _frontier_formulate(self, env_outputs)\n    return _frontier_validate_result(self, trusted_outputs, result)\ndef _frontier_validate_result(self, env_outputs, result):\n    import hashlib as _hashlib, json as _json, math as _math\n    required = ("input_ids", "attention_mask", "position_ids", "responses", "loss_mask", "rm_scores", "original_rm_scores")\n    if result.batch is None or any(k not in result.batch.keys() for k in required):\n        raise _frontier_private_module.CandidateStructureError("required_fields")\n    b = result.batch\n    _frontier_private_module.validate_trajectory_layout(\n        {{key: b[key].tolist() for key in required}},\n        result.non_tensor_batch["env_ids"].tolist(),\n        [row["env_id"] for row in env_outputs], self.tokenizer.pad_token_id)\n    n = len(env_outputs)\n    if n == 0 or b["input_ids"].ndim != 2 or b["input_ids"].shape[0] != n:\n        raise _frontier_private_module.CandidateStructureError("row_count")\n    if b["responses"].shape != b["loss_mask"].shape or b["responses"].shape != b["rm_scores"].shape:\n        raise _frontier_private_module.CandidateStructureError("tensor_shapes")\n    if b["responses"].shape[1] + 1 != b["input_ids"].shape[1] or not torch.equal(b["responses"], b["input_ids"][:, 1:]):\n        raise _frontier_private_module.CandidateStructureError("shifted_responses")\n    if any(not torch.isfinite(b[k].float()).all().item() for k in required):\n        raise _frontier_private_module.CandidateStructureError("finite_tensors")\n    attn = b["attention_mask"]\n    if not (((attn == 0) | (attn == 1)).all().item() and (attn[:, 1:] >= attn[:, :-1]).all().item()):\n        raise _frontier_private_module.CandidateStructureError("left_padding")\n    if not b["loss_mask"].bool().any(dim=-1).all().item():\n        raise _frontier_private_module.CandidateStructureError("trainable_tokens")\n    gids = [int(x) for x in result.non_tensor_batch["group_ids"].tolist()]\n    expected_gids = [int(x["group_id"]) for x in env_outputs]\n    if gids != expected_gids:\n        raise _frontier_private_module.CandidateStructureError("group_identity")\n    raw = []\n    groups = {{}}\n    initial_hashes = {{}}\n    episodes = []\n    for row, env_output in enumerate(env_outputs):\n        episodes.append({{"env_id": int(env_output["env_id"]), "group_id": int(env_output["group_id"]), "initial_state": str(env_output["history"][0].get("state", "")), "turns": [[int(a) for a in turn["actions"]] for turn in env_output["history"] if "actions" in turn]}})\n        rewards = [float(turn.get("reward", 0.0)) for turn in env_output["history"] if "reward" in turn]\n        episode_reward = sum(rewards)\n        raw.append(episode_reward)\n        actions = [a for turn in env_output["history"] for a in turn.get("actions", [])]\n        if any(a not in (1, 2, 3, 4) for a in actions):\n            raise _frontier_private_module.CandidateStructureError("executed_action")\n        gid = int(env_output["group_id"])\n        state = str(env_output["history"][0].get("state", ""))\n        initial_hashes.setdefault(gid, set()).add(_hashlib.sha256(state.encode()).hexdigest())\n        success = float(env_output["metrics"]["SimpleSokoban/success"])\n        groups.setdefault(gid, []).append((success, episode_reward))\n    expected_rewards = torch.tensor(raw, dtype=b["original_rm_scores"].dtype, device=b["original_rm_scores"].device)\n    penalties = torch.tensor([float(row["penalty"]) for row in env_outputs], dtype=expected_rewards.dtype, device=expected_rewards.device)\n    training_rewards = expected_rewards + penalties\n    if not torch.allclose(b["rm_scores"].sum(-1), training_rewards, atol=1e-6, rtol=0):\n        raise _frontier_private_module.CandidateStructureError("training_reward")\n    recorded_rewards = b["original_rm_scores"].sum(-1)\n    if not (torch.allclose(recorded_rewards, expected_rewards, atol=1e-6, rtol=0) or torch.allclose(recorded_rewards, training_rewards, atol=1e-6, rtol=0)):\n        raise _frontier_private_module.CandidateStructureError("original_reward")\n    if b["original_rm_scores"].shape[1] > 1 and torch.count_nonzero(b["original_rm_scores"][:, :-1]).item():\n        raise _frontier_private_module.CandidateStructureError("terminal_reward")\n    if any(len(v) != 1 for v in initial_hashes.values()) or any(len(v) != {GROUP_SIZE} for v in groups.values()):\n        raise _frontier_private_module.CandidateStructureError("initial_state_group")\n    payload = {{"groups": {{str(g): {{"success_rate": sum(x[0] for x in rows)/len(rows), "average_episodic_reward": sum(x[1] for x in rows)/len(rows)}} for g, rows in groups.items()}}, "structural_ok": True, "initial_state_hashes": {{str(g): next(iter(values)) for g, values in initial_hashes.items()}}, "episodes": episodes}}\n    with open({quoted_output}, "w", encoding="utf-8") as handle:\n        _json.dump(payload, handle, sort_keys=True)\n    return result\n_frontier_validate_result = _frontier_private_module.observe_validation(_frontier_validate_result, {str(remote / '.frontier-private-execution')!r})\nContextManager.formulate_rollouts = _frontier_checked_formulate\n''', encoding="utf-8")


def _inspect(workdir: Path) -> dict | None:
    """The isolated source inspector's public diagnostic of `workdir` (None: no rejection). It
    compiles/analyzes bytes only, here on the control image, before submitted code can run."""
    inspection = subprocess.run(
        [sys.executable, "-I", str(TASK_DIR / "source_preflight.py"),
         "--inspect", str(workdir), str(PRISTINE)],
        cwd=PRISTINE, capture_output=True, text=True, timeout=60,
    )
    if inspection.returncode != 0:
        print((inspection.stdout + inspection.stderr)[-8192:], file=sys.stderr)
        raise InfrastructureError("the source inspector failed") from subprocess.CalledProcessError(
            inspection.returncode, inspection.args, output=inspection.stdout, stderr=inspection.stderr)
    if len(inspection.stdout.encode()) > MAX_DIAGNOSTIC_BYTES:
        raise InfrastructureError("the source inspector returned oversized output")
    try:
        value = json.loads(inspection.stdout)
        return None if value is None else public_diagnostic(value)
    except (ValueError, TypeError) as cause:
        raise InfrastructureError("the source inspector returned invalid output") from cause


def _sandbox_run(tree: Path, files: Path, seed: int, root: PurePosixPath, *,
                 train: bool) -> SandboxRun:
    """One [test_sandboxes.candidate] run: files/work (the instrumented `tree`), files/pristine
    and files/observer (the observation hooks the PPO process loads), then the steps: device
    check, build (60 s), the untouched public imports (120 s) and, with `train`, the PPO run
    (750 s). Only these files reach the runner."""
    shutil.copytree(tree, files / "work", symlinks=True)
    shutil.copytree(PRISTINE, files / "pristine", symlinks=True)
    (files / "observer").mkdir()
    for name in ("private_execution.py", "action_provenance.py"):
        shutil.copy2(TASK_DIR / name, files / "observer" / name)
    for path in (files, *files.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    work, pristine = root / "work", root / "pristine"
    steps = [
        SandboxStep([PYTHON, "-c", CUDA_CHECK, str(root / "out/device.json"), MODEL_PATH], 60),
        SandboxStep(["/bin/bash", str(work / "frontier-build.sh")], 60, {
            "FRONTIER_REPO_ROOT": str(work), "PYTHONPATH": f"{work}:{work}/verl"}),
        SandboxStep([PYTHON, "-c", "import importlib; importlib.import_module("
                     "'ragen.llm_agent.ctx_manager')"], 120,
                    {"PYTHONPATH": f"{pristine}:{pristine}/verl"}),
    ]
    if train:
        _install_instrumentation(files / "work", work, root / "observer", seed)
        steps.append(SandboxStep(
            ["/bin/sh", "-c", f'cd {work} && exec "$0" "$@"', PYTHON,
             str(work / AUTHENTIC_ENTRYPOINT), *TRAIN_OVERRIDES, f"model_path={MODEL_PATH}"],
            750, {"PYTHONHASHSEED": str(seed),
                  "FRONTIER_TRAJECTORY_OUTPUT": str(work / ".frontier_rollouts.json"),
                  "FRONTIER_REPO_ROOT": str(work), "PYTHONPATH": f"{work}:{work}/verl",
                  "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
                  "TMPDIR": "/workspace"}))
    return SandboxRun("candidate", files, steps, list(OUTPUTS))


def _log(result: SandboxResult, step: int) -> None:
    if result.logs[step]:
        print(result.logs[step].decode("utf-8", errors="replace"), file=sys.stderr)


def _device(result: SandboxResult) -> dict:
    """The device step's diagnostic; a runner without the declared GPU is invalid execution."""
    step = result.steps[0]
    path = None if result.outputs is None else result.outputs / "out/device.json"
    diagnostic = None
    if path is not None and path.is_file() and path.stat().st_size <= 4096:
        diagnostic = json.loads(path.read_text(encoding="utf-8"))
    print("frontier private CUDA runtime: " + json.dumps(diagnostic, sort_keys=True)[:4096],
          file=sys.stderr)
    if step.timed_out or diagnostic is None:
        raise InfrastructureError("the sandbox device check did not complete")
    if step.exit_code == 2:
        raise InfrastructureError("the baked offline model snapshot is unavailable")
    if step.exit_code != 0:
        raise ExecutionEvidenceError({"schema": EXECUTION_SCHEMA, "device": diagnostic,
                                      "candidate": None, "reference": None})
    return diagnostic


def _lost_from(result: SandboxResult) -> int | None:
    """The step whose session the machine lost, if any."""
    return next((index for index, step in enumerate(result.steps)
                 if step is not None and step.lost_session), None)


def _machine(result: SandboxResult, role: str) -> dict | None:
    """The fixed environment of one runner: device, build and public imports. Submitted code
    runs from the build step on (it runs the tree's frontier-build.sh and installs the tree):
    a lost session there is the submission's (judged by _rollouts), before it infrastructure,
    and on the reference machine always infrastructure. The device diagnostic, or None when
    nothing came back (a lost session or an output-cap overrun, judged by _rollouts)."""
    lost = _lost_from(result)
    if lost is not None and (lost == 0 or role == "reference"):
        raise InfrastructureError("a sandbox lost its session")
    device = None if result.outputs is None else _device(result)
    if lost is None:
        _prepared(result)
    return device


def _prepared(result: SandboxResult) -> None:
    """Build and public-import steps: the fixed environment, never the submission's fault."""
    for index, message in ((1, "the submitted tree could not be built"),
                           (2, "the untouched public dependency imports failed")):
        step = result.steps[index]
        if step is None or step.timed_out or step.exit_code != 0:
            _log(result, index)
            raise InfrastructureError(message)


def _rollouts(result: SandboxResult, private_execution: dict, role: str,
              seed: int) -> dict[str, Any]:
    """The PPO step's outcome from its returned evidence, judged as the in-runner judge did;
    the per-group metrics are recomputed here by replaying the executed episodes."""
    if _lost_from(result) is not None:
        # The submission's machine lost a session from the build step on: nothing came back
        # (owner 2026-11-05: the submission's scored zero).
        private_execution[role] = {"schema": EXECUTION_SCHEMA, "valid": False,
                                   "process_returncode": None,
                                   "issues": ["training_session_lost"]}
        raise ExecutionEvidenceError(private_execution)
    step = result.steps[3]
    if result.outputs_exceeded:
        # The sandbox returned more than the output cap: no evidence to judge.
        private_execution[role] = {"schema": EXECUTION_SCHEMA, "valid": False,
                                   "process_returncode": step.exit_code,
                                   "issues": ["evidence_exceeded_output_cap"]}
        raise ExecutionEvidenceError(private_execution)
    private = result.outputs / "work/.frontier-private-execution"
    if step.timed_out:
        observed = collect_execution(private)
        observed.update(valid=False, process_returncode=None)
        observed["issues"].append("training_process_timeout")
        private_execution[role] = observed
        raise ExecutionEvidenceError(private_execution)
    rejection = private / "provenance-rejection.json"
    if rejection.is_file() and rejection.stat().st_size <= 256:
        try:
            record = json.loads(rejection.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            record = None
        if (isinstance(record, dict) and set(record) == {"schema", "reason"}
                and record["schema"] == "ragen.action-provenance/v1"
                and record["reason"] in {"action_provenance", "prompt_provenance", "assistant_provenance"}):
            raise CandidateSourceError(provenance_diagnostic(record["reason"], "ragen/llm_agent/ctx_manager.py", 1))
    observed = collect_execution(private)
    observed["process_returncode"] = step.exit_code
    private_execution[role] = observed
    if step.exit_code != 0:
        _log(result, 3)
        observed["valid"] = False
        observed["issues"].append("training_process_failed")
        raise ExecutionEvidenceError(private_execution)
    if observed["valid"] is not True:
        raise ExecutionEvidenceError(private_execution)
    output = result.outputs / "work/.frontier_rollouts.json"
    if not output.is_file():
        _log(result, 3)
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
    replayed = _replay(payload, seed)
    if replayed["mismatch"] is not None or set(replayed["groups"]) != set(groups):
        # The returned episodes are not what the environment does from the judge's seed.
        observed["valid"] = False
        observed["issues"].append("rollout_replay_mismatch")
        raise ExecutionEvidenceError(private_execution)
    # Scoring inputs: the replayed metrics, never the numbers the run reported.
    payload["reported_groups"], payload["groups"] = groups, replayed["groups"]
    return payload


def _replay(payload: dict, seed: int) -> dict:
    """judge/replay.py on this image: the episodes stepped from the judge's seed."""
    replay = subprocess.run(
        [sys.executable, "-I", str(TASK_DIR / "replay.py"), str(PRISTINE), str(seed),
         str(GROUP_SIZE)],
        input=json.dumps({"episodes": payload["episodes"]}), capture_output=True, text=True,
        timeout=300, env={**os.environ, "PYTHONHASHSEED": str(seed)})
    if replay.returncode != 0:
        print(replay.stderr[-8192:], file=sys.stderr)
        raise InfrastructureError("the rollout replay failed")
    return json.loads(replay.stdout)


def _final() -> bool:
    return os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"


def _reference_tree(scratch: Path) -> Path:
    """The pristine starter with judge/reference.patch applied (trusted, judge-owned)."""
    tree = scratch / "reference"
    shutil.copytree(PRISTINE, tree, symlinks=True)
    for path in (tree, *tree.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(scratch.resolve())}
    applied = subprocess.run(["git", "apply", "--whitespace=nowarn", str(REFERENCE_PATCH)],
                             cwd=tree, capture_output=True, text=True, timeout=120, env=env)
    if applied.returncode != 0:
        raise InfrastructureError("judge reference could not be prepared") from RuntimeError(
            applied.stderr[-2000:])
    return tree


def _score(groups: dict[str, dict[str, float]]) -> float:
    successes = [float(v["success_rate"]) for v in groups.values()]
    rewards = [float(v["average_episodic_reward"]) for v in groups.values()]
    reward_quality = sum(max(0.0, min(1.0, (r + 1.0) / 12.0)) for r in rewards) / len(rewards)
    return max(0.0, min(100.0, 10.0 + 60.0 * sum(successes) / len(successes) + 30.0 * reward_quality))


def _candidate_run_failure(private_execution):
    """The submission's PPO process timed out or exited non-zero (without a typed rejection)."""
    candidate = private_execution.get("candidate")
    issues = candidate.get("issues", []) if isinstance(candidate, dict) else []
    if "training_process_timeout" in issues:
        return "the submission's PPO run exceeded its 750-second deadline"
    if "training_process_failed" in issues:
        return "the submission's PPO run failed"
    if "training_session_lost" in issues:
        # Owner 2026-11-05: its own machine lost the PPO command's session.
        return "the submission's PPO run lost its session"
    if "evidence_exceeded_output_cap" in issues:
        return "the submission's PPO run returned more than the output cap"
    return None


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


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    """Shared-verifier contract: `tree` is the pristine starter with only the submission's
    allowed files overlaid (edit scope and check_tree already passed)."""
    reference_scratch = None
    private_execution = {"schema": EXECUTION_SCHEMA, "device": None, "candidate": None, "reference": None}
    try:
        candidate_dir = Path(tree)
        valid, reason = _source_valid(candidate_dir)
        if not valid:
            return _zero(reason)
        reference_scratch = Path(tempfile.mkdtemp(prefix="ragen-reference-"))
        reference_dir = _reference_tree(reference_scratch)
        seed = 314159 if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else 271828
        root = PurePosixPath(sandbox_workdir("candidate"))
        diagnostic = _inspect(candidate_dir)
        if _inspect(reference_dir) is not None:
            raise InfrastructureError("the exact reference failed the source inspector")
        runs = []
        for name, tree, train in (("candidate", PRISTINE if diagnostic else candidate_dir,
                                   diagnostic is None),
                                  ("reference", reference_dir, True)):
            if name == "reference" and diagnostic is not None:
                continue
            files = reference_scratch / f"files-{name}"
            files.mkdir()
            runs.append(_sandbox_run(tree, files, seed, root, train=train))
        outputs = reference_scratch / "outputs"
        outputs.mkdir()
        # Candidate and reference on two runners of the same shape, at the same time.
        results = run_sandboxes(runs, into=outputs)
        private_execution["device"] = {name: _machine(result, name)
                                       for name, result in zip(("candidate", "reference"), results)}
        candidate_rejection = None
        candidate_run_failure = None
        try:
            if diagnostic is not None:
                raise CandidateSourceError(diagnostic)
            candidate = _rollouts(results[0], private_execution, "candidate", seed)
        except ExecutionEvidenceError as error:
            candidate_rejection = _candidate_structure_reason(error)
            if candidate_rejection is None:
                candidate_run_failure = _candidate_run_failure(private_execution)
                if candidate_run_failure is None:
                    raise
        except CandidateSourceError as error:
            # A scored zero with a fixed message; the diagnostic stays private (final only).
            return 0.0, 0.0, ("candidate structure rejected" if error.diagnostic["category"].endswith("_provenance")
                              else "candidate source preflight rejected before execution"), (
                {"source_diagnostic": error.diagnostic} if _final() else {})
        try:
            reference = _rollouts(results[1], private_execution, "reference", seed)
        except ExecutionEvidenceError:
            raise
        except Exception as cause:
            failure = InfrastructureError("the exact reference execution failed")
            failure.private_execution_evidence = private_execution
            raise failure from cause
        if candidate_run_failure is not None:
            # The submission's own PPO process crashed or overran its deadline while the exact
            # reference ran: a scored zero whose message names it (#757).
            return 0.0, 0.0, candidate_run_failure, (
                {"private_execution": private_execution} if _final() else {})
        if candidate_rejection is not None:
            # A scored zero with a fixed message. Train metrics stay empty (the shared verifier
            # admits only beat-spec metrics there); final keeps the private reason and evidence.
            return 0.0, 0.0, "candidate training DataProto violates the task contract", (
                {"candidate_rejection": candidate_rejection, "private_execution": private_execution}
                if _final() else {})
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
        if reference_scratch is not None:
            shutil.rmtree(reference_scratch, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), sort_keys=True))
