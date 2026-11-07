"""Validate observed workload completion, retaining the standard trust boundary.

These checks detect incomplete or inconsistent runtime evidence. They do not
prove that malicious code in the same process cannot falsify instrumentation;
selected submissions and their execution evidence still require trial audit.
The run's state and checkpoints stay in its isolate instance; files.json
(written there by instance_entry.py) lists their sizes.
"""
import fnmatch
import json
import math
import time
from pathlib import Path


def _regular_nonempty(files, relative):
    return type(files.get(relative)) is int and files[relative] > 0


def validate_execution(tree, output, profile, result, *, deadline):
    output = Path(output)
    if time.monotonic() > deadline or result.get("complete") is not True:
        return False
    if (type(result.get("verified_optimizer_updates")) is not int
            or result["verified_optimizer_updates"] != profile["updates"]):
        return False
    if profile.get('qualification_status') == 'qualified':
        validation = result.get('transition_validation', {})
        if (validation.get('tolerance') != profile.get('transition_tolerance')
                or type(validation.get('checked_calls', {}).get('replayed')) is not int
                or validation['checked_calls']['replayed'] <= 0):
            return False
    for key in ("rollout_batches", "teacher_forwards", "student_forwards",
                "backward_calls", "max_memory_allocated_bytes"):
        if type(result.get(key)) is not int or result[key] <= 0:
            return False
    elapsed = result.get("elapsed_seconds")
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
            or not 0 < elapsed <= profile["max_runner_seconds"]):
        return False
    rewards = result.get("ocr_rewards")
    if (not isinstance(rewards, list) or len(rewards) != len(profile["evaluation"])
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   or not 0 <= v <= 1 for v in rewards)):
        return False
    try:
        files = json.loads((output / "files.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(files, dict) or not (output / "effective_config.json").is_file():
        return False
    if not all(_regular_nonempty(files, path) for path in
               ("state/endpoint.pt", "state/optimizer.bin", "effective_config.json")):
        return False
    adapters = [path for path in files
                if fnmatch.fnmatchcase(path, "checkpoints/checkpoint-*/lora/adapter_model.safetensors")
                and path.count("/") == 3]
    if len(adapters) != 1 or not _regular_nonempty(files, adapters[0]):
        return False
    config = json.loads((output / "effective_config.json").read_text())
    return (config["resolution"] == 512 and config["sample"]["num_steps"] == 10
            and config["sample"]["eval_num_steps"] == 40
            and config["train"]["reward_mode"] == "kl_only")
