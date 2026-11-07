"""Control-side recomputation of the rollout metrics (owner 2026-11-05).

The PPO run on the submission's machine returns, per episode, its environment, its initial state
and the actions the environment executed in each turn. Here, on the judge's control image, every
SimpleSokoban environment is reset from the seed the judge set (the train reset seed plus the
episode's group: es_manager's _expand_seed) with the starter's own environment code, the
recorded actions are stepped under the starter's limits (the turn stops at done, at most
max_actions_per_traj actions per episode), and each episode's undiscounted reward and success
are recomputed. Those, not the numbers the run reported, are the scoring inputs.

  replay.py PRISTINE SEED GROUP_SIZE < payload.json > {"groups", "initial_states", "mismatch"}

Run with PYTHONHASHSEED=SEED, as the PPO process ran. `mismatch` names the first episode whose
record the environment does not reproduce (its initial state, or an action after the episode
ended or past its budget); it is None when every episode replays.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

TAG = "SimpleSokoban"


def _environment(pristine: Path):
    """The starter's SokobanEnv and its configuration, without the env registry's other
    environments (ragen.env imports them all)."""
    sys.path[:0] = [str(pristine)]
    for name, path in (("ragen", pristine / "ragen"), ("ragen.env", pristine / "ragen/env")):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    env = importlib.import_module("ragen.env.sokoban.env")
    from omegaconf import OmegaConf

    # _2_sokoban.yaml takes custom_envs from base.yaml's defaults: config/envs.yaml.
    custom = OmegaConf.load(pristine / "config/envs.yaml").custom_envs[TAG]
    config = env.SokobanEnvConfig(**OmegaConf.to_container(custom.env_config))
    return env.SokobanEnv, config, int(custom.max_actions_per_traj)


def _well_formed(episode) -> bool:
    return (isinstance(episode, dict)
            and set(episode) == {"env_id", "group_id", "initial_state", "turns"}
            and type(episode["env_id"]) is int and episode["env_id"] >= 0
            and type(episode["group_id"]) is int and isinstance(episode["initial_state"], str)
            and isinstance(episode["turns"], list)
            and all(isinstance(turn, list) and all(type(a) is int for a in turn)
                    for turn in episode["turns"]))


def replay(payload: dict, pristine: Path, seed: int, group_size: int) -> dict:
    environment, config, budget = _environment(pristine)
    if not isinstance(payload.get("episodes"), list) or not payload["episodes"]:
        return {"groups": {}, "initial_states": {}, "mismatch": "no episodes"}
    groups: dict[str, list] = {}
    initial_states: dict[str, set] = {}
    mismatch = None
    for episode in payload["episodes"]:
        if not _well_formed(episode):
            return {"groups": {}, "initial_states": {}, "mismatch": "malformed episode"}
        env_id, group_id = episode["env_id"], episode["group_id"]
        if group_id != env_id // group_size:
            mismatch = mismatch or f"episode {env_id}: group"
        env = environment(config)
        state = env.reset(seed=seed + env_id // group_size)
        if state != episode["initial_state"]:
            mismatch = mismatch or f"episode {env_id}: initial state"
        done = success = False
        executed, reward = 0, 0.0
        for turn in episode["turns"]:
            for action in turn:
                if done or executed >= budget or action not in (1, 2, 3, 4):
                    mismatch = mismatch or f"episode {env_id}: action after the episode ended"
                    break
                _, step_reward, step_done, info = env.step(action)
                reward += float(step_reward)
                executed += 1
                if step_done:
                    done, success = True, bool(info["success"])
        groups.setdefault(str(group_id), []).append((float(done and success), reward))
        initial_states.setdefault(str(group_id), set()).add(
            hashlib.sha256(state.encode()).hexdigest())
    return {
        "groups": {group: {"success_rate": sum(r[0] for r in rows) / len(rows),
                           "average_episodic_reward": sum(r[1] for r in rows) / len(rows)}
                   for group, rows in groups.items()},
        "initial_states": {group: sorted(values) for group, values in initial_states.items()},
        "mismatch": mismatch,
    }


if __name__ == "__main__":
    pristine, seed, group_size = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    print(json.dumps(replay(json.load(sys.stdin), pristine, seed, group_size), sort_keys=True))
