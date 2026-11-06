"""Training procedure for the Unitree Go1 joystick policy.

This file is the task's solver interface. The judge imports this module from the
submitted tree and calls ``train(seed, deadline_seconds)`` once per training
seed on one H100. The version shipped here is the reference procedure: Brax PPO
on ``Go1JoystickFlatTerrain`` with the repository's own environment, reward,
domain randomisation and PPO configuration
(``mujoco_playground/config/locomotion_params.py``), unmodified: 200 million
environment steps with that configuration's own evaluation schedule.

Contract
--------
``train`` returns ``policy``, a JAX-traceable function

    policy(obs_state: jax.Array[48], rng: jax.Array) -> action: jax.Array[12]

* ``obs_state`` is the environment's actor observation ``obs["state"]`` for one
  environment (float32, shape ``(48,)``). No privileged observation is passed.
* ``rng`` is a JAX PRNG key (``jax.random.PRNGKey``, uint32 shape ``(2,)``). The
  graded policy is evaluated once per control step; a deterministic policy may
  ignore the key.
* ``action`` is float32 with shape ``(12,)`` and must be finite.
* The function must be traceable by ``jax.jit``/``jax.vmap`` with those shapes.
  Trained parameters are captured by closure; the judge serialises the traced
  computation and evaluates it in its own unmodified copy of the environment.

``deadline_seconds`` is the training budget: the wall-clock time, in seconds
from the moment ``train`` is called, by which ``train`` must have returned.
Everything inside ``train`` is charged, including JIT compilation. The judge
kills a training run that overruns, and the submission is then invalid.
"""

import functools

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax.numpy as jp

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

ENV_NAME = "Go1JoystickFlatTerrain"


def train(seed: int, deadline_seconds: float):
  """Trains a Go1 joystick policy and returns ``policy(obs_state, rng)``."""
  del deadline_seconds  # The reference procedure runs a fixed number of steps.
  env = registry.load(ENV_NAME)
  env_cfg = registry.get_default_config(ENV_NAME)
  ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)

  training_params = dict(ppo_params)
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in training_params:
    del training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )

  make_inference_fn, params, _ = ppo.train(
      environment=env,
      eval_env=registry.load(ENV_NAME, config=env_cfg),
      wrap_env_fn=wrapper.wrap_for_brax_training,
      randomization_fn=registry.get_domain_randomizer(ENV_NAME),
      network_factory=network_factory,
      seed=seed,
      **training_params,
  )
  brax_policy = make_inference_fn(params, deterministic=True)
  # Brax normalises the whole observation dictionary before the actor network
  # selects its "state" entry. The privileged entry is never read by the actor,
  # so a constant placeholder of the right shape stands in for it.
  privileged_placeholder = jp.zeros_like(params[0].mean["privileged_state"])

  def policy(obs_state, rng):
    observation = {"state": obs_state, "privileged_state": privileged_placeholder}
    action, _ = brax_policy(observation, rng)
    return action

  return policy
