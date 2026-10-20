"""Carry primitive RNG state across the isolated sampling boundary."""

import base64
import math
import random

import numpy as np
import torch


def capture():
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        "torch_cpu": base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode(),
        "torch_cuda": [base64.b64encode(state.cpu().numpy().tobytes()).decode()
                       for state in torch.cuda.get_rng_state_all()],
    }


def _integers(values, length, maximum):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError("invalid RNG integer array length")
    if any(type(value) is not int or value < 0 or value > maximum for value in values):
        raise ValueError("invalid RNG integer")


def _gaussian(value):
    try:
        valid = value is None or (type(value) in (int, float) and math.isfinite(value))
    except OverflowError:
        # JSON integers can exceed the finite float range of Gaussian caches.
        valid = False
    if not valid:
        raise ValueError("invalid RNG Gaussian cache")


def _bytes(value, expected):
    if not isinstance(value, str) or len(value) != 4 * ((expected + 2) // 3):
        raise ValueError("invalid RNG byte length")
    data = base64.b64decode(value, validate=True)
    if len(data) != expected:
        raise ValueError("invalid decoded RNG byte length")
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def restore(state):
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("invalid RNG state fields")
    python = state["python"]
    if not isinstance(python, (list, tuple)) or len(python) != 3 or python[0] != 3:
        raise ValueError("invalid Python RNG state")
    _integers(python[1], 625, 2**32 - 1)
    if python[1][-1] > 624:
        raise ValueError("invalid Python RNG cursor")
    _gaussian(python[2])
    numpy = state["numpy"]
    if not isinstance(numpy, (list, tuple)) or len(numpy) != 5 or numpy[0] != "MT19937":
        raise ValueError("invalid NumPy RNG state")
    _integers(numpy[1], 624, 2**32 - 1)
    if type(numpy[2]) is not int or not 0 <= numpy[2] <= 624 or numpy[3] not in (0, 1):
        raise ValueError("invalid NumPy RNG cursor")
    _gaussian(numpy[4])
    cpu = _bytes(state["torch_cpu"], torch.get_rng_state().numel())
    cuda_states = torch.cuda.get_rng_state_all()
    if not isinstance(state["torch_cuda"], list) or len(state["torch_cuda"]) != len(cuda_states):
        raise ValueError("invalid CUDA RNG count")
    cuda = [_bytes(value, original.numel())
            for value, original in zip(state["torch_cuda"], cuda_states)]
    python_state = (3, tuple(python[1]), python[2])
    numpy_state = (numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:])
    # A correctly sized byte array can still encode an invalid generator state.
    # Validate every generator before changing any trusted caller RNG.
    try:
        random.Random().setstate(python_state)
        np.random.RandomState().set_state(numpy_state)
        torch.Generator(device="cpu").set_state(cpu)
        for index, value in enumerate(cuda):
            torch.Generator(device="cuda:" + str(index)).set_state(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("invalid encoded RNG state") from exc
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(cuda)
