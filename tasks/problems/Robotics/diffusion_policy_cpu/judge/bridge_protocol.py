"""Closed, value-only sampler protocol. No Python object serialization."""

from __future__ import annotations

import json
import math

VERSION = 1
MAX_FRAME_BYTES = 2097152
MAX_ELEMENTS = 65536
DTYPES = {"float32", "float64", "int64", "bool"}


class ProtocolError(ValueError):
    pass


def exact_fields(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ProtocolError("Unexpected protocol fields.")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("Duplicate protocol key.")
        result[key] = value
    return result


def encode(value):
    try:
        raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("ascii") + b"\n"
    except (ValueError, TypeError, RecursionError) as exc:
        raise ProtocolError("Protocol values must be finite JSON.") from exc
    if len(raw) > MAX_FRAME_BYTES:
        raise ProtocolError("Protocol frame exceeds its bound.")
    return raw


def decode(raw):
    if len(raw) > MAX_FRAME_BYTES or not raw.endswith(b"\n"):
        raise ProtocolError("Invalid frame boundary.")
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("Non-finite JSON.")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("Malformed protocol JSON.") from exc


def tensor(value):
    exact_fields(value, {"dtype", "shape", "data"})
    dtype, shape, data = value["dtype"], value["shape"], value["data"]
    if type(dtype) is not str or dtype not in DTYPES or type(shape) is not list or len(shape) > 4:
        raise ProtocolError("Unsupported tensor metadata.")
    size = 1
    for dimension in shape:
        if type(dimension) is not int or not 0 < dimension <= MAX_ELEMENTS:
            raise ProtocolError("Invalid tensor dimension.")
        size *= dimension
        if size > MAX_ELEMENTS:
            raise ProtocolError("Tensor exceeds its element bound.")
    if type(data) is not list or len(data) != size:
        raise ProtocolError("Tensor shape and storage disagree.")
    for item in data:
        if dtype == "bool":
            valid = type(item) is bool
        elif dtype == "int64":
            valid = type(item) is int and -(2**63) <= item < 2**63
        else:
            try:
                valid = type(item) in (int, float) and math.isfinite(item)
            except OverflowError:
                valid = False
            if valid:
                bound = 3.4028234663852886e38 if dtype == "float32" else 1.7976931348623157e308
                valid = abs(item) <= bound
        if not valid:
            raise ProtocolError("Invalid tensor element.")
    # Return a fresh value tree, never caller-owned mutable storage.
    return {"dtype": dtype, "shape": list(shape), "data": list(data)}


def sample_arguments(value):
    exact_fields(value, {"condition_data", "condition_mask", "cond", "kwargs"})
    data, mask = tensor(value["condition_data"]), tensor(value["condition_mask"])
    cond = None if value["cond"] is None else tensor(value["cond"])
    if data["dtype"] not in {"float32", "float64"} or mask["dtype"] != "bool" or data["shape"] != mask["shape"]:
        raise ProtocolError("Invalid conditioning tensors.")
    if cond is not None and cond["dtype"] != data["dtype"]:
        raise ProtocolError("Conditioning dtype mismatch.")
    # The selected checkpoint supplies no extra scheduler.step keywords. A
    # nonempty checkpoint contract needs explicit support, never arbitrary RPC.
    if value["kwargs"] != {}:
        raise ProtocolError("Scheduler keyword support requires contract review.")
    return {"condition_data": data, "condition_mask": mask, "cond": cond, "kwargs": {}}


def validate_result(value, arguments):
    result = tensor(value)
    expected = arguments["condition_data"]
    if result["shape"] != expected["shape"] or result["dtype"] != expected["dtype"]:
        raise ProtocolError("Sampler result shape or dtype changed.")
    for keep, before, after in zip(arguments["condition_mask"]["data"], expected["data"], result["data"]):
        if keep and before != after:
            raise ProtocolError("Sampler changed a conditioned entry.")
    return result


def model_arguments(value):
    exact_fields(value, {"sample", "timestep", "cond"})
    sample, timestep = tensor(value["sample"]), tensor(value["timestep"])
    cond = None if value["cond"] is None else tensor(value["cond"])
    if sample["dtype"] not in {"float32", "float64"} or len(sample["shape"]) != 3:
        raise ProtocolError("Model input must be a floating batch of sequences.")
    if timestep["dtype"] not in {"int64", "float32", "float64"} or len(timestep["shape"]) > 1:
        raise ProtocolError("Unsupported timestep tensor.")
    return {"sample": sample, "timestep": timestep, "cond": cond}


def normal_arguments(value):
    exact_fields(value, {"dtype", "shape"})
    shape = value["shape"]
    if type(value["dtype"]) is not str or value["dtype"] not in {"float32", "float64"} or type(shape) is not list or len(shape) > 4:
        raise ProtocolError("Invalid normal request.")
    size = 1
    for dimension in shape:
        if type(dimension) is not int or not 0 < dimension <= MAX_ELEMENTS:
            raise ProtocolError("Invalid normal shape.")
        size *= dimension
        if size > MAX_ELEMENTS:
            raise ProtocolError("Normal request is too large.")
    return {"dtype": value["dtype"], "shape": list(shape)}


def public_configuration(value):
    keys = {"horizon", "obs_dim", "action_dim", "n_action_steps", "n_obs_steps",
            "num_inference_steps", "obs_as_cond", "pred_action_steps_only", "scheduler"}
    exact_fields(value, keys)
    for key in keys - {"scheduler", "obs_as_cond", "pred_action_steps_only"}:
        if type(value[key]) is not int or not 0 < value[key] <= 10000:
            raise ProtocolError("Invalid public policy dimension.")
    for key in ("obs_as_cond", "pred_action_steps_only"):
        if type(value[key]) is not bool:
            raise ProtocolError("Invalid public policy option.")
    scheduler = value["scheduler"]
    expected = {"num_train_timesteps", "beta_start", "beta_end", "beta_schedule",
                "variance_type", "clip_sample", "prediction_type"}
    exact_fields(scheduler, expected)
    if type(scheduler["num_train_timesteps"]) is not int or not 0 < scheduler["num_train_timesteps"] <= 10000:
        raise ProtocolError("Invalid public schedule length.")
    for key in ("beta_start", "beta_end"):
        if type(scheduler[key]) not in (float, int) or not 0 < scheduler[key] < 1:
            raise ProtocolError("Invalid public beta parameter.")
    if type(scheduler["beta_schedule"]) is not str or scheduler["beta_schedule"] not in {"linear", "scaled_linear", "squaredcos_cap_v2"}:
        raise ProtocolError("Unsupported public beta schedule.")
    if type(scheduler["variance_type"]) is not str or scheduler["variance_type"] not in {"fixed_small", "fixed_small_log", "fixed_large", "fixed_large_log"}:
        raise ProtocolError("Unsupported public variance mode.")
    if type(scheduler["clip_sample"]) is not bool or type(scheduler["prediction_type"]) is not str or scheduler["prediction_type"] not in {"epsilon", "sample", "v_prediction"}:
        raise ProtocolError("Unsupported public prediction mode.")
    return decode(encode(value))
