"""Trusted native-call adapter; targets and scoring never enter the child."""

import json
import os
import time
from pathlib import Path

from process_transport import run_worker
from wire import ACTION_SHAPE, CONTEXT_SHAPE, SAMPLE_SHAPE
from wire import InvalidResponse, decode_tensor, encode_tensor, memory_sources


def audit(event, detail=None):
    with open(os.environ["NANOWM_BRIDGE_AUDIT"], "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "detail": detail}) + "\n")


def _isolated_sample(diffusion, model, shape, context=None, n_context_frames=0,
                    scheduling_mode="pyramid", num_sampling_steps=50,
                    model_kwargs=None, device=None, progress=True, eta=0.0,
                    clip_denoised=False, n_generate_frames=None,
                    history_stabilization_level=0.0):
    import torch
    import rng_state

    if tuple(shape) != SAMPLE_SHAPE or n_context_frames != 1:
        raise RuntimeError("trusted caller changed the isolated task shape")
    if model_kwargs is None or set(model_kwargs) != {"y", "action"} or model_kwargs["y"] is not None:
        raise RuntimeError("trusted caller changed conditioning")
    if tuple(context.shape) != CONTEXT_SHAPE or tuple(model_kwargs["action"].shape) != ACTION_SHAPE:
        raise RuntimeError("trusted caller changed context or action shape")
    state = rng_state.capture()

    def message(tensor, expected):
        # A view of the context may share storage with all target latents.
        # Only compact logical bytes cross the pipe, never a Tensor/pickle object.
        copied = tensor.detach().cpu().contiguous().clone()
        return encode_tensor(copied.numpy().tobytes(), expected)

    request = {
        "context": message(context, CONTEXT_SHAPE),
        "action": message(model_kwargs["action"], ACTION_SHAPE),
        "rng": state,
        "sources": memory_sources(os.environ["NANOWM_CANDIDATE_REPO"]),
        "public_repo": os.environ["NANOWM_PUBLIC_REPO"],
        "checkpoint": str(Path(os.environ["FRONTIER_MODEL_ROOT"]) / "point-maze/model.safetensors"),
        "backend": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32},
        "options": {"n_context_frames": n_context_frames, "scheduling_mode": scheduling_mode,
                    "num_sampling_steps": num_sampling_steps, "progress": progress,
                    "eta": eta, "clip_denoised": clip_denoised,
                    "n_generate_frames": n_generate_frames,
                    "history_stabilization_level": history_stabilization_level},
    }
    environment = {name: os.environ[name] for name in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
                   if name in os.environ}
    remaining = float(os.environ["NANOWM_CALLER_DEADLINE"]) - time.monotonic()
    if remaining <= 0:
        raise InvalidResponse("candidate exceeded its native caller time budget")
    response = run_worker(Path(__file__).with_name("sampler_worker.py"), request,
                          timeout=remaining, extra_env=environment)
    try:
        raw = decode_tensor(response["tensor"], SAMPLE_SHAPE)
        rng_state.restore(response["rng"])
    except (ValueError, TypeError, KeyError) as exc:
        raise InvalidResponse("candidate returned malformed tensor or RNG state") from exc
    answer = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(SAMPLE_SHAPE).clone().to(device)
    return answer


def isolated_sample(*args, **kwargs):
    audit("call_start")
    try:
        answer = _isolated_sample(*args, **kwargs)
    except InvalidResponse:
        audit("candidate_failure")
        raise
    except Exception as exc:
        audit("infrastructure_failure", str(exc)[-8192:])
        raise
    audit("call_complete")
    return answer


def install():
    import diffusion.df_sample as sampling
    sampling.dfot_sample = isolated_sample
    audit("installed")
