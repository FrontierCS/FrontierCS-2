"""Trusted bootstrap: initialize public model, seal, then execute the sampler."""

import contextlib
import json
import os
from pathlib import Path
import resource
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wire import ACTION_SHAPE, CONTEXT_SHAPE, MAX_MESSAGE_BYTES, MAX_SOURCE_BYTES
from wire import SAMPLE_SHAPE, decode_tensor, encode_tensor
from seal_process import bind_to_parent, seal_process


def bootstrap_public_repo(public_repo):
    # Python -I skips sitecustomize/PYTHONPATH, so install the trusted adapter here.
    public_repo = Path(public_repo)
    sys.path.insert(0, str(public_repo / "frontier_runtime"))
    from pruned_imports import install

    install()
    sys.path.insert(0, str(public_repo / "src"))


def main():
    bind_to_parent(int(sys.argv[2]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_MESSAGE_BYTES, MAX_MESSAGE_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (1650, 1650))
    raw = sys.stdin.buffer.read(MAX_SOURCE_BYTES * 2 + 1)
    if len(raw) > MAX_SOURCE_BYTES * 2:
        raise ValueError("oversized trusted request")
    request = json.loads(raw)
    handshake = int(sys.argv[1])
    from memory_modules import CandidateModules

    modules = CandidateModules(request["sources"])
    # Only the source-locked public model package is imported before sealing.
    bootstrap_public_repo(request["public_repo"])
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from safetensors.torch import load_file
        from models.nanowm import NanoWM_models
        import rng_state

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("one NVIDIA GPU is required for the isolated sampler")
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = request["backend"]["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = request["backend"]["cudnn_tf32"]
        device = torch.device("cuda:0")
        model = NanoWM_models["NanoWM-B/2"](
            input_size=32, in_channels=4, num_classes=1000, num_frames=4,
            extras=1, use_action=True, action_dim=10,
            action_injection_type="additive", causal=True,
        )
        model.load_state_dict(load_file(request["checkpoint"], device="cpu"), strict=True)
        model = model.eval().requires_grad_(False).to(device)
        context = torch.frombuffer(bytearray(decode_tensor(request["context"], CONTEXT_SHAPE)),
                                   dtype=torch.float32).reshape(CONTEXT_SHAPE).clone().to(device)
        action = torch.frombuffer(bytearray(decode_tensor(request["action"], ACTION_SHAPE)),
                                  dtype=torch.float32).reshape(ACTION_SHAPE).clone().to(device)
        with torch.no_grad():
            # Initialize CUDA kernels before filesystem access becomes unavailable.
            model(torch.zeros(SAMPLE_SHAPE, device=device),
                  torch.zeros((1, 4), dtype=torch.long, device=device),
                  y=None, action=action)
            functional.scaled_dot_product_attention(
                torch.zeros((1, 1, 4, 32), device=device),
                torch.zeros((1, 1, 4, 32), device=device),
                torch.zeros((1, 1, 4, 32), device=device))
            torch.cuda.synchronize()
        rng_state.capture()
        modules.install()
        gpu_descriptors = []
        for entry in Path("/proc/self/fd").iterdir():
            try:
                target = os.readlink(entry)
                descriptor = int(entry.name)
                if (target.startswith("/dev/nvidia")
                        and stat.S_ISCHR(os.fstat(descriptor).st_mode)):
                    gpu_descriptors.append(descriptor)
            except OSError:
                continue
        if not gpu_descriptors:
            raise RuntimeError("no initialized NVIDIA device descriptors")
        seal_process(allowed_ioctl_fds=gpu_descriptors)
        os.write(handshake, b"sealed-v1\n")
        os.close(handshake)
        from diffusion import create_diffusion
        from diffusion.df_sample import dfot_sample

        diffusion = create_diffusion(
            timestep_respacing="", noise_schedule="squaredcos_cap_v2",
            pred_name="v", diffusion_steps=1000, snr_gamma=5.0,
            zero_terminal_snr=True,
        )
        rng_state.restore(request["rng"])
        with torch.no_grad():
            answer = dfot_sample(
                diffusion=diffusion, model=model, shape=SAMPLE_SHAPE,
                context=context, model_kwargs={"y": None, "action": action},
                device=device, **request["options"],
            )
        if not isinstance(answer, torch.Tensor) or tuple(answer.shape) != SAMPLE_SHAPE:
            raise ValueError("sampler returned an invalid tensor")
        if answer.dtype != torch.float32 or answer.device != device:
            raise ValueError("sampler returned an invalid dtype or device")
        tensor = encode_tensor(answer.detach().cpu().contiguous().numpy().tobytes(), SAMPLE_SHAPE)
        rng = rng_state.capture()
    print(json.dumps({"tensor": tensor, "rng": rng}, allow_nan=False))


if __name__ == "__main__":
    main()
