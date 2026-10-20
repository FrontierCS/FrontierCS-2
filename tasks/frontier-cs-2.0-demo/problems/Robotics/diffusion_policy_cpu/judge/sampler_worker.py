"""Sampler subprocess using bounded values inside the standard Verifier sandbox.

The exact restored sampler and candidate sampler use this same entrypoint.
Private reference data is not sent through the protocol; post-trial review
checks submitted code and available evidence for concrete misuse.
"""

import sys

from bridge_protocol import (
    MAX_FRAME_BYTES, ProtocolError, decode, encode, exact_fields,
    public_configuration, sample_arguments, tensor, validate_result,
)


def read_frame():
    raw = sys.stdin.buffer.readline(MAX_FRAME_BYTES + 1)
    if not raw:
        raise EOFError
    return decode(raw)


def write_frame(value):
    sys.stdout.buffer.write(encode(value))
    sys.stdout.buffer.flush()


def main():
    configure = read_frame()
    exact_fields(configure, {"type", "version", "config"})
    if configure["type"] != "configure" or type(configure["version"]) is not int or configure["version"] != 1:
        raise ProtocolError("Invalid handshake.")
    config = public_configuration(configure["config"])

    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    request_id = 0

    def pack(value):
        dtype = str(value.dtype).removeprefix("torch.")
        return tensor({"dtype": dtype, "shape": list(value.shape),
                       "data": value.detach().cpu().reshape(-1).tolist()})

    def unpack(value):
        value = tensor(value)
        return torch.tensor(value["data"], dtype=getattr(torch, value["dtype"]), device="cpu").reshape(value["shape"])

    def request(operation, arguments):
        nonlocal request_id
        request_id += 1
        write_frame({"type": "request", "id": request_id, "op": operation, "arguments": arguments})
        result = read_frame()
        exact_fields(result, {"type", "id", "tensor"})
        if result["type"] != "response" or result["id"] != request_id:
            raise ProtocolError("Response identity mismatch.")
        return unpack(result["tensor"])

    class FrozenModel(torch.nn.Module):
        def forward(self, sample, timestep, cond=None):
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.as_tensor(timestep)
            return request("model_forward", {"sample": pack(sample), "timestep": pack(timestep),
                                             "cond": None if cond is None else pack(cond)})

    scheduler = DDPMScheduler(**config["scheduler"])
    policy = DiffusionTransformerLowdimPolicy(
        model=FrozenModel(), noise_scheduler=scheduler,
        **{key: value for key, value in config.items() if key != "scheduler"})
    policy.to("cpu").eval()

    def normal(*size, **kwargs):
        if "size" in kwargs:
            if size:
                raise ProtocolError("Duplicate normal shape.")
            shape = kwargs.pop("size")
        else:
            shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
        dtype = kwargs.pop("dtype", None) or torch.get_default_dtype()
        device = kwargs.pop("device", None)
        generator = kwargs.pop("generator", None)
        layout = kwargs.pop("layout", torch.strided)
        requires_grad = kwargs.pop("requires_grad", False)
        if (device is not None and str(device) != "cpu") or generator is not None or layout != torch.strided or requires_grad or kwargs:
            raise ProtocolError("Unsupported random tensor request.")
        return request("random_normal", {"dtype": str(dtype).removeprefix("torch."), "shape": list(shape)})

    def normal_like(value, **kwargs):
        kwargs.setdefault("dtype", value.dtype)
        kwargs.setdefault("device", value.device)
        return normal(size=value.shape, **kwargs)

    # Random values come from the trusted per-episode generator; its seed and
    # state never enter worker configuration or a response object.
    torch.randn = normal
    torch.randn_like = normal_like
    write_frame({"type": "ready", "version": 1})
    while True:
        message = read_frame()
        exact_fields(message, {"type", "id", "arguments"})
        if message["type"] != "sample" or type(message["id"]) is not int:
            raise ProtocolError("Expected sample request.")
        arguments = sample_arguments(message["arguments"])
        with torch.no_grad():
            result = policy.conditional_sample(
                unpack(arguments["condition_data"]), unpack(arguments["condition_mask"]),
                cond=None if arguments["cond"] is None else unpack(arguments["cond"]),
                generator=None, **arguments["kwargs"])
        payload = validate_result(pack(result), arguments)
        write_frame({"type": "sample_result", "id": message["id"], "tensor": payload})


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        pass
    except Exception:
        # No untrusted exception text is promoted to trusted feedback.
        raise SystemExit(2)
