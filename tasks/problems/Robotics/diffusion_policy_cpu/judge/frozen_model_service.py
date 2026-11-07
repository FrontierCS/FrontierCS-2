"""Trusted tensor adapters; construction only, no inference has been run."""

from bridge_protocol import ProtocolError, model_arguments, normal_arguments, tensor


class FrozenModelService:
    def __init__(self, model, generator, *, horizon, action_dim, obs_dim, n_obs_steps,
                 num_train_timesteps, max_batch):
        self.model = model
        self.generator = generator
        self.horizon = horizon
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.n_obs_steps = n_obs_steps
        self.num_train_timesteps = num_train_timesteps
        self.max_batch = max_batch
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ProtocolError("Trusted model must be frozen in evaluation mode.")

    @staticmethod
    def pack(value):
        return tensor({"dtype": str(value.dtype).removeprefix("torch."),
                       "shape": list(value.shape), "data": value.detach().cpu().reshape(-1).tolist()})

    @staticmethod
    def unpack(value):
        import torch
        value = tensor(value)
        return torch.tensor(value["data"], dtype=getattr(torch, value["dtype"]), device="cpu").reshape(value["shape"])

    def forward(self, value):
        import torch
        args = model_arguments(value)
        batch, steps, dimensions = args["sample"]["shape"]
        if not 0 < batch <= self.max_batch or steps != self.horizon or dimensions != self.action_dim:
            raise ProtocolError("Model request exceeds the fixed tensor interface.")
        if args["sample"]["dtype"] != "float32":
            raise ProtocolError("The selected checkpoint uses float32.")
        cond = args["cond"]
        if cond is None or cond["shape"] != [batch, self.n_obs_steps, self.obs_dim] or cond["dtype"] != "float32":
            raise ProtocolError("Observation conditioning shape mismatch.")
        if args["timestep"]["shape"] not in ([], [1], [batch]):
            raise ProtocolError("Timestep batch shape mismatch.")
        if any(not 0 <= t < self.num_train_timesteps for t in args["timestep"]["data"]):
            raise ProtocolError("Timestep is outside the trained schedule.")
        with torch.no_grad():
            result = self.model(self.unpack(args["sample"]), self.unpack(args["timestep"]), self.unpack(cond))
        payload = self.pack(result)
        if payload["shape"] != args["sample"]["shape"] or payload["dtype"] != "float32":
            raise ProtocolError("Trusted model output contract changed.")
        return payload

    def normal(self, value):
        import torch
        args = normal_arguments(value)
        result = torch.randn(args["shape"], dtype=getattr(torch, args["dtype"]),
                             device="cpu", generator=self.generator)
        return self.pack(result)


def connect_sampler(policy, bridge):
    """One task-owned hook for either candidate or restored-reference worker."""
    def conditional_sample(condition_data, condition_mask, cond=None, generator=None, **kwargs):
        if generator is not None:
            raise ProtocolError("The evaluator owns the episode generator.")
        payload = bridge.sample({
            "condition_data": FrozenModelService.pack(condition_data),
            "condition_mask": FrozenModelService.pack(condition_mask),
            "cond": None if cond is None else FrozenModelService.pack(cond), "kwargs": kwargs})
        return FrozenModelService.unpack(payload)
    policy.conditional_sample = conditional_sample
