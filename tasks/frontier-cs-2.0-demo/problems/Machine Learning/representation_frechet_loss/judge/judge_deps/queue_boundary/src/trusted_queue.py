"""Trusted nn.Module/autograd proxy; submitted Python stays in its worker."""

import torch
from torch import nn
from torch.autograd.function import once_differentiable

import queue_wire as wire
from queue_process import QueueProcess


class QueueFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, op, x):
        identity, values = owner._forward(op, x)
        ctx.owner, ctx.identity = owner, identity
        ctx.input_shape, ctx.input_dtype, ctx.input_device = x.shape, x.dtype, x.device
        ctx.descriptors = [(v.shape, v.dtype, v.device) for v in values]
        return tuple(values)

    @staticmethod
    @once_differentiable
    def backward(ctx, *cotangents):
        values = [torch.zeros(shape, dtype=dtype, device=device) if g is None else g
                  for g, (shape, dtype, device) in zip(cotangents, ctx.descriptors)]
        owner = ctx.owner
        try:
            if owner.pending != ctx.identity or owner._versions() != owner.versions:
                raise ValueError("Queue state changed before backward")
            _, result = owner.worker.call(
                wire.VJP, [torch.tensor([ctx.identity], dtype=torch.int64), *values])
            if (len(result) != 1 or result[0].shape != ctx.input_shape
                    or result[0].dtype != ctx.input_dtype):
                raise ValueError("Queue VJP descriptor mismatch")
            owner.pending = None
            return None, None, result[0].to(ctx.input_device)
        except BaseException:
            owner.close()
            raise


class TrustedFeatureQueue(nn.Module):
    def __init__(self, size=50000, feat_dim=2048, online_accum=False, ema_beta=0.0, *, factory, allow_cuda=False):
        super().__init__()
        self.size, self.feat_dim = size, feat_dim
        self.online_accum, self.ema_beta = online_accum, float(ema_beta)
        self.ema_stats = self.ema_beta > 0
        self.layout = wire.state_layout(size, feat_dim, online_accum, self.ema_beta)
        for name, (shape, dtype) in self.layout.items():
            self.register_buffer(name, torch.zeros(shape, dtype=getattr(torch, dtype)))
        self.pending, self.versions = None, None
        self.stale, self.allow_cuda = False, allow_cuda
        self.queue_device = torch.device("cpu")
        self.worker = factory()
        try:
            _, response = self.worker.call(wire.CREATE, [
                torch.tensor([size, feat_dim, int(online_accum)], dtype=torch.int64),
                torch.tensor([self.ema_beta], dtype=torch.float64)])
            if response:
                raise ValueError("CREATE must not supply data or a report")
        except BaseException:
            self.close()
            raise

    def _versions(self):
        return tuple((id(self._buffers[n]), self._buffers[n]._version) for n in self.layout)

    def _idle(self):
        if self.pending is not None:
            raise ValueError("Finish backward before another queue operation")

    def _sync(self):
        self._idle()
        if self.stale:
            if self.versions != self._versions():
                raise ValueError("A retained buffer alias changed while worker state was newer")
            return
        if self.versions != self._versions():
            tensors = [self._buffers[name] for name in self.layout]
            wire.validate_state(tensors, self.layout)
            _, result = self.worker.call(wire.IMPORT, tensors)
            if result:
                self.close()
                raise ValueError("Unexpected state import response")
            self.versions = self._versions()

    def _refresh(self):
        self._idle()
        if self.stale:
            self._sync()
            _, values = self.worker.call(wire.EXPORT)
            self._accept_state(values)

    def __getattr__(self, name):
        # Keep ordinary upstream tensor slice/copy access working without
        # retransmitting the entire population after each mutation.
        buffers = self.__dict__.get("_buffers", {})
        if name in buffers and self.__dict__.get("stale", False):
            self._refresh()
        return super().__getattr__(name)

    def _accept_state(self, values):
        try:
            wire.validate_state(values, self.layout)
            with torch.no_grad():
                for name, value in zip(self.layout, values):
                    self._buffers[name].copy_(value)
            self.versions = self._versions()
            self.stale = False
        except BaseException:
            self.close()
            raise

    def _features(self, x):
        if (type(x) not in (torch.Tensor, nn.Parameter) or x.device != self.queue_device or x.ndim != 2
                or x.shape[1] != self.feat_dim or not 1 <= x.shape[0] <= (self.size or 50000)
                or not x.is_floating_point()):
            raise ValueError("Expected bounded, plain feature tensor on the queue device")

    def _forward(self, op, x):
        self._features(x)
        self._sync()
        try:
            identity, values = self.worker.call(op, [x.detach()])
            stats = op in (wire.STATS, wire.STATS_NOGRAD)
            shapes = [(self.feat_dim,), (self.feat_dim, self.feat_dim)] if stats else [
                (self.size if self.size else x.shape[0], self.feat_dim)]
            if len(values) != len(shapes) or any(
                    tuple(v.shape) != shape or not v.is_floating_point() for v, shape in zip(values, shapes)):
                raise ValueError("Queue forward descriptor mismatch")
            if op in (wire.STATS, wire.SNAPSHOT):
                self.pending = identity
            self.stale = True
            return identity, [value.to(x.device) for value in values]
        except BaseException:
            self.close()
            raise

    def _build(self, x, op, no_grad_op):
        if torch.is_grad_enabled() and x.requires_grad:
            return QueueFunction.apply(self, op, x)
        return tuple(self._forward(no_grad_op, x)[1])

    def build_feats_stats(self, new_feats):
        return self._build(new_feats, wire.STATS, wire.STATS_NOGRAD)

    def _build_feats_stats_ema(self, new_feats):
        return self.build_feats_stats(new_feats)

    def build_feats_snapshot(self, new_feats):
        return self._build(new_feats, wire.SNAPSHOT, wire.SNAPSHOT_NOGRAD)[0]

    def _mutation(self, op, inputs=()):
        self._sync()
        for tensor in inputs:
            self._features(tensor)
        _, result = self.worker.call(op, inputs)
        if result:
            self.close()
            raise ValueError("Mutation response must be empty")
        self.stale = True

    def enqueue(self, new_feats):
        self._mutation(wire.ENQUEUE, [new_feats.detach()])

    def accumulate_batch(self, feats):
        self._mutation(wire.ACCUMULATE, [feats.detach()])

    def _init_accumulators(self):
        self._mutation(wire.INIT)

    def _finalize_streaming_init(self):
        self._mutation(wire.FINALIZE)

    @property
    def pointer(self):
        return int(self.ptr.item())

    def state_dict(self, *args, **kwargs):
        self._sync()
        _, values = self.worker.call(wire.EXPORT)
        self._accept_state(values)
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._idle()
        if not strict or assign or set(state_dict) != set(self.layout):
            raise ValueError("Queue state must match its registered surface")
        wire.validate_state([state_dict[name] for name in self.layout], self.layout)
        self.stale = False
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self._sync()
        return result

    def cuda(self, device=None):
        if not self.allow_cuda:
            raise RuntimeError("CUDA was not enabled in this queue's confinement profile")
        target = torch.device("cuda", 0 if device is None else device) if isinstance(device, (int, type(None))) else torch.device(device)
        if target.type != "cuda" or target.index not in (None, 0):
            raise ValueError("Only the allocated CUDA device 0 is permitted")
        return self._apply(lambda tensor: tensor.to("cuda:0"))

    def _apply(self, fn, recurse=True):
        self._refresh()
        # Validate a tiny probe before moving a potentially large population.
        probe = None
        for dtype in {getattr(torch, name) for _, name in self.layout.values()}:
            probe = fn(torch.empty(0, dtype=dtype, device=self.queue_device))
            if probe.dtype != dtype or probe.device.type not in ("cpu", "cuda"):
                raise ValueError("Queue .to() preserves its declared buffer dtypes")
        if probe.device.type == "cuda" and (not self.allow_cuda or probe.device.index != 0):
            raise ValueError("Unallocated queue device")
        result = super()._apply(fn, recurse=recurse)
        device = next(iter(self._buffers.values())).device
        if device != self.queue_device:
            _, response = self.worker.call(wire.DEVICE, [torch.tensor([int(device.type == "cuda")])])
            if response:
                self.close()
                raise ValueError("Unexpected queue device response")
            self.queue_device = device
        self.versions = None
        return result

    def close(self):
        self.worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def bind_queue_class(source, scratch_root, *, python, deadline, seconds=15, processes=None, allow_cuda=False, namespace=False):
    """Bind trusted launch paths outside the caller-visible upstream signature."""
    def factory():
        worker = QueueProcess(source, scratch_root, python=python, deadline=deadline, seconds=seconds,
                              allow_cuda=allow_cuda, namespace=namespace)
        if processes is not None:
            processes.append(worker)
        return worker

    class FeatureQueue(TrustedFeatureQueue):
        def __init__(self, size=50000, feat_dim=2048, online_accum=False, ema_beta=0.0):
            super().__init__(size, feat_dim, online_accum, ema_beta, factory=factory, allow_cuda=allow_cuda)

    return FeatureQueue
