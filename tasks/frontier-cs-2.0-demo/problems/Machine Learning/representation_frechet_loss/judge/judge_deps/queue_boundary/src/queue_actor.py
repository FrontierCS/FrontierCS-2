"""Untrusted-side queue host: never creates trusted reports or checkpoints."""

import math
import os
from pathlib import Path
import select
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from restricted_runtime import install, restrict_syscalls
from queue_profile import runtime_paths
import queue_wire as wire


def main():
    source, scratch = (Path(p).resolve(strict=True) for p in sys.argv[1:3])
    deadline = float(sys.argv[4])
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise ValueError("Queue lifetime must have a finite future deadline")
    if sys.argv[3] not in ("cpu", "namespace_cuda", "namespace_cpu"):
        raise ValueError("Unknown queue runtime profile")
    allow_cuda = sys.argv[3] == "namespace_cuda"
    os.chdir(scratch)
    if sys.argv[3].startswith("namespace_"):
        if os.readlink("/proc/self/ns/pid") == os.environ["FQ_PARENT_PIDNS"]:
            raise RuntimeError("Queue actor shares its parent's PID namespace")
        if not os.statvfs("/").f_flag & os.ST_RDONLY:
            raise RuntimeError("Queue namespace root must be read-only")
        if not allow_cuda:
            restrict_syscalls("/actor/libseccomp.so.2")
    else:
        reads, writes = runtime_paths(source, scratch, Path(__file__).parent)
        install(reads, writes)
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    if os.geteuid() == 0:
        raise RuntimeError("Queue worker must not run submitted code as root")
    import torch
    torch.set_num_threads(1)
    if allow_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one isolated CUDA device is required")
        torch.empty(0, device="cuda:0")
        torch.cuda.synchronize()
        # Initialize only trusted libraries inside the isolated namespace.
        # Confine their helper threads before READY or candidate import.
        restrict_syscalls("/actor/libseccomp.so.2", synchronize_threads=True)
    # Reserve fd 1 for framed data even if submitted code prints.
    output = os.dup(1)
    os.dup2(2, 1)
    wire.send(output, 0, wire.READY, [], min(deadline, time.monotonic() + 10))
    sys.path.insert(0, str(source))
    from frechet_distance.queue import FeatureQueue
    queue, layout, graph, previous = None, None, None, 0
    device = torch.device("cpu")
    while True:
        try:
            # Parent-only feature generation can be idle for minutes. Bound
            # that wait by the phase lifetime, then bound each started frame.
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([0], [], [], remaining)[0]:
                raise TimeoutError("Queue lifetime deadline")
            seq, op, tensors = wire.receive(0, min(deadline, time.monotonic() + 30))
        except EOFError:
            return
        if seq != previous + 1:
            raise ValueError("Unordered queue request")
        previous = seq
        result = []
        if op == wire.CREATE:
            if queue is not None or len(tensors) != 2:
                raise ValueError("Invalid create")
            spec, rate = tensors
            if spec.dtype != torch.int64 or tuple(spec.shape) != (3,) or rate.dtype != torch.float64 or tuple(rate.shape) != (1,):
                raise ValueError("Invalid create descriptor")
            size, dim, online = spec.tolist()
            if online not in (0, 1):
                raise ValueError("Invalid online flag")
            beta = float(rate.item())
            layout = wire.state_layout(size, dim, bool(online), beta)
            queue = FeatureQueue(size, dim, bool(online), beta)
            if set(queue.state_dict()) != set(layout):
                raise ValueError("Queue changed registered buffer surface")
        else:
            if queue is None:
                raise ValueError("Queue not initialized")
            if graph is not None and op != wire.VJP:
                raise ValueError("Forward graph is still live")
            if op == wire.DEVICE:
                if (len(tensors) != 1 or tensors[0].dtype != torch.int64
                        or tuple(tensors[0].shape) != (1,) or tensors[0].item() not in (0, 1)):
                    raise ValueError("Invalid queue device request")
                if tensors[0].item() and not allow_cuda:
                    raise ValueError("CUDA was not enabled in the confined profile")
                device = torch.device("cuda:0" if tensors[0].item() else "cpu")
                queue.to(device)
            elif op == wire.IMPORT:
                wire.validate_state(tensors, layout)
                queue.load_state_dict(dict(zip(layout, tensors)), strict=True)
            elif op == wire.EXPORT:
                if tensors:
                    raise ValueError("Unexpected export input")
                result = list(queue.state_dict()[name] for name in layout)
                wire.validate_state(result, layout)
            elif op in (wire.INIT, wire.ACCUMULATE, wire.FINALIZE, wire.ENQUEUE):
                needs_input = op in (wire.ACCUMULATE, wire.ENQUEUE)
                if len(tensors) != int(needs_input):
                    raise ValueError("Mutation arity mismatch")
                if needs_input:
                    check_features(tensors[0], dim, size)
                    tensors = [tensors[0].to(device)]
                method = {wire.INIT: "_init_accumulators", wire.ACCUMULATE: "accumulate_batch",
                          wire.FINALIZE: "_finalize_streaming_init", wire.ENQUEUE: "enqueue"}[op]
                with torch.no_grad():
                    getattr(queue, method)(*tensors)
                # The parent fetches state only when upstream code observes it.
                # Echoing the full population on every enqueue costs hundreds
                # of MiB per step for a single queue.
            elif op in (wire.STATS, wire.SNAPSHOT, wire.STATS_NOGRAD, wire.SNAPSHOT_NOGRAD):
                if len(tensors) != 1:
                    raise ValueError("Forward arity mismatch")
                x = tensors[0].to(device)
                check_features(x, dim, size)
                differentiable = op in (wire.STATS, wire.SNAPSHOT)
                x.requires_grad_(differentiable)
                with torch.set_grad_enabled(differentiable):
                    value = (queue.build_feats_stats(x) if op in (wire.STATS, wire.STATS_NOGRAD)
                             else queue.build_feats_snapshot(x))
                result = list(value) if isinstance(value, tuple) else [value]
                if differentiable:
                    graph = seq, x, result
            elif op == wire.VJP:
                if graph is None or len(tensors) != len(graph[2]) + 1:
                    raise ValueError("Missing forward graph")
                identity = tensors[0]
                if identity.dtype != torch.int64 or tuple(identity.shape) != (1,) or identity.item() != graph[0]:
                    raise ValueError("Stale VJP graph")
                for value, tangent in zip(graph[2], tensors[1:]):
                    if value.shape != tangent.shape or value.dtype != tangent.dtype:
                        raise ValueError("Cotangent descriptor mismatch")
                cotangents = [tangent.to(value.device) for value, tangent in zip(graph[2], tensors[1:])]
                result = [torch.autograd.grad(graph[2], graph[1], grad_outputs=cotangents)[0]]
                graph = None
            else:
                raise ValueError("Unknown queue operation")
        wire.send(output, seq, op, result, min(deadline, time.monotonic() + 30))


def check_features(x, dim, size):
    if (x.ndim != 2 or x.shape[1] != dim or not 1 <= x.shape[0] <= (size or 50000)
            or not x.is_floating_point()):
        raise ValueError("Feature tensor outside queue contract")


if __name__ == "__main__":
    main()
