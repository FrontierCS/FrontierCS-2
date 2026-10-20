"""Observe unchanged upstream calls without replacing their scientific bodies.

Only the admitted, byte-exact upstream spine and trusted queue proxy may execute
in this process. queue_observer_entry.py installs the proxy before any upstream
import in either phase; submitted Python runs only inside the confined actor.
"""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    opts, argv = parser.parse_known_args()
    if argv and argv[0] == "--":
        argv = argv[1:]
    sys.path.insert(0, str(opts.repo))
    import torch
    import main_fd
    import utils.checkpoint_util as checkpoint_util

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one NVIDIA GPU is required")
    device = torch.cuda.get_device_properties(0)
    if "A100" not in device.name or device.total_memory < 75 * 1024**3:
        raise ValueError("workload requires the declared A100 80GB")
    report = {"gpu": True, "device": device.name, "device_memory_bytes": device.total_memory}
    report["runtime"] = {"pid": os.getpid(), "uid": os.getuid(), "gid": os.getgid(),
                         "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
                         "device_count": torch.cuda.device_count()}
    opts.report.write_text(json.dumps(report, allow_nan=False) + "\n")
    loaded = []
    original_load = checkpoint_util._load_model_sd

    def checked_load(model, state, strict=True):
        result = original_load(model, state, strict=strict)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError("checkpoint does not exactly match the generation model")
        loaded.append(True)
        return result

    checkpoint_util._load_model_sd = checked_load

    def finite(value):
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError("non-finite scientific tensor")
        elif isinstance(value, (tuple, list)):
            for item in value:
                finite(item)

    if opts.phase == "train":
        args = main_fd.get_args_parser().parse_args(argv)
        import frechet_distance.judges as judge_module
        import frechet_distance.losses as loss_module

        counts = {"warm": 0, "features": 0, "covariances": 0, "steps": 0, "optimizer": 0}
        state = {}
        original_create = main_fd.create_generation_model

        def create_model(options):
            model, ema = original_create(options)
            if sum(p.numel() for p in model.parameters()) != 118234240:
                raise ValueError("unexpected pMF-B architecture")
            state["model"] = model
            return model, ema

        main_fd.create_generation_model = create_model
        original_extract = judge_module.extract_judge_features

        def extract(judge, images):
            finite(images)
            features = original_extract(judge, images)
            finite(features)
            if any(p.requires_grad for p in judge["model"].parameters()):
                raise ValueError("representation model is not frozen")
            counts["features"] += 1
            return features

        judge_module.extract_judge_features = extract
        main_fd.extract_judge_features = extract
        original_trace = loss_module._compute_trace_term

        def trace(sigma, sigma_ref, sigma_ref_sqrt=None):
            finite((sigma, sigma_ref, sigma_ref_sqrt))
            counts["covariances"] += 1
            return original_trace(sigma, sigma_ref, sigma_ref_sqrt)

        loss_module._compute_trace_term = trace
        original_fill = main_fd.fill_all_queues

        def fill(judges, model, options, tokenizer=None):
            state["judges"] = judges
            generate = model.generate

            def counted_generate(batch_size, *a, **kw):
                images = generate(batch_size, *a, **kw)
                finite(images)
                if images.shape[0] != batch_size:
                    raise ValueError("warm-start generation count mismatch")
                counts["warm"] += batch_size
                return images

            model.generate = counted_generate
            try:
                result = original_fill(judges, model, options, tokenizer=tokenizer)
            finally:
                model.generate = generate
            if counts["warm"] != options.queue_size:
                raise ValueError("warm-start population was not generated")
            return result

        main_fd.fill_all_queues = fill
        original_step_factory = main_fd.get_fd_train_step

        def step_factory(model, judges, sampling_args, options, tokenizer=None):
            step = original_step_factory(model, judges, sampling_args, options, tokenizer=tokenizer)

            def checked_step():
                loss, metrics = step()
                finite(loss)
                if not all(math.isfinite(v) for v in metrics.values()):
                    raise ValueError("non-finite training loss")
                nonzero = False
                for param in model.parameters():
                    if param.grad is not None:
                        finite(param.grad)
                        nonzero |= bool(torch.count_nonzero(param.grad))
                if not nonzero:
                    raise ValueError("no nonzero generator gradient")
                counts["steps"] += 1
                return loss, metrics

            return checked_step

        main_fd.get_fd_train_step = step_factory
        original_optimizer = main_fd.create_optimizer

        def create_optimizer(options, model, **kwargs):
            optimizer = original_optimizer(options, model, **kwargs)
            # A bounded sample across every parameter proves a real change;
            # the complete persisted state is also compared after training.
            before = []

            def before_step(opt, step_args, step_kwargs):
                if counts["optimizer"] == 0:
                    before.extend(p.detach().reshape(-1)[:16].clone() for p in model.parameters())

            def after_step(opt, step_args, step_kwargs):
                counts["optimizer"] += 1
                if counts["optimizer"] == 1:
                    state["parameter_update"] = any(
                        not torch.equal(old, p.detach().reshape(-1)[:16])
                        for old, p in zip(before, model.parameters()))
                    before.clear()

            optimizer.register_step_pre_hook(before_step)
            optimizer.register_step_post_hook(after_step)
            return optimizer

        main_fd.create_optimizer = create_optimizer
        result = main_fd.train_and_evaluate(args)
        if result != 0 or counts["steps"] != args.total_steps or counts["optimizer"] != args.total_steps:
            raise ValueError("original training did not complete every optimizer update")
        if not loaded or not state.get("parameter_update"):
            raise ValueError("model initialization/update was not established")
        checkpoint = Path(args.ckpt_dir) / f"step_{args.total_steps - 1:07d}.pth"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["current_step"] != args.total_steps or saved["samples_seen"] != args.total_steps * args.batch_size:
            raise ValueError("checkpoint training counters do not match")
        if not saved["optimizer"]["state"]:
            raise ValueError("optimizer state was not persisted")
        current = state["model"].state_dict()
        if saved["model"].keys() != current.keys() or any(
                not torch.equal(value.cpu(), saved["model"][key]) for key, value in current.items()):
            raise ValueError("persisted generation model differs from trained state")
        queues = {entry["name"]: entry["queue"] for entry in saved["fd_queue_states"]}
        for judge in state["judges"]:
            queue = judge["queue"]
            stored = queues[judge["name"]]
            actual = queue.state_dict()
            if actual.keys() != stored.keys() or any(
                    not torch.equal(v.cpu(), stored[k]) for k, v in actual.items()):
                raise ValueError("persisted queue state differs from trained state")
            restored = type(queue)(size=queue.size, feat_dim=queue.feat_dim,
                                   online_accum=queue.online_accum, ema_beta=queue.ema_beta)
            try:
                restored.load_state_dict(stored, strict=True)
                if any(not torch.equal(v, stored[k]) for k, v in restored.state_dict().items()):
                    raise ValueError("queue checkpoint round trip failed")
            finally:
                restored.close()
        if counts["features"] == 0 or counts["covariances"] == 0:
            raise ValueError("feature or covariance observation is missing")
        report.update(model_lock=True, finite_features=True, finite_covariances=True,
                      finite_gradients=True, parameter_update=True, queue_state_roundtrip=True,
                      authentic_warm_start=True, entrypoint_executed=True,
                      updates=counts["steps"], warm_start_images=counts["warm"])
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    else:
        import eval_all_fds as evaluation

        args = evaluation.get_args_parser().parse_args(argv)
        original_models = evaluation._load_repr_models
        feature_calls = [0]

        def models(*a, **kw):
            entries = original_models(*a, **kw)

            def check_features(module, inputs, outputs):
                finite(outputs)
                feature_calls[0] += 1

            seen = set()
            for entry in entries:
                model = entry["model"]
                if any(p.requires_grad for p in model.parameters()):
                    raise ValueError("held-out representation model is not frozen")
                if id(model) not in seen:
                    model.register_forward_hook(check_features)
                    seen.add(id(model))
            return entries

        evaluation._load_repr_models = models
        original_reduce = evaluation._reduce_and_compute
        observations = []

        def reduce(entries, accumulators, *a, **kw):
            for accumulator in accumulators:
                if accumulator["count"] != args.num_images:
                    raise ValueError("held-out accumulator count mismatch")
                total = accumulator["count"]
                feat_sum = accumulator["feat_sum"]
                outer = accumulator["feat_outer"]
                finite((feat_sum, outer, (outer - torch.outer(feat_sum, feat_sum) / total) / (total - 1)))
            result = original_reduce(entries, accumulators, *a, **kw)
            raw = result[0]
            mapping = {"FID(ADM)": "inception_fdr", "convnext": "convnext_fdr",
                       "dinov2_cls": "dinov2_fdr", "mae_cls": "mae_fdr",
                       "siglip_cls": "siglip_fdr", "clip_cls": "clip_fdr"}
            # Capture the original unrounded FD results before CSV formatting.
            observations.append({metric: float(raw[name]) / evaluation.FDR_VALIDATION_FD[name]
                                 for name, metric in mapping.items()})
            return result

        evaluation._reduce_and_compute = reduce
        evaluation.main_generate(args)
        if not loaded or feature_calls[0] == 0 or len(observations) != 1:
            raise ValueError("held-out original evaluation was not observed exactly once")
        report.update(model_lock=True, finite_features=True, finite_covariances=True,
                      metrics=observations[0], num_images=args.num_images)
    report["runtime"] = {
        "pid": os.getpid(), "uid": os.getuid(), "gid": os.getgid(),
        "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
        "device_count": torch.cuda.device_count(),
        "parent_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "parent_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "parent_rss_high_water_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    opts.report.write_text(json.dumps(report, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
