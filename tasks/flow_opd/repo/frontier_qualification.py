"""Calibration-only probes of real transitions and persisted endpoint state."""
import hashlib
import inspect
import json
import math
import random


def state_digest(value):
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b'\x00')
    visit(value)
    return digest.hexdigest()


def rng_state():
    import numpy as np
    import torch
    return [random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all()]


def deterministic_slots(result, noise_level):
    """Change only unused zero-noise log slots; retain every state object."""
    import torch
    if noise_level != 0:
        return result
    state, logp, mean, coefficient = result[:4]
    if not bool((coefficient == 0).all()):
        raise ValueError('Nonzero coefficient in deterministic evaluation.')
    support = (state == mean).reshape(state.shape[0], -1).all(dim=1)
    sentinel = torch.where(support, torch.zeros_like(logp),
                           torch.full_like(logp, -torch.inf))
    return (state, sentinel, *result[2:])


class TransitionProbe:
    def __init__(self, runtime, step):
        self.runtime = runtime
        self.step = step
        self.signature = inspect.signature(step)
        self.seen = set()
        self.records = []
        self.normalized_slots = 0

    def __call__(self, /, *args, **kwargs):
        import torch
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        scheduler, model_output, timestep, sample = (
            values[name] for name in ('self', 'model_output', 'timestep', 'sample'))
        noise = values['noise_level']
        result = deterministic_slots(self.step(*bound.args, **bound.kwargs), noise)
        if noise == 0:
            self.normalized_slots += result[1].numel()
        # Probe each real sampling timestep once, independently of trainer replay.
        batch_timesteps = torch.as_tensor(timestep, device=sample.device).reshape(-1)
        if batch_timesteps.numel() == 1:
            batch_timesteps = batch_timesteps.expand(sample.shape[0])
        elif batch_timesteps.numel() != sample.shape[0]:
            raise ValueError('Timestep must be scalar or contain one value per sample.')
        key = (float(noise), tuple(float(t) for t in batch_timesteps))
        if bound.arguments['prev_sample'] is not None or key in self.seen:
            return result
        self.seen.add(key)
        state, logp, mean, coefficient = result[:4]
        if not all(torch.isfinite(v).all() for v in (sample, model_output, state, mean, coefficient)):
            raise ValueError('Nonfinite captured transition.')
        index = [scheduler.index_for_timestep(t) for t in batch_timesteps]
        dt = (scheduler.sigmas[index] - scheduler.sigmas[[i+1 for i in index]]).reshape(-1, 1, 1, 1)
        scale = coefficient * dt.sqrt()
        record = {'noise_level': noise, 'timesteps': list(key[1]),
                  'sample_is_cuda': sample.is_cuda, 'shape': list(sample.shape)}
        before = state_digest(rng_state())
        with torch.enable_grad():
            prediction = model_output.detach().clone().requires_grad_(noise != 0)
            replay = deterministic_slots(self.step(scheduler, prediction, timestep, sample.detach(),
                noise_level=noise, prev_sample=state.detach(), return_sqrt_dt=True), noise)
            record.update(replay_state_equal=torch.equal(replay[0], state),
                          replay_mean_max_abs=float((replay[2]-mean).abs().max()),
                          replay_logp_max_abs=float((replay[1]-logp).abs().max()),
                          replay_coefficient_equal=torch.equal(replay[3], coefficient),
                          sqrt_dt_max_abs=float((replay[4]-dt.sqrt()).abs().max()))
            if noise:
                if not bool((scale > 0).all()):
                    raise ValueError('Nonpositive sampled stochastic scale.')
                density = (-0.5*((state.detach().double()-mean.double())/scale.double()).square()
                           -scale.double().log()-0.5*math.log(2*math.pi)).mean(dim=(1,2,3))
                record['density_float64_max_abs'] = float((logp.double()-density).abs().max())
                record['logp_max_abs'] = float(logp.abs().max())
                gradients = [torch.autograd.grad(v.sum(), prediction, retain_graph=True)[0]
                             for v in (replay[1], replay[2])]
                record['gradient_finite_nonzero'] = [bool(torch.isfinite(g).all() and (g != 0).any()) for g in gradients]
            else:
                record['on_support_zero'] = bool((logp == 0).all() and torch.equal(state, mean))
                off = state.detach().clone()
                off.reshape(off.shape[0], -1)[:, 0] += 1
                off_result = deterministic_slots(self.step(scheduler, prediction, timestep, sample.detach(),
                    noise_level=0, prev_sample=off, return_sqrt_dt=True), 0)
                record['off_support_negative_infinity'] = bool(torch.isneginf(off_result[1]).all())
        record['replay_preserves_rng'] = before == state_digest(rng_state())
        if not record['replay_preserves_rng']:
            raise ValueError('Replay probe consumed randomness.')
        self.records.append(record)
        # The native metric validator permits at most six nested levels.
        self.runtime.result['transition_probe_records'] = self.records
        self.runtime.result['zero_noise_slots_normalized'] = self.normalized_slots
        self.runtime.result['density_tolerance_frozen'] = False
        return result


def verify_checkpoint(runtime, local):
    import numpy as np
    import torch
    accelerator = local['accelerator']
    params = local['transformer_trainable_parameters']
    optimizer = local['optimizer']
    before = {'parameters': state_digest(params), 'optimizer': state_digest(optimizer.state_dict()),
              'rng': state_digest(rng_state()), 'ema': state_digest(local['ema'].state_dict())}
    # Deliberately perturb state so a no-op loader cannot pass this diagnostic.
    with torch.no_grad():
        params[0].reshape(-1)[0] += 1
        for state in optimizer.state.values():
            if 'exp_avg' in state:
                state['exp_avg'].reshape(-1)[0] += 1
                break
        local['ema'].ema_parameters[0].reshape(-1)[0] += 1
    torch.rand(1); torch.rand(1, device=accelerator.device); random.random(); np.random.rand()
    accelerator.load_state(str(runtime.output/'state'))
    endpoint = torch.load(runtime.output/'state/endpoint.pt', map_location=accelerator.device, weights_only=False)
    local['ema'].load_state_dict(endpoint['ema'])
    after = {'parameters': state_digest(params), 'optimizer': state_digest(optimizer.state_dict()),
             'rng': state_digest(rng_state()), 'ema': state_digest(local['ema'].state_dict())}
    receipt = {'restored': {key: before[key] == after[key] for key in before},
               'epoch': endpoint['epoch'], 'global_step': endpoint['global_step'],
               'ema_update_step_interval': endpoint['ema_update_step_interval'],
               'sampler_continuation_verified': False, 'lr_scheduler': 'none in retained trainer'}
    runtime.result['checkpoint_reload'] = receipt
    if not all(receipt['restored'].values()):
        raise ValueError('Persisted endpoint failed actual state restoration.')


def reload_ema_adapter(runtime, local):
    model = local['accelerator'].unwrap_model(local['transformer'])
    before_base = {name: value for name, value in model.named_parameters() if '.lora_' not in name}
    if state_digest(before_base) != runtime.initial_base_digest:
        raise ValueError('Frozen base differs from its initialized snapshot.')
    path = runtime.output/'checkpoints'/('checkpoint-'+str(local['global_step']))/'lora'
    model.load_adapter(str(path), adapter_name='frontier_reload', is_trainable=False)
    model.set_adapter('frontier_reload')
    original_names = [name for name, value in model.named_parameters() if '.default.' in name]
    loaded = dict(model.named_parameters())
    equal = len(original_names) == len(local['ema'].ema_parameters) and all(
        runtime.torch.equal(loaded[name.replace('.default.', '.frontier_reload.')], expected)
        for name, expected in zip(original_names, local['ema'].ema_parameters))
    runtime.result['ema_adapter_reload'] = {'same_pinned_base_unchanged': True,
        'all_ema_parameters_equal': equal, 'parameter_count': len(original_names),
        'reload_path': 'PEFT load_adapter onto the unchanged exact base',
        'decoded_images_equal': None, 'ocr_equal': None}
    if not equal:
        raise ValueError('Saved adapter differs from endpoint EMA.')
    return model
