"""Draft numerical guards for real captured SD3 transitions, never a score.

No model, latent or trajectory is synthesized here. Invoke only on captured
current-student inputs on the Driver Verifier GPU. Guard tolerances must be
calibrated and frozen before use; this module supplies no guessed tolerances.
These checks complement, and do not replace, independent execution isolation,
model-forward provenance and saved-checkpoint verification.
"""
import math


def check_transition(step, scheduler, model_output, timestep, sample, observed,
                     *, noise_level, tolerance, require_gradient):
    import torch
    if not sample.is_cuda or not model_output.is_cuda:
        raise ValueError('Transition evidence must come from the GPU path.')
    if sample.ndim!=4 or sample.shape[0]<1 or len(observed) not in (4,5):
        raise ValueError('Invalid captured transition shape.')
    next_state,logp,mean,coefficient=observed[:4]
    if next_state.shape!=sample.shape or mean.shape!=sample.shape or logp.shape!=(sample.shape[0],):
        raise ValueError('Invalid transition output shape.')
    if not all(torch.isfinite(v).all() for v in (sample,model_output,next_state,mean,coefficient)):
        raise ValueError('Nonfinite transition state.')
    if not (coefficient>=0).all():
        raise ValueError('Negative diffusion coefficient.')
    batch_timesteps = torch.as_tensor(timestep, device=sample.device).reshape(-1)
    if batch_timesteps.numel() == 1:
        batch_timesteps = batch_timesteps.expand(sample.shape[0])
    elif batch_timesteps.numel() != sample.shape[0]:
        raise ValueError('Timestep must be scalar or contain one value per sample.')
    indices=[scheduler.index_for_timestep(t) for t in batch_timesteps]
    sigma=scheduler.sigmas[indices].reshape(-1,1,1,1)
    sigma_next=scheduler.sigmas[[i+1 for i in indices]].reshape(-1,1,1,1)
    sqrt_dt=(sigma-sigma_next).sqrt()
    if not torch.isfinite(sqrt_dt).all(): raise ValueError('Invalid scheduler time interval.')
    if len(observed)==5 and not torch.allclose(observed[4],sqrt_dt,**tolerance):
        raise ValueError('Time-factor convention changed.')
    # Fixed retained KL wiring requires a Gaussian transition covariance compatible
    # with this coefficient. The mean/integration/exploration schedule is free.
    scale=coefficient*sqrt_dt
    all_zero=bool((scale==0).all())
    if noise_level==0 and not all_zero:
        raise ValueError('Noise introduced into deterministic evaluation.')
    if all_zero:
        support=(next_state==mean).reshape(sample.shape[0],-1).all(dim=1)
        expected=torch.where(support,torch.zeros_like(logp),torch.full_like(logp,-torch.inf))
        if not torch.equal(logp,expected):
            raise ValueError('Invalid deterministic log-mass sentinel.')
        if not support.all(): raise ValueError('Sampled deterministic transition is off support.')
    else:
        if not (scale>0).all():
            raise ValueError('Mixed zero and positive transition variance requires explicit qualification.')
        expected=(-0.5*((next_state.detach()-mean)/scale).square()-scale.log()-0.5*math.log(2*math.pi)).mean(dim=(1,2,3))
        if not torch.isfinite(logp).all() or not torch.allclose(logp,expected,**tolerance):
            raise ValueError('Transition density or normalization inconsistent with returned law.')
    with torch.enable_grad():
        replay_output=model_output.detach().clone().requires_grad_(require_gradient)
        cpu_rng=torch.get_rng_state().clone()
        gpu_rng=torch.cuda.get_rng_state(sample.device).clone()
        replay=step(scheduler,replay_output,timestep,sample.detach(),noise_level=noise_level,
                    prev_sample=next_state.detach(),return_sqrt_dt=True)
        if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(gpu_rng,torch.cuda.get_rng_state(sample.device)):
            raise ValueError('Scoring supplied state consumed sampling randomness.')
        if not torch.equal(replay[0],next_state):
            raise ValueError('Replay resampled supplied next state.')
        if not torch.allclose(replay[1],logp,**tolerance) or not torch.allclose(replay[2],mean,**tolerance):
            raise ValueError('Rollout and unchanged-parameter replay disagree.')
        if not torch.allclose(replay[3],coefficient,**tolerance) or not torch.allclose(replay[4],sqrt_dt,**tolerance):
            raise ValueError('Replay noise/time convention differs.')
        if require_gradient:
            if all_zero: raise ValueError('Zero-variance transition must not enter training.')
            for value in (replay[1],replay[2]):
                if not value.requires_grad:
                    raise ValueError('Replay lost model-output gradients.')
                gradient=torch.autograd.grad(value.sum(),replay_output,retain_graph=True,allow_unused=True)[0]
                if gradient is None or not torch.isfinite(gradient).all():
                    raise ValueError('Replay derivative is missing or nonfinite.')
    return {'density_valid':True,'replay_valid':True,'gradient_checked':bool(require_gradient)}
