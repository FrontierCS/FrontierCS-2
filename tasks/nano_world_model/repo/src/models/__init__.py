import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from latent_codecs import get_model_latent_channels, get_model_latent_size
from .nanowm import NanoWM_models

from torch.optim.lr_scheduler import LambdaLR


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    """Build the world-model backbone from the fully resolved config.

    Legacy run configs did not record `model.latent_channels`; defaulting it to
    4 keeps existing SD-VAE checkpoints loadable while new codecs can make the
    channel count explicit.
    """
    if 'NanoWM' not in args.model.arch:
        raise ValueError(f"{args.model.arch} Model Not Supported!")

    action_dim = args.dataset.spec.action_dim * args.dataset.frame_interval
    return NanoWM_models[args.model.arch](
        input_size=get_model_latent_size(args),
        in_channels=get_model_latent_channels(args),
        num_classes=args.model.num_classes,
        num_frames=args.model.num_frames,
        extras=args.model.extras,
        use_action=args.model.use_action,
        action_dim=action_dim,
        action_injection_type=args.model.action_injection.type,
        causal=args.model.causal,
    )
