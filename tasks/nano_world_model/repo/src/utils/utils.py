import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist
import os
from einops import rearrange
from torch.nn import functional as F

def none_or_int(value):
    if value.lower() == 'none':
        return None
    return int(value)

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )

def resize_video(video_tensor, new_size):
    """
    Resize a tensor of shape (b, t, h, w, c) to (b, t, new_h, new_w, c).

    Parameters:
    tensor (torch.Tensor): Input tensor of shape (b, t, h, w, c)
    new_size (tuple): New size (new_h, new_w)

    Returns:
    torch.Tensor: Resized tensor of shape (b, t, new_h, new_w, c)
    """
    original_dtype = video_tensor.dtype
    video_tensor = video_tensor.float()
    b, t, _, _, _ = video_tensor.shape
    video_tensor = rearrange(video_tensor, "b t h w c -> (b t) c h w")

    # Apply the resizing transform using interpolate
    resized_tensor = F.interpolate(video_tensor, size=new_size, mode="bilinear", align_corners=False)
    resized_tensor = rearrange(resized_tensor, "(b t) c h w -> b t h w c", b=b, t=t)
    resized_tensor = resized_tensor.to(original_dtype)
    return resized_tensor