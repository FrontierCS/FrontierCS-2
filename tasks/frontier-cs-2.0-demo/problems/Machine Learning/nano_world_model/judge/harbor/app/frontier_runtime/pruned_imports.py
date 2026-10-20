"""Satisfy eager upstream imports of omitted, unselected experiment families.

No selected implementation is replaced. Instantiating an omitted backend fails
explicitly; Point Maze, NanoWM, SD-VAE and diffusion use retained upstream code.
"""

import sys
from types import ModuleType

PRUNED = {
    "experiments.planning_experiment": ("PlanningExperiment",),
    "latent_codecs.semantic": ("VJEPA21LatentCodec", "WebDINOLatentCodec"),
    "wm_datasets.data_source.lerobot": ("LeRobotDataSource",),
    "wm_datasets.data_source.game": ("CSGODataSource",),
    "wm_datasets.data_source.dino_wm.pusht": ("PushTDataSource",),
    "wm_datasets.data_source.dino_wm.deformable": ("DeformableEnvDataSource",),
}


class UnselectedBackend:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("this backend is outside the Point Maze evaluation path")


def install():
    for name, symbols in PRUNED.items():
        module = ModuleType(name)
        for symbol in symbols:
            setattr(module, symbol, UnselectedBackend)
        sys.modules[name] = module
