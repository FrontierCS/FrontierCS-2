"""Logger factory functions for PyTorch Lightning."""

import os
from pytorch_lightning.loggers import TensorBoardLogger


def create_tensorboard_logger(experiment_dir, name="nanowm"):
    """
    Create a TensorBoard logger.

    Args:
        experiment_dir: Root experiment directory
        name: Logger name (default: "nanowm")

    Returns:
        TensorBoardLogger instance
    """
    tb_dir = os.path.join(experiment_dir, "tb")
    # Ensure directory exists (safe with exist_ok=True in multi-process)
    os.makedirs(tb_dir, exist_ok=True)
    # Also create the version directory that TensorBoard expects
    version_dir = os.path.join(tb_dir, name)
    os.makedirs(version_dir, exist_ok=True)
    return TensorBoardLogger(tb_dir, name=name)


def create_wandb_logger(project, name, experiment_dir, entity=None, mode="online"):
    """
    Create a Weights & Biases logger (optional).

    Args:
        project: WandB project name
        name: Experiment name
        experiment_dir: Root experiment directory for saving
        entity: WandB entity (username/organization), optional
        mode: "online", "offline", or "disabled"

    Returns:
        WandbLogger instance or None if WandB not available or disabled
    """
    if mode == "disabled":
        return None

    from pytorch_lightning.loggers import WandbLogger
    return WandbLogger(
        project=project,
        name=name,
        save_dir=experiment_dir,
        entity=entity,
        offline=(mode == "offline")
    )
