from .train_experiment import TrainExperiment
from .planning_experiment import PlanningExperiment


def build_experiment(cfg):
    name = getattr(cfg.experiment, "name", None) if hasattr(cfg, "experiment") else None
    if name is None:
        raise ValueError("Missing experiment.name in config. Set configs/experiment/*.yaml")

    if name in ("train", "evaluate"):
        return TrainExperiment(cfg)
    if name == "planning":
        return PlanningExperiment(cfg)

    raise ValueError(f"Unknown experiment_name: {name}")


__all__ = [
    "build_experiment",
    "TrainExperiment",
    "PlanningExperiment",
]
