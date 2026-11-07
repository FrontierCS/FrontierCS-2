from __future__ import annotations

from typing import Iterable


class BaseExperiment:
    def __init__(self, cfg):
        self.cfg = cfg

        # No default value - require explicit config
        if not hasattr(cfg, "experiment") or not hasattr(cfg.experiment, "tasks"):
            raise ValueError("Missing required config: experiment.tasks")
        self.tasks: Iterable[str] = cfg.experiment.tasks

    def exec(self) -> None:
        if not self.tasks:
            raise ValueError("No tasks specified in config (experiment.tasks is empty)")
        for task in self.tasks:
            if not hasattr(self, task):
                raise ValueError(f"Task '{task}' not implemented for {self.__class__.__name__}")
            getattr(self, task)()
