"""
Base DataSource definitions for world model datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
import torch


@dataclass
class TrajectoryData:
    """
    Unified representation of a single trajectory.

    Attributes:
        states: Full state tensor [max_T, state_dim], padded if necessary
        actions: Action tensor [max_T, action_dim], padded if necessary
        seq_length: Actual valid length of this trajectory (before padding)
        meta: Additional metadata (e.g., shapes, velocities, episode info)
    """
    states: torch.Tensor
    actions: torch.Tensor
    seq_length: int
    meta: Dict

    def __post_init__(self) -> None:
        """Validate trajectory data consistency."""
        assert self.states.shape[0] == self.actions.shape[0], \
            f"States and actions must have same length: {self.states.shape[0]} vs {self.actions.shape[0]}"
        assert self.seq_length <= self.states.shape[0], \
            f"seq_length {self.seq_length} exceeds tensor length {self.states.shape[0]}"


class DataSource(ABC):
    """
    Abstract base class for all data sources.
    """

    @abstractmethod
    def load_trajectory(self, index: int) -> TrajectoryData:
        """Load metadata for the trajectory at the given index."""
        raise NotImplementedError

    @abstractmethod
    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1
    ) -> torch.Tensor:
        """Load visual frames for a specific trajectory and time range."""
        raise NotImplementedError

    @abstractmethod
    def get_num_trajectories(self) -> int:
        """Return total number of trajectories in this data source."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action dimension."""
        raise NotImplementedError

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """State dimension."""
        raise NotImplementedError
