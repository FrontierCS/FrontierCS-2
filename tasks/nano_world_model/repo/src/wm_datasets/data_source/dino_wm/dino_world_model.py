"""
DINO World Model file-based data source.
"""

import pickle
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import torch
from einops import rearrange

import decord
from decord import VideoReader

from ..base import DataSource, TrajectoryData

decord.bridge.set_bridge("torch")


class DinoWorldModelDataSource(DataSource):
    """
    Data source for DINO-WM datasets stored in the file system.

    Expected directory structure:
        data_path/
        ├── states.pth          # [N, max_T, state_dim]
        ├── actions.pth         # [N, max_T, action_dim]
        ├── seq_lengths.pth     # [N] or [N,] tensor or list
        ├── velocities.pth      # [N, max_T, vel_dim] (optional)
        ├── shapes.pkl          # List of shapes (optional, for PushT)
        └── obses/
            ├── episode_000.pth  # [T, H, W, C] tensor (PointMaze, Wall)
            ├── episode_000.mp4  # Video file (PushT)
            └── ...

    Args:
        data_path: Path to the dataset directory
        video_format: Format of visual data, either "pth" or "mp4"
        action_scale: Scale factor for actions (will be divided by this)
        n_rollout: Limit number of trajectories to load (None = load all)
        use_relative_actions: For PushT: use rel_actions.pth vs abs_actions.pth
    """

    def __init__(
        self,
        data_path: str,
        video_format: Literal["pth", "mp4"] = "pth",
        action_scale: float = 1.0,
        n_rollout: Optional[int] = None,
        use_relative_actions: bool = True,
    ):
        self.data_path = Path(data_path)
        self.video_format = video_format
        self.action_scale = action_scale

        # Load trajectory data into memory (typically small)
        print(f"Loading trajectory data from {self.data_path}...")
        self.states = torch.load(self.data_path / "states.pth").float()

        # Load actions (handle different naming conventions)
        actions_file = self.data_path / "actions.pth"
        if not actions_file.exists():
            if use_relative_actions:
                actions_file = self.data_path / "rel_actions.pth"
            else:
                actions_file = self.data_path / "abs_actions.pth"

        self.actions = torch.load(actions_file).float()

        # Handle different seq_lengths formats
        seq_lengths_pth = self.data_path / "seq_lengths.pth"
        seq_lengths_pkl = self.data_path / "seq_lengths.pkl"

        if seq_lengths_pth.exists():
            seq_lengths = torch.load(seq_lengths_pth)
            if isinstance(seq_lengths, list):
                self.seq_lengths = torch.tensor(seq_lengths)
            else:
                self.seq_lengths = seq_lengths
        elif seq_lengths_pkl.exists():
            with open(seq_lengths_pkl, "rb") as f:
                seq_lengths = pickle.load(f)
            if isinstance(seq_lengths, list):
                self.seq_lengths = torch.tensor(seq_lengths)
            else:
                self.seq_lengths = seq_lengths
        else:
            max_length = self.states.shape[1]
            self.seq_lengths = torch.full((len(self.states),), max_length, dtype=torch.long)
            print(f"  Warning: No seq_lengths file found, assuming all trajectories have length {max_length}")

        # Apply action scaling
        if action_scale != 1.0:
            self.actions = self.actions / action_scale

        # Load optional data
        self.velocities = None
        velocities_file = self.data_path / "velocities.pth"
        if velocities_file.exists():
            self.velocities = torch.load(velocities_file).float()

        self.shapes = None
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, "rb") as f:
                self.shapes = pickle.load(f)

        # Limit number of trajectories if requested
        if n_rollout is not None:
            n = min(n_rollout, len(self.states))
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]

        if self.velocities is not None:
            self.velocities = self.velocities[:n]
        if self.shapes is not None:
            self.shapes = self.shapes[:n]

        self.num_trajectories = n
        self._action_dim = self.actions.shape[-1]
        self._state_dim = self.states.shape[-1]

        print(f"Loaded {n} trajectories from {self.data_path}")
        print(f"  State dim: {self._state_dim}, Action dim: {self._action_dim}")

    def load_trajectory(self, index: int) -> TrajectoryData:
        """Load trajectory metadata (no visual data)."""
        if index >= self.num_trajectories:
            raise IndexError(f"Index {index} out of range [0, {self.num_trajectories})")

        seq_len = self.seq_lengths[index].item()

        meta: Dict = {}
        if self.velocities is not None:
            meta["velocity"] = self.velocities[index]
        if self.shapes is not None:
            meta["shape"] = self.shapes[index]

        return TrajectoryData(
            states=self.states[index],
            actions=self.actions[index],
            seq_length=seq_len,
            meta=meta,
        )

    def load_visual_frames(
        self,
        index: int,
        start: int,
        end: int,
        step: int = 1
    ) -> torch.Tensor:
        """
        Load visual frames from disk.

        Returns:
            Frames tensor [T, C, H, W], normalized to [0, 1], float32
        """
        obs_dir = self.data_path / "obses"

        if self.video_format == "pth":
            video_file = obs_dir / f"episode_{index:03d}.pth"

            if not video_file.exists():
                raise FileNotFoundError(
                    f"Visual frames not found: {video_file}\n"
                    f"Trajectory {index} exists in metadata but has no corresponding video file."
                )

            full_video = torch.load(video_file, weights_only=False, map_location="cpu")

            if isinstance(full_video, np.ndarray):
                full_video = torch.from_numpy(full_video)

            frames = full_video[start:end:step].contiguous()
            del full_video

            if not isinstance(frames, torch.Tensor):
                frames = torch.from_numpy(frames)
            frames = frames.float() / 255.0
            if frames.ndim == 4 and frames.shape[-1] in [1, 3]:
                frames = rearrange(frames, "T H W C -> T C H W")

        elif self.video_format == "mp4":
            video_file = obs_dir / f"episode_{index:03d}.mp4"

            if not video_file.exists():
                raise FileNotFoundError(
                    f"Visual frames not found: {video_file}\n"
                    f"Trajectory {index} exists in metadata but has no corresponding video file."
                )

            reader = VideoReader(str(video_file), num_threads=1)
            frame_indices = list(range(start, end, step))
            frames = reader.get_batch(frame_indices)

            if not isinstance(frames, torch.Tensor):
                if hasattr(frames, "asnumpy"):
                    frames = frames.asnumpy()
                frames = torch.from_numpy(frames)
            frames = frames.float() / 255.0
            frames = rearrange(frames, "T H W C -> T C H W")

        else:
            raise ValueError(f"Unsupported video format: {self.video_format}")

        return frames

    def get_seq_length(self, index: int) -> int:
        return int(self.seq_lengths[index].item())

    def get_num_trajectories(self) -> int:
        return self.num_trajectories

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim
