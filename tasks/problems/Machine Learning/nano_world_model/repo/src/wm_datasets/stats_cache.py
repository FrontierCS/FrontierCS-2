"""On-disk cache + data-source fallback for action/state normalization stats.

Factored out of WorldModelDataset so the storage key scheme and the LeRobot
source-stats handshake stay in one place.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch

from .data_source import DataSource


def _identity(ds: DataSource) -> Optional[str]:
    for attr in ("repo_id", "data_path", "root"):
        val = getattr(ds, attr, None)
        if val:
            return f"{attr}={val}"
    return None


def _storage_root(ds: DataSource) -> Path:
    for attr in ("root", "data_path"):
        val = getattr(ds, attr, None)
        if not val:
            continue
        try:
            candidate = Path(val)
        except (TypeError, ValueError):
            continue
        if candidate.exists() and candidate.is_dir():
            return candidate
    return Path.home() / ".cache" / "nano-world-model"


def cache_path(
    ds: DataSource,
    kind: str,
    split: str,
    traj_indices: Iterable[int],
    num_trajectories: int,
    dim: int,
) -> Optional[Path]:
    identity = _identity(ds)
    if identity is None:
        return None

    indices_tuple = tuple(sorted(int(i) for i in traj_indices))
    payload = {
        "id": identity,
        "kind": kind,
        "split": split,
        "num_traj_total": int(num_trajectories),
        "num_traj_split": len(indices_tuple),
        "indices_hash": hashlib.sha1(repr(indices_tuple).encode()).hexdigest(),
        "dim": int(dim),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return _storage_root(ds) / "wm_stats_cache" / f"{kind}_{split}_{digest}.pt"


def load(path: Optional[Path]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if path is None or not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Warning: failed to load stats cache {path}: {e}; recomputing")
        return None
    if not isinstance(payload, dict) or "mean" not in payload or "std" not in payload:
        print(f"Warning: stats cache {path} has unexpected format; recomputing")
        return None
    return payload["mean"].float(), payload["std"].float()


def save(path: Optional[Path], mean: torch.Tensor, std: torch.Tensor) -> None:
    # Best-effort: a cache write failure must never block training.
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean.detach().cpu(), "std": std.detach().cpu()}, path)
    except Exception as e:
        print(f"Warning: failed to write stats cache to {path}: {e}")


def try_source_stats(
    ds: DataSource, kind: str
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Read (mean, std) for `kind` out of `ds.stats`, or (None, None).

    std is returned WITHOUT the +1e-6 epsilon; callers add it.
    """
    ds_stats = getattr(ds, "stats", None)
    if not isinstance(ds_stats, dict):
        return None, None
    mean = ds_stats.get(f"{kind}_mean")
    std = ds_stats.get(f"{kind}_std")
    if mean is None or std is None:
        return None, None
    try:
        return torch.as_tensor(mean).float(), torch.as_tensor(std).float()
    except Exception as e:
        print(f"Warning: could not coerce data source {kind} stats to tensors: {e}")
        return None, None
