"""Fixed offline placement; package acquisition and image construction are external."""

from pathlib import Path

from task_guards import TaskIntegrityError


def validate_layout(runtime, declaration):
    for field in ("prefix", "checkpoint", "checkpoint_sha256", "ema_sha256"):
        if runtime.get(field) != declaration[field]:
            raise TaskIntegrityError("Measurement runtime differs from its declared placement")
    if not Path(runtime["prefix"]).is_absolute() or not Path(runtime["checkpoint"]).is_absolute():
        raise TaskIntegrityError("Measurement runtime paths must be absolute")
