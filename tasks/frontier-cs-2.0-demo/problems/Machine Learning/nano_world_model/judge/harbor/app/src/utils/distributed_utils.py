"""Distributed training utilities for rank-aware logging and operations."""

from pytorch_lightning.utilities.rank_zero import rank_zero_only

# Check if current process is rank 0
is_rank_zero = rank_zero_only.rank == 0


def rank_zero_print(*args, **kwargs):
    """Print function that only executes on rank 0."""
    if is_rank_zero:
        print(*args, **kwargs)


def rank_zero_log(func):
    """
    Decorator to ensure logging functions only execute on rank 0.

    Usage:
        @rank_zero_log
        def log_something(message):
            logger.info(message)
    """
    return rank_zero_only(func)
