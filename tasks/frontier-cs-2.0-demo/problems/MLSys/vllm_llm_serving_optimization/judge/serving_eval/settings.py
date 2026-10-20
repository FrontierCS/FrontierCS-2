"""Effective original conversation knobs, independent of cloud configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeSettings:
    model: str = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
    step_limit: int = 50
    temperature: float = 0.0
    max_completion_tokens: int = 2048
    bfcl_max_tokens: int = 768
    memory_max_steps: int = 20
