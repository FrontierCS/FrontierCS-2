"""Expose fixed-target CPU fixtures to the mandatory repository test suite."""

import deployment_test_guard  # noqa: F401

import sys

from vllm_task import REPO_ROOT  # noqa: E402

JUDGE = REPO_ROOT / "tasks/problems/MLSys/vllm_llm_serving_optimization/judge"
sys.path[:0] = [str(JUDGE), str(JUDGE.parent / "verifier")]
from test_fixed_targets import FixedTargetsTests  # noqa: E402,F401
