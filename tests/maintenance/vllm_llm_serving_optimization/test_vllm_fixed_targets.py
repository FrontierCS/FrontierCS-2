"""Expose fixed-target CPU fixtures to the mandatory repository test suite."""
from vllm_task import REPO_ROOT  # noqa: E402
import deployment_test_guard  # noqa: F401
from pathlib import Path
import sys

JUDGE = REPO_ROOT / 'tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization/judge'
sys.path[:0] = [str(JUDGE), str(JUDGE.parent/'verifier')]
from test_fixed_targets import FixedTargetsTests  # noqa: E402,F401
