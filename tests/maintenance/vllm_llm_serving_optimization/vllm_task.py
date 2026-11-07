"""Where the vllm_llm_serving_optimization sources live, for its maintenance tests: the judge,
the task's standalone Verifier and the shared verifier modules its Verifier base provides."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = REPO_ROOT / "tasks/problems/MLSys/vllm_llm_serving_optimization"
for path in (REPO_ROOT / "tests", REPO_ROOT / "tasks/shared/verifier", TASK_ROOT / "verifier",
             TASK_ROOT / "judge", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
