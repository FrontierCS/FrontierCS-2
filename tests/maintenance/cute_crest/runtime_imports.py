"""Load task modules without leaking generic names into repository test imports."""
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Software Engineering/cute_crest"
REPO_TREE = TASK / "repo"
sys.path.insert(0, str(ROOT / "scripts"))
import task_manifest  # noqa: E402

# The judge as the Verifier image holds it: judge/ with the starter at pristine/.
JUDGE = task_manifest.judge_bundle(TASK)
SHARED_VERIFIER = task_manifest.SHARED / "verifier"
NAMES = ("evaluator", "gate", "beat_spec_feedback", "fcs2_verifier",
         "solution_policy", "crest_runner", "frontier_build")


def load_runtime():
    old_path = list(sys.path)
    saved = {name: sys.modules.get(name) for name in NAMES}
    try:
        for name in NAMES:
            sys.modules.pop(name, None)
        sys.path[:0] = [str(JUDGE), str(SHARED_VERIFIER), str(task_manifest.SHARED / "lib"),
                        str(REPO_TREE)]
        return {name: importlib.import_module(name) for name in NAMES}
    finally:
        sys.path[:] = old_path
        for name in NAMES:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


MODULES = load_runtime()
