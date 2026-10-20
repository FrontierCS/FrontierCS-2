import deployment_test_guard  # noqa: F401
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / 'tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization/verifier'
SPEC = importlib.util.spec_from_file_location('safe_frames_test', VERIFIER / 'safe_error_frames.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def failure(filename, depth=1):
    namespace = {}
    code = "def trigger(n):\n    secret_local = 'PRIVATE_LOCAL_SENTINEL'\n    if n: return trigger(n-1)\n    raise ValueError('PRIVATE_MESSAGE_SENTINEL')\n"
    exec(compile(code, str(filename), 'exec'), namespace)
    try:
        namespace['trigger'](depth)
    except Exception as error:
        return MODULE.safe_error_frames(error)


class SafeErrorFramesTests(unittest.TestCase):
    def test_private_message_and_locals_are_absent(self):
        frames = failure(VERIFIER / 'verify.py')
        assert frames and set(frames[0]) == {'file', 'line', 'function'}
        assert all(item['file'] == 'verifier/verify.py' for item in frames)
        assert 'PRIVATE_' not in json.dumps(frames)


    def test_unknown_files_are_filtered(self):
        assert failure('/tmp/unknown-candidate.py') == []
        assert failure('<unknown>') == []


    def test_frame_count_is_bounded(self):
        assert len(failure(VERIFIER / 'verify.py', depth=40)) == 16
