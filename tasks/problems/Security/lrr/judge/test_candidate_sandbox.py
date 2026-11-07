"""Escape PoC: the submitted prompt constructor runs under the shared isolate launcher and reaches none
of the judge's state.

A malicious scalpel_to_thinking_steps.py tries to read the judge (workloads, reference patch,
evaluator), the FRONTIER_* environment, shared /tmp state from an earlier run, and leaves a
double-forked daemon; it reports what it saw through the rendered chat. Locally the landlock
backend serves; the GPU runner builds userns (same boundary, tests/test_isolate.py).
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parent
SHARED_LIB = ROOT.parents[4] / "shared" / "lib" if len(ROOT.parents) > 4 else None
if SHARED_LIB is not None and SHARED_LIB.is_dir():
    sys.path.insert(0, str(SHARED_LIB))
spec = importlib.util.spec_from_file_location("lrr_evaluator_sandbox", ROOT / "evaluator.py")
ev = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(ev)
# The starter: pristine/ in the image, repo/ next to judge/ in the task source.
PRISTINE = ROOT / "pristine" if (ROOT / "pristine").is_dir() else ROOT.parent / "repo"
ev.PRISTINE = PRISTINE

needs_sandbox = pytest.mark.skipif(not ev.isolate.available(),
                                   reason="needs an isolate backend")

HOOK = '''
import os, sys
from parse_regex_scalpel import Result as ScalpelResult

def _probe(path):
    try:
        with open(path, "rb") as stream:
            stream.read(1)
        return "OPEN"
    except OSError as error:
        return "DENIED:" + type(error).__name__

REPORT = ";".join(name + "=" + _probe(path) for name, path in (
    ("workloads", JUDGE + "/workloads.json"), ("reference", JUDGE + "/reference.patch"),
    ("evaluator", JUDGE + "/evaluator.py"), ("image_judge", "/opt/frontier/task/workloads.json"),
    ("planted", PLANTED)))
try:
    REPORT += ";judge_dir=" + ",".join(sorted(os.listdir(JUDGE)))
except OSError as error:
    REPORT += ";judge_dir=DENIED:" + type(error).__name__
REPORT += ";frontier_env=" + ",".join(sorted(k for k in os.environ if k.startswith("FRONTIER_")))
for name in ("/tmp/" + MARKER, os.environ.get("TMPDIR", "/tmp") + "/" + MARKER):
    REPORT += ";left_" + name.rsplit("/", 2)[-2] + "=" + _probe(name)
    try:
        with open(name, "w") as stream:
            stream.write("state")
    except OSError:
        pass
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        os.execv(sys.executable, [sys.executable, "-c", "import time; time.sleep(600)", MARKER])
    os._exit(0)
os.wait()


def subpattern_to_string(subpattern: str) -> str:
    return REPORT


def generate_cot_prompt(result: ScalpelResult) -> tuple[str, list[str]]:
    return "", []
'''


def _survivors(marker: str) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if marker.encode() in (entry / "cmdline").read_bytes():
                    found.append(int(entry.name))
            except OSError:
                pass
    return found


@needs_sandbox
def test_constructor_cannot_reach_judge_state(tmp_path, monkeypatch):
    marker = "lrr-escape-" + uuid.uuid4().hex
    planted = tmp_path / "judge-scratch-secret"
    planted.write_text("hidden")
    monkeypatch.setenv("FRONTIER_SUBMISSION_ROLE", "final")
    monkeypatch.setenv("FRONTIER_DRIVER_TOKEN", "secret")
    tree = tmp_path / "repo"
    shutil.copytree(PRISTINE, tree)
    (tree / "scripts" / "scalpel_to_thinking_steps.py").write_text(
        f"JUDGE = {str(ROOT)!r}\nMARKER = {marker!r}\nPLANTED = {str(planted)!r}\n" + HOOK,
        encoding="utf-8")
    item = json.loads((ROOT / "workloads.json").read_text())["development"][0]["items"][0]["input"]
    item = ev._policy_input(item)
    item["result"]["repairs"][0]["subpatterns"] = ["(a+)+"]
    try:
        reports = []
        for _ in range(2):  # the second run must not see the first run's /tmp or TMPDIR files
            rows = ev._render_chats(tree, "five_shot_localization", [json.dumps(item)])
            reports.append(rows[0][0][0]["content"])
        for content in reports:
            for name in ("workloads", "reference", "evaluator", "image_judge", "planted"):
                assert f"{name}=DENIED" in content, content
            assert "judge_dir=DENIED" in content, content
            assert "frontier_env=;" in content, content
            assert "=OPEN" not in content, content
        assert not Path("/tmp", marker).exists()
        assert _survivors(marker) == []
    finally:
        Path("/tmp", marker).unlink(missing_ok=True)


@needs_sandbox
def test_reference_constructor_renders_in_the_sandbox():
    tree = ev._reference_tree()
    try:
        workload = json.loads(ev.DATA.read_text())["development"][0]
        lines = [json.dumps(ev._policy_input(item["input"])) for item in workload["items"]]
        rows = ev._render_chats(tree, "five_shot_localization", lines)
    finally:
        shutil.rmtree(tree.parent, ignore_errors=True)
    assert len(rows) == len(lines)
    for row, item in zip(rows, workload["items"]):
        assert row is not None and row[1] == item["input"]["pattern"]
        assert row[0] and all(set(m) == {"role", "content"} for m in row[0])
