"""The submitted policy cannot read judge files, and no process it starts survives.

Needs unprivileged user namespaces (as on the Modal verifier runners); skipped
where the host forbids them.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lrr_evaluator_isolation", ROOT / "evaluator.py")
ev = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(ev)


def _userns_available() -> bool:
    try:
        return subprocess.run(
            ["/usr/bin/unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork",
             "--mount-proc", "--kill-child", "true"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_userns = pytest.mark.skipif(not _userns_available(), reason="user namespaces unavailable")

HOOK = '''
import os, subprocess, sys
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
    ("evaluator", JUDGE + "/evaluator.py"), ("image_judge", "/opt/frontier/task/workloads.json")))
try:
    REPORT += ";judge_dir=" + ",".join(sorted(os.listdir(JUDGE)))
except OSError as error:
    REPORT += ";judge_dir=DENIED:" + type(error).__name__
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


@needs_userns
def test_policy_cannot_read_judge_and_children_die(tmp_path):
    marker = "lrr-isolation-" + uuid.uuid4().hex
    tree = tmp_path / "repo"
    shutil.copytree(ROOT / "harbor" / "app", tree)
    (tree / "scripts" / "scalpel_to_thinking_steps.py").write_text(
        f"JUDGE = {str(ROOT)!r}\nMARKER = {marker!r}\n" + HOOK, encoding="utf-8")
    item = json.loads((ROOT / "workloads.json").read_text())["development"][0]["items"][0]["input"]
    item = ev._policy_input(item)
    item["result"]["repairs"][0]["subpatterns"] = ["(a+)+"]
    rows = ev._render_chats(tree, "five_shot_localization", [json.dumps(item)])
    content = rows[0][0][0]["content"]
    for name in ("workloads", "reference", "evaluator", "image_judge"):
        assert f"{name}=DENIED" in content, content
    assert "judge_dir=policy" in content, content
    assert _survivors(marker) == []


def test_policy_input_keeps_only_pattern_and_first_localization():
    raw = {"pattern": "^(a+)+$", "error": None, "result": {
        "regex": "LEAK", "final_regex": "LEAK-REPAIRED",
        "repairs": [{"subpatterns": ["(a+)+"], "repaired": "LEAK", "nested_quantifier_result": "LEAK"},
                    {"subpatterns": ["LEAK-LATER"], "repaired": "LEAK"}]}}
    text = json.dumps(ev._policy_input(raw))
    assert "LEAK" not in text
    assert json.loads(text)["result"]["repairs"][0]["subpatterns"] == ["(a+)+"]
