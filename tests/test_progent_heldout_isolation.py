"""Progent: the agent image must not ship the held-out final AgentDojo user tasks.

The judge keeps the full AgentDojo suites (judge/harbor/app); the agent-visible
copy (agent-base/frontier/task/repo) drops every user task in the final split of
judge/workloads.py, and every combined user task built from one of them.
"""
import deployment_test_guard  # noqa: F401 # isort: skip

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / (
    "tasks/frontier-cs-2.0-demo/problems/Security/progent_privilege_control")
AGENT = TASK / "agent-base"
AGENT_REPO = AGENT / "frontier/task/repo"
JUDGE_REPO = TASK / "judge/harbor/app"
SUITES_REL = Path("agentdojo/src/agentdojo/default_suites")
DATA_REL = Path("agentdojo/src/agentdojo/data/suites")
MIN_LITERAL = 16


def _load_workloads():
    spec = importlib.util.spec_from_file_location("progent_workloads", TASK / "judge/workloads.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workloads = _load_workloads()
FULL = json.loads((TASK / "judge/suite_tasks.json").read_text())


def _num(task_id):
    return int(task_id.rsplit("_", 1)[1])


def _user_task_files(repo, suite):
    return sorted((repo / SUITES_REL).glob(f"v1*/{suite}/user_tasks.py"))


def _combinator(node):
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr in ("create_combined_task", "update_combined_task")):
        return None
    call = node.value
    new = int(re.fullmatch(r"UserTask(\d+)", call.args[0].value).group(1))
    parts = [int(re.search(r"user_task_(\d+)", ast.unparse(arg)).group(1)) for arg in call.args[1:3]]
    return node.value.func.attr, new, parts


def _node_id(node):
    if isinstance(node, ast.ClassDef):
        match = re.fullmatch(r"UserTask(\d+)", node.name)
        return int(match.group(1)) if match else None
    combined = _combinator(node)
    return combined[1] if combined else None


def _held_out(suite):
    _, final = workloads.split(suite, FULL[suite]["user"])
    return {_num(t) for t in final}


def _removed(suite):
    held = _held_out(suite)
    removed = set(held)
    for path in _user_task_files(JUDGE_REPO, suite):
        for node in ast.parse(path.read_text()).body:
            combined = _combinator(node)
            if combined and (combined[1] in held or set(combined[2]) & held):
                removed.add(combined[1])
    return removed


def _agent_dev(suite):
    train, _ = workloads.split(suite, FULL[suite]["user"])
    removed = _removed(suite)
    return [t for t in train if _num(t) not in removed]


def _literals(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.value


def _judge_split_literals():
    """(per-suite literals of removed task definitions, text of the rest of the judge tree).

    A literal shared with the judge's tree minus the removed definitions (a tool
    name, an environment value, an injection task's argument) is not a leak signal.
    """
    removed = {suite: _removed(suite) for suite in workloads.SUITES}
    needles = {suite: set() for suite in workloads.SUITES}
    kept = []
    for path in sorted(JUDGE_REPO.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        suite = path.parent.name
        if path.name != "user_tasks.py" or suite not in removed:
            kept.append(text)
            continue
        for node in ast.parse(text).body:
            if _node_id(node) in removed[suite]:
                needles[suite].update(s for s in _literals(node) if len(s) >= MIN_LITERAL)
            else:
                kept.append(ast.get_source_segment(text, node, padded=True))
    return needles, "\n".join(kept)


def _agent_files():
    return {p: p.read_text(errors="replace") for p in sorted(AGENT.rglob("*")) if p.is_file()}


class ProgentHeldoutIsolationTests(unittest.TestCase):
    def test_split_is_the_published_one(self):
        expected = {
            "banking": [1, 2, 3, 4, 6, 10, 11],
            "slack": [1, 3, 8, 10, 11, 12, 14, 15, 18],
            "travel": [0, 4, 5, 7, 8, 12, 14, 18],
            "workspace": [0, 2, 9, 12, 15, 16, 18, 19, 21, 22, 24, 26, 27, 31, 32, 33],
        }
        for suite in workloads.SUITES:
            self.assertEqual(sorted(_held_out(suite)), expected[suite], suite)
            self.assertEqual(FULL[suite]["user"], [f"user_task_{i}" for i in range(len(FULL[suite]["user"]))])

    def test_agent_suite_tasks_lists_only_runnable_dev_tasks(self):
        agent = json.loads((AGENT / "frontier/task/harness/suite_tasks.json").read_text())
        self.assertEqual(list(agent), list(FULL))
        for suite in workloads.SUITES:
            self.assertEqual(agent[suite]["user"], _agent_dev(suite), suite)
            self.assertFalse({_num(t) for t in agent[suite]["user"]} & _removed(suite))
            self.assertEqual(agent[suite]["injection"], FULL[suite]["injection"], suite)

    def test_agent_user_task_modules_register_exactly_the_dev_tasks(self):
        for suite in workloads.SUITES:
            dev = {_num(t) for t in _agent_dev(suite)}
            registered, updated = set(), set()
            for path in _user_task_files(AGENT_REPO, suite):
                judge_nodes = {ast.dump(n) for n in ast.parse(
                    (JUDGE_REPO / path.relative_to(AGENT_REPO)).read_text()).body}
                for node in ast.parse(path.read_text()).body:
                    # Everything kept is byte-for-byte the judge's definition.
                    self.assertIn(ast.dump(node), judge_nodes, f"{path}: {ast.unparse(node)[:80]}")
                    ident = _node_id(node)
                    if ident is None:
                        continue
                    if isinstance(node, ast.ClassDef):
                        decorators = [ast.unparse(d) for d in node.decorator_list]
                        target = registered if "task_suite.register_user_task" in decorators else updated
                    else:
                        target = registered if _combinator(node)[0] == "create_combined_task" else updated
                    target.add(ident)
                    if _combinator(node):
                        self.assertTrue(set(_combinator(node)[2]) <= dev, f"{path}: {ident}")
            self.assertEqual(registered, dev, suite)
            self.assertTrue(updated <= dev, suite)

    def test_only_user_task_modules_and_harness_split_differ_from_judge(self):
        agent = {p.relative_to(AGENT_REPO) for p in AGENT_REPO.rglob("*") if p.is_file()}
        judge = {p.relative_to(JUDGE_REPO) for p in JUDGE_REPO.rglob("*") if p.is_file()}
        self.assertEqual(agent, judge)
        changed = {p for p in agent if (AGENT_REPO / p).read_bytes() != (JUDGE_REPO / p).read_bytes()}
        self.assertTrue(all(p.name == "user_tasks.py" for p in changed), changed)
        for path in (AGENT_REPO / DATA_REL).rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (JUDGE_REPO / path.relative_to(AGENT_REPO)).read_bytes())

    def test_no_removed_task_literal_anywhere_in_agent_tree(self):
        """Every distinctive string literal of a removed task is absent from agent-base/."""
        files = _agent_files()
        needles, kept_text = _judge_split_literals()
        for suite in workloads.SUITES:
            needles[suite] = {s for s in needles[suite] if s not in kept_text}
            self.assertGreater(len(needles[suite]), 10, suite)
            for needle in sorted(needles[suite]):
                hits = [str(p.relative_to(TASK)) for p, text in files.items() if needle in text]
                self.assertEqual(hits, [], f"{suite}: {needle[:80]!r}")


_PROBE = r"""
import json, sys
repo, trusted = sys.argv[1:3]
sys.path[:0] = [trusted, repo]
from agentdojo.task_suite.load_suites import get_suite
out = {}
for name in ("banking", "slack", "travel", "workspace"):
    by_version = {}
    for version in ("v1", "v1.1", "v1.1.1", "v1.1.2", "v1.2"):
        suite = get_suite(version, name)
        by_version[version] = {
            "user": {tid: [t.PROMPT, str(getattr(t, "GROUND_TRUTH_OUTPUT", "") or "")]
                     for tid, t in suite.user_tasks.items()},
            "injection": {tid: [t.GOAL, str(getattr(t, "GROUND_TRUTH_OUTPUT", "") or "")]
                          for tid, t in suite.injection_tasks.items()},
        }
    out[name] = by_version
print(json.dumps(out))
"""


def _probe_python():
    python = os.environ.get("PROGENT_AGENTDOJO_PYTHON", sys.executable)
    probe = subprocess.run([python, "-I", "-B", "-c", "import pydantic, yaml, deepdiff, openai"],
                           capture_output=True)
    return python if probe.returncode == 0 else None


class ProgentAgentDojoImportTests(unittest.TestCase):
    """Imports both AgentDojo copies (needs the task's pinned dependencies).

    Set PROGENT_AGENTDOJO_PYTHON to an interpreter with
    agent-base/frontier/task/requirements-lock.txt installed.
    """

    @classmethod
    def setUpClass(cls):
        python = _probe_python()
        if python is None:
            raise unittest.SkipTest("AgentDojo dependencies unavailable; set PROGENT_AGENTDOJO_PYTHON")

        def load(repo, trusted):
            env = {"PATH": os.environ.get("PATH", ""), "SECAGENT_SUITE": "banking",
                   "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run(
                [python, "-I", "-B", "-c", _PROBE, str(repo / "agentdojo/src"), str(trusted)],
                capture_output=True, text=True, env=env, timeout=300)
            if result.returncode:
                raise AssertionError(result.stderr[-4000:])
            return json.loads(result.stdout)

        cls.agent = load(AGENT_REPO, AGENT / "frontier/task/harness/trusted")
        cls.judge = load(JUDGE_REPO, TASK / "judge/trusted")

    def test_agent_enumerates_exactly_the_dev_tasks_and_all_injections(self):
        for suite in workloads.SUITES:
            agent, judge = self.agent[suite]["v1.1.2"], self.judge[suite]["v1.1.2"]
            self.assertEqual(sorted(agent["user"], key=_num), _agent_dev(suite), suite)
            self.assertEqual(sorted(judge["user"], key=_num), FULL[suite]["user"], suite)
            self.assertEqual(agent["injection"], judge["injection"], suite)
            for tid, value in agent["user"].items():
                self.assertEqual(value, judge["user"][tid], f"{suite}/{tid}")

    def test_no_held_out_prompt_or_ground_truth_in_agent_tree(self):
        files = _agent_files()
        data = {p for p in files if (AGENT_REPO / DATA_REL) in p.parents}
        _, shared_text = _judge_split_literals()
        for suite in workloads.SUITES:
            _, final = workloads.split(suite, FULL[suite]["user"])
            kept = []
            for version in self.judge[suite].values():
                kept += [v for tid, pair in version["user"].items() if _num(tid) not in _removed(suite)
                         for v in pair]
                kept += [v for pair in version["injection"].values() for v in pair]
            kept_text = "\n".join(kept) + shared_text
            checked = 0
            for version, tasks in self.judge[suite].items():
                for tid in final:
                    if tid not in tasks["user"]:
                        continue
                    prompt, truth = tasks["user"][tid]
                    # The environment (identical to the judge's) is world state the
                    # tools return in every episode; a prompt quoted there (a TODO
                    # page) is not a task definition.
                    hits = [p for p, text in files.items() if prompt in text and p not in data]
                    self.assertEqual(hits, [], f"{suite}/{tid}@{version} PROMPT")
                    checked += 1
                    # Short answers (a rating) and answers shared with the rest of
                    # the judge tree (a file name a dev task names, a code an
                    # injection task sends) are not attributable to the task.
                    if len(truth) < 6 or truth in kept_text:
                        continue
                    hits = [p for p, text in files.items() if truth in text and p not in data]
                    self.assertEqual(hits, [], f"{suite}/{tid}@{version} GROUND_TRUTH_OUTPUT")
            self.assertGreaterEqual(checked, len(final), suite)


if __name__ == "__main__":
    unittest.main()
