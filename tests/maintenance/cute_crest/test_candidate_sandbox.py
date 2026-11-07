"""The strategy compiles under the shared isolate launcher: the compiler sees none of the judge's files.

Regression for the compile-time probes the old source rules banned (__has_include, __FILE__,
__DATE__ ...): the build now runs in an isolate instance, so a probe of a judge file finds
nothing and the PoC (which only compiles when it can see /opt/frontier/task) fails to build.
Needs Linux x86_64 with Landlock ABI >= 6 and g++ with a static libstdc++.
"""
import deployment_test_guard  # noqa: F401  # isort: skip

import os
import shutil
import subprocess

import pytest

from tests.maintenance.cute_crest.runtime_imports import JUDGE, MODULES, REPO_TREE

e = MODULES["evaluator"]
pytestmark = pytest.mark.skipif(not e.isolate.available() or shutil.which("g++") is None,
                                reason="needs Linux x86_64 with Landlock ABI >= 6 and g++")

# Compiles only where the judge's files are visible to the compiler.
POC = '''#if __has_include("/opt/frontier/task/programs.json") || __has_include("%s")
#define POC_JUDGE_VISIBLE 1
#else
#define POC_JUDGE_VISIBLE 0
#endif
namespace poc {
static_assert(POC_JUDGE_VISIBLE, "judge files are not visible to the compile");
const char* kBuilt = __DATE__ " " __TIME__ " " __FILE__;
}
'''


@pytest.fixture()
def built(tmp_path, monkeypatch):
    """The image's built tree as far as the strategy build reads it: tree/src."""
    root = tmp_path / "crest-task"
    shutil.copytree(REPO_TREE / "src", root / "tree/src")
    monkeypatch.setattr(e, "BUILT", root)
    return root


def reference_solution(tmp_path):
    tree = tmp_path / "reference"
    shutil.copytree(REPO_TREE, tree, ignore=shutil.ignore_patterns("cil", "frontier_assets"))
    subprocess.run(["git", "apply", "--include=src/frontier_solution/*",
                    str(JUDGE / "reference.patch")], cwd=tree, check=True,
                   env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path)})
    return tree / e.SOLUTION


def test_reference_builds_in_the_sandbox(built, tmp_path):
    host = e.build_candidate(e.load_policy(), reference_solution(tmp_path), tmp_path / "out")
    assert host.is_file() and os.access(host, os.X_OK)
    assert host.read_bytes()[:4] == b"\x7fELF"


def test_compile_cannot_see_the_judge(built, tmp_path):
    secret = tmp_path / "judge-secret.json"
    secret.write_text("{}")
    solution = reference_solution(tmp_path)
    (solution / "poc.cc").write_text(POC % secret)
    policy = e.load_policy()
    assert policy.check_sources(solution)            # no source rule rejects the probe any more
    # Unconfined, the compiler sees the judge's file and the PoC builds.
    subprocess.run(["g++", "-std=c++17", "-fsyntax-only", str(solution / "poc.cc")], check=True)
    # In the sandbox it does not: the probe finds nothing and the build fails (a scored zero).
    with pytest.raises(e.InvalidCandidate) as error:
        e.build_candidate(policy, solution, tmp_path / "out")
    assert error.value.code == "build_failed"
