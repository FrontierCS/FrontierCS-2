"""polygon's searches on a trusted CPU command test sandbox, validation on the judge's MariaDB.

The judge (tasks/.../polygon/judge/evaluator.py) runs on the control image and sends every
search of both sides to one [test_sandboxes.search] machine through run_sandboxes(); the trusted
harness there (judge/search_harness.py) runs each search in a fresh isolate instance with no
network and times it; the counterexamples come back as data and are validated here on a MariaDB
started fresh for each. These cases run the shared verifier as the control image does, on the
adapter harness of tests/test_shared_gpu_adapter.py (a fake Driver and a fake SSH gateway that
runs each remote command on this host), so the real upload, step and collect commands run.

verifier.json is the one scripts/task_manifest.py generates for task.toml (a CPU machine,
image verifier_cpu); the fake Driver answers the CPU create. Needs the polygon Verifier image's
runtime (MariaDB, Z3, the pinned Python packages) and an isolate backend: run inside that
image, e.g. tests/run_polygon_search_sandbox.sh.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from test_shared_gpu_adapter import AdapterCase, task_manifest
from test_template_task import payload

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tasks/shared/lib"))
import isolate  # noqa: E402

TASK = REPO / "tasks/problems/Programming Language/polygon"
REFERENCE = (TASK / "judge/reference.patch").read_text()
ENVIRONMENT = (TASK / "repo/sql_testing/environment.py").read_text()
# The runner the fake Driver hands out first: its /workspace is <scratch>/runners/<id>/workspace.
FIRST_RUNNER = "gpurunner_0" + "0" * 25
DECLARED = tomllib.loads((TASK / "task.toml").read_text())["test_sandboxes"]["search"]

# Bits a probe search reports as its iteration count: what it managed to reach.
READ_JUDGE, READ_OTHER_TREE, WRITE_HARNESS_OUT, NETWORK, MARIADB, OLD_STATE, JUDGE_ENV, \
    HARNESS_PROC, SIGNAL_HARNESS = (1 << n for n in range(9))


def _runtime_available() -> bool:
    modules = ("pandas", "ujson", "prettytable", "mysql.connector", "multiprocess", "z3")
    return (shutil.which("mariadb-install-db") is not None and shutil.which("z3") is not None
            and all(importlib.util.find_spec(name) is not None for name in modules)
            and isolate.available())


def _region_patch(body: str) -> str:
    """A patch replacing the counterexample-discovery region of Environment.check."""
    begin = "        # FRONTIER_EDIT_BEGIN: counterexample discovery\n"
    end = "        # FRONTIER_EDIT_END: counterexample discovery\n"
    head, rest = ENVIRONMENT.split(begin)
    _old, tail = rest.split(end)
    new = head + begin + body + end + tail
    import difflib
    return "".join(difflib.unified_diff(ENVIRONMENT.splitlines(keepends=True),
                                        new.splitlines(keepends=True),
                                        "a/sql_testing/environment.py",
                                        "b/sql_testing/environment.py"))


def _probe_body(judge_dir: Path, runner_root: Path, marker: str) -> str:
    """Stand-in for fully compromised submitted code: tries everything, reports the bit mask,
    returns a counterexample that refutes nothing."""
    sandbox = runner_root / "workspace" / "polygon-search"
    return f'''\
        import glob as _g, json as _j, os as _o, signal as _s, socket as _k
        bits = 0
        for bit, path in (({READ_JUDGE}, {str(judge_dir / "reference.patch")!r}),
                          ({READ_JUDGE}, {str(judge_dir / "workloads/final.json")!r}),
                          ({READ_OTHER_TREE}, {str(sandbox / "trees/reference/frontier_runner.py")!r}),
                          ({READ_OTHER_TREE}, {str(sandbox / "cases.json")!r})):
            try:
                with open(path, "rb") as handle:
                    handle.read(1)
                bits |= bit
            except OSError:
                pass
        try:
            with open({str(sandbox / "out/0.json")!r}, "w") as handle:
                handle.write('{{"decided": true, "seconds": 0.0, "reason": null}}')
            bits |= {WRITE_HARNESS_OUT}
        except OSError:
            pass
        try:
            _k.create_connection(("127.0.0.1", 9), timeout=1).close()
            bits |= {NETWORK}
        except OSError as error:
            if getattr(error, "errno", None) == 111:
                bits |= {NETWORK}
        # Any MariaDB socket on the machine (the judge's validator never runs during a search,
        # and in production runs on the control machine, not this one).
        for path in _g.glob("/frontier/**/*.sock", recursive=True) + _g.glob("/tmp/**/*.sock",
                                                                             recursive=True):
            try:
                client = _k.socket(_k.AF_UNIX, _k.SOCK_STREAM)
                client.connect(path)
                client.close()
                bits |= {MARIADB}
            except OSError:
                pass
        for place in ({marker!r}, _o.path.join(_o.environ.get("HOME", "/nonexistent"), "m")):
            if _o.path.exists(place):
                bits |= {OLD_STATE}
            try:
                with open(place, "w") as handle:
                    handle.write("left by an earlier search")
            except OSError:
                pass
        if any(key.startswith("FRONTIER_") for key in _o.environ):
            bits |= {JUDGE_ENV}
        for cmdline in _g.glob("/proc/[0-9]*/cmdline"):
            try:
                with open(cmdline, "rb") as handle:
                    if b"search_harness.py" not in handle.read():
                        continue
                pid = int(cmdline.split("/")[2])
            except (OSError, ValueError):
                continue
            try:
                with open(f"/proc/{{pid}}/environ", "rb") as handle:
                    handle.read(1)
                bits |= {HARNESS_PROC}
            except OSError:
                pass
            try:
                _o.kill(pid, 0)
                bits |= {SIGNAL_HARNESS}
            except OSError:
                pass
        return False, {{}}, 0.0, 0.0, {{"iters": bits, "backtracks": 0}}
'''


# Writes a forged result line straight to the search's channel (the instance's stdout pipe,
# file descriptor 3 in search_child.py): "refuted" and a zero time, with a counterexample that
# refutes nothing, then ends the search.
FORGED_CHANNEL = '''\
        import os as _o
        _o.write(3, b'{"decided": true, "eq": false, "cex": {"$dict": []}, "refuted": true, '
                    b'"time_seconds": 0.0, "iterations": 7, "backtracks": 0}\\n')
        _o._exit(0)
'''

# Writes a refuting-looking result line one byte over the 128 KiB output cap to the channel.
OVERSIZED_CHANNEL = '''\
        import os as _o
        _line = b'{"decided": true, "eq": false, "cex": {"$dict": []}, "pad": "'
        _line += b"x" * (128 * 1024 + 1 - len(_line) - 2) + b'"}'
        _o.write(3, _line + b"\\n")
        _o._exit(0)
'''


@unittest.skipUnless(_runtime_available(), "needs the polygon Verifier image's runtime and isolate")
class PolygonSearchSandbox(AdapterCase):
    def setUp(self):
        super().setUp()
        # A CPU-only task: no `gpu` table and no FRONTIER_VERIFIER_GPU_* (as on the platform).
        os.environ.pop("FRONTIER_VERIFIER_GPU_TYPE")
        os.environ.pop("FRONTIER_VERIFIER_GPU_COUNT")
        generated = task_manifest.generated_files(TASK)
        self.parameters.write_bytes(generated[".frontier-generated/verifier/verifier.json"])
        self.bundle = self.scratch / "bundle"
        shutil.copytree(TASK / "judge", self.bundle)
        shutil.copytree(TASK / "repo", self.bundle / "pristine")
        (self.bundle / "config.yaml").write_bytes(generated[".frontier-generated/judge/config.yaml"])
        environment = mock.patch.dict(os.environ, {"PYTHONPATH": str(REPO / "tasks/shared/lib")})
        environment.start()
        self.addCleanup(environment.stop)
        self.judge_globals = {"PYTHON": sys.executable}
        original = self.verifier.load_evaluator
        self.judges = []

        def load_evaluator(task_dir):
            evaluate, check_tree = original(task_dir)
            evaluate.__wrapped__.__globals__.update(self.judge_globals)
            self.judges.append(evaluate.__wrapped__.__globals__)
            return evaluate, check_tree

        loader = mock.patch.object(self.verifier, "load_evaluator", load_evaluator)
        loader.start()
        self.addCleanup(loader.stop)

    def verify(self, patch: str, mode: str) -> dict:
        os.environ["FRONTIER_VERIFICATION_MODE"] = mode
        work = Path(tempfile.mkdtemp(dir=self.scratch))
        result = work / "result.json"
        self.verifier.verify(bundle=self.bundle, entrypoint="payload/solution.patch",
                             payload=payload(patch, work),
                             dataset=TASK / "datasets" / f"{mode}.json", mode=mode,
                             result_path=result, evidence_path=work / "evidence.tar.zst")
        return json.loads(result.read_text())

    def judge(self) -> dict:
        """The evaluator module's globals, loaded as the shared verifier loads it."""
        self.verifier.load_evaluator(self.bundle)
        return self.judges[-1]

    def measure(self, patch: str, count: int, judge: dict | None = None):
        """_measure_pair on the first `count` dev workloads with `patch` applied WITHOUT the
        shared verifier's policy checks, so the submitted code really runs."""
        os.environ["FRONTIER_VERIFICATION_MODE"] = "train"
        os.environ["FRONTIER_SUBMISSION_ROLE"] = "development"
        judge = judge or self.judge()
        tree = self.scratch / "attack-tree"
        shutil.copytree(TASK / "repo", tree)
        for path in (tree, *tree.rglob("*")):
            path.chmod(path.stat().st_mode | 0o200)
        subprocess.run(["git", "apply", "-"], input=patch.encode(), cwd=tree, check=True,
                       env={**os.environ, "GIT_CEILING_DIRECTORIES": str(self.scratch)})
        reference = judge["_reference_tree"]()
        records: list[dict] = []
        try:
            candidate, reference_rows = judge["_measure_pair"](
                tree, reference, judge["_load_suite"]()[:count], records)
        finally:
            shutil.rmtree(reference.parent, ignore_errors=True)
        return judge, candidate, reference_rows, records

    def test_the_lease_covers_every_search_limit(self):
        # Every instance of both sides at its limit, plus the harness reserve, the per-step
        # grace and the upload/collect reserves of run_sandboxes: a candidate's slow searches
        # are its own scored failures, never a trusted machine running out of time.
        judge = self.judge()
        for role, mode in (("development", "train"), ("final", "eval")):
            os.environ["FRONTIER_SUBMISSION_ROLE"] = role
            workloads = judge["_load_suite"]()
            limits = sum(judge["_limit_seconds"](w, b) for w in workloads
                         for b in (w.timeout_seconds, judge["_reference_budget_seconds"](w)))
            needed = (limits + len(workloads) * 2 * 2 + 30
                      + self.gpu.STEP_KILL_AFTER_SECONDS + self.gpu.STEP_TRANSPORT_GRACE_SECONDS
                      + self.gpu.SANDBOX_UPLOAD_SECONDS + self.gpu.SANDBOX_COLLECT_SECONDS)
            self.assertLessEqual(needed, DECLARED["lease_seconds"][mode], role)

    def test_reference_reproduces_on_the_search_sandbox(self):
        result = self.verify(REFERENCE, "train")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["summary"], "validated 21/21 Polygon workloads", result)
        # Round 6 on the platform: 95.46 (95.41-95.47 across re-verifications).
        self.assertGreater(result["raw_score"], 95.0, result)
        self.assertEqual(len(self.driver.creates), 1)
        create = self.driver.creates[0]
        self.assertEqual({key: create[key] for key in ("image", "cpu_millis", "memory_bytes")},
                         {"image": "verifier_cpu", **DECLARED["resources"]})
        self.assertEqual(len(self.driver.deleted), 1)

    def test_compromised_search_reaches_nothing(self):
        marker = f"/tmp/polygon-search-probe-{os.getpid()}"
        try:
            patch = _region_patch(_probe_body(self.bundle, self.scratch / "runners" / FIRST_RUNNER,
                                              marker))
            _judge, candidate, reference, records = self.measure(patch, 2)
        finally:
            if os.path.exists(marker):
                os.unlink(marker)
        self.assertTrue(all(row.valid and row.refuted for row in reference), records)
        # Every probe ran (the instance returned its result) and reached nothing: no judge
        # file, no other tree or plan, no harness output, no network, no MariaDB, no state from
        # the earlier search, no judge environment, no harness process. Its non-refuting
        # database is validated as such.
        self.assertEqual([(row.valid, row.iterations, row.refuted) for row in candidate],
                         [(True, 0, False)] * 2, records)

    def test_forged_channel_result_is_validated_and_timed_by_the_judge(self):
        _judge, candidate, _reference, records = self.measure(_region_patch(FORGED_CHANNEL), 2)
        # The forged "refuted" and "time_seconds" fields are not inputs: the judge validates
        # the counterexample itself (it refutes nothing) and the time is the harness's stamp.
        for row, record in zip(candidate, [r for r in records if r["side"] == "candidate"]):
            self.assertTrue(row.valid, record)
            self.assertFalse(row.refuted, record)
            self.assertEqual(row.iterations, 7)
            self.assertGreater(row.time_seconds, 0.0)

    def test_the_harness_records_of_either_suite_fit_one_collect(self):
        # out/ comes back in one collect of SEARCH_OUTPUT_BYTES, which run_sandboxes() allows
        # (one SSH chunk of the platform's default capture): every entry's result line at its
        # cap plus its largest record (every stderr byte non-ASCII: 6 JSON bytes each).
        judge = self.judge()
        self.assertLessEqual(judge["SEARCH_OUTPUT_BYTES"], self.gpu._upload_chunk_bytes(
            {"FRONTIER_MAX_SSH_BYTES_PER_DIRECTION": task_manifest.verifier_channel_bytes(None)}))
        sys.path.insert(0, str(self.bundle))
        import search_harness
        record = json.dumps({"index": 20, "side": "reference", "decided": False,
                             "seconds": 1e308, "reason": "search process did not start" * 4,
                             "result_over_cap": False, "returncode": -9, "limit": "memory",
                             "backend": "landlock", "wall_seconds": 1e308,
                             "stderr_tail": "\ufffd" * search_harness.DIAGNOSTIC_STDERR_BYTES},
                            sort_keys=True)
        for role in ("development", "final"):
            os.environ["FRONTIER_SUBMISSION_ROLE"] = role
            entries = judge["_plan"](judge["_load_suite"]())
            self.assertEqual({entry["result_max_bytes"] for entry in entries}, {128 * 1024})
            needed = (len(entries) * (128 * 1024 + len(record))
                      + self.gpu.MAX_LOG_TAIL_BYTES)
            self.assertLessEqual(needed, judge["SEARCH_OUTPUT_BYTES"], role)
            # The collect's tar stream on the SSH channel: per member (<n>.json, <n>.message,
            # out/, the step log tail, the unreadable list) one 512-byte header and at most 511
            # bytes of padding, plus the closing blocks rounded to a 10 KiB record. It stays
            # within 7/8 of the Verifier's channel, so the stream never reaches the channel.
            members = 2 * len(entries) + 3
            stream = needed + members * 1024 + 10240
            self.assertLessEqual(stream, self.gpu._upload_chunk_bytes(
                {"FRONTIER_MAX_SSH_BYTES_PER_DIRECTION": task_manifest.verifier_channel_bytes(
                    TASK)}), role)
            print(f"polygon {role}: {len(entries)} entries, collect stream <= {stream} bytes",
                  file=sys.stderr)

    def test_an_over_cap_result_fails_its_case_loudly(self):
        # Never truncated or validated: the case fails (refuted 0, charged its budget), the
        # record says why, and the evaluation's summary names the workload.
        _judge, candidate, reference, records = self.measure(_region_patch(OVERSIZED_CHANNEL), 2)
        self.assertTrue(all(row.valid and row.refuted for row in reference), records)
        rows = [r for r in records if r["side"] == "candidate"]
        self.assertEqual([(row.valid, row.refuted, row.time_seconds) for row in candidate],
                         [(False, False, 2.0)] * 2, records)
        self.assertEqual({(r["result_over_cap"], r["reason"]) for r in rows},
                         {(True, "search result exceeds the 131072-byte output cap")})
        judge = self.judge()
        os.environ["FRONTIER_FAST_TEST"] = "1"
        tree = self.scratch / "attack-tree"
        score, _unbounded, message, metrics = judge["evaluate"](str(tree))
        self.assertEqual(message, "validated 0/1 Polygon workloads; failed: the search result "
                                  "exceeded the 131072-byte output cap on workload-1")
        self.assertEqual(metrics["refuted_pairs"], [0])

    def test_validation_starts_only_after_every_search_ended(self):
        judge = self.judge()
        events = []
        real_run, real_fresh = self.gpu.run_sandboxes, judge["_MariaDB"].fresh

        def run(*args, **kwargs):
            events.append("search-start")
            try:
                return real_run(*args, **kwargs)
            finally:
                events.append("search-end")

        def fresh(server):
            events.append("validator")
            return real_fresh(server)

        with mock.patch.object(self.gpu, "run_sandboxes", run), \
                mock.patch.object(judge["_MariaDB"], "fresh", fresh):
            _j, candidate, reference, records = self.measure(REFERENCE, 2, judge)
        self.assertTrue(all(r.valid and r.refuted for r in candidate + reference), records)
        self.assertEqual(events[:2], ["search-start", "search-end"])
        self.assertEqual(events[2:], ["validator"] * 4)

    def test_malformed_results_fail_their_case_only(self):
        judge = self.judge()
        workload = judge["_load_suite"]()[0]
        entry = judge["_plan"]((workload,))[1]
        harness = {"decided": True, "seconds": 0.5, "reason": None, "returncode": 0,
                   "limit": None, "wall_seconds": 1.0, "stderr_tail": "", "backend": "x",
                   "result_over_cap": False}
        tables = judge["_adapter"]()._pair_inputs(workload.payload, judge["_pristine_root"]())
        evil = "'); DROP DATABASE mysql; --"
        # Values reach MariaDB only as parameters (MySQLTester's executemany).
        injection = {"$dict": [[table["TableName"].lower(), [
            [column["Name"] for column in table["PKeys"] + table["Others"]],
            [evil for _column in table["PKeys"] + table["Others"]]]]
            for table in tables["schema"]["Tables"]]}
        with judge["_Validator"]() as validator:
            for message, valid in ((b"not json", False), (b"[1, 2]", True),
                                   (b'{"decided": true, "eq": false, "cex": {"$evil": 1}}', False),
                                   (b'{"decided": true, "eq": false, "cex": ' + b'[' * 40
                                    + b']' * 40 + b'}', False),
                                   (json.dumps({"decided": True, "eq": False,
                                               "cex": injection}).encode(), True)):
                row = judge["_observe"](workload, entry, dict(harness), message, validator, {})
                self.assertEqual((row.valid, row.refuted), (valid, False), message)

    def test_a_failed_search_machine_is_unscored(self):
        self.driver.fail_scope_suffix = 1
        result = self.verify(REFERENCE, "train")
        self.assertEqual((result["status"], result["failure"]["code"]),
                         ("failed", "verifier_gpu_error"), result)

    def test_a_failed_harness_is_unscored(self):
        (self.bundle / "search_harness.py").chmod(0o644)
        (self.bundle / "search_harness.py").write_text("raise SystemExit(3)\n")
        result = self.verify(REFERENCE, "train")
        self.assertEqual(result["status"], "failed", result)
        self.assertNotIn("raw_score", result)


if __name__ == "__main__":
    unittest.main()
