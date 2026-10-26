"""EVM State Checkpoints: one metric set on every workload (Preview issue #646).

Spectrum workloads used to report only committed TPS and the checkpoint-micro trace
only per-call create/restore/bytes, so a backend could move cost between them
uncounted (restore eagerly on shallow Spectrum stacks, defer the reroot past
``load_checkpoint()`` at height 1,000, where nothing timed the following accesses
and only ``get()`` was checked). Now every workload reports throughput, create,
restore, retained bytes and correctness, the gate and composite use all of them,
and restores are checked through ``top()``/``pop()`` as well as ``get()``.
"""
import deployment_test_guard  # noqa: F401 # isort: skip

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
EVM = ROOT / "tasks/frontier-cs-2.0-demo/problems/Database/evm_state_checkpoints"
HEADER = Path("lib/evmcow/execution_state.hpp")
GXX = shutil.which("g++")

# The #646 pattern on the shipped header: above height 64 load_checkpoint() only
# records the target; get()/get_mut()/push() finish the restore, top()/pop() do not.
LAZY_EDITS = (
    ("    inline void push(const uint256& item) {\n        values[height++] = item;",
     "    const StackTop* pending = nullptr;\n"
     "    void sync() const {\n"
     "        if (pending) { auto* self = const_cast<StackTop*>(this);"
     " self->values = pending->values; self->pending = nullptr; }\n"
     "    }\n"
     "    inline void push(const uint256& item) {\n        sync();\n        values[height++] = item;"),
    ("    inline const uint256& top() {\n        return get(0);",
     "    inline const uint256& top() {\n        return values[height - 1];"),
    ("    inline const uint256& get(size_t index) const {\n        index = height - 1 - index;",
     "    inline const uint256& get(size_t index) const {\n        sync();\n        index = height - 1 - index;"),
    ("    inline uint256& get_mut(size_t index) {\n        index = height - 1 - index;",
     "    inline uint256& get_mut(size_t index) {\n        sync();\n        index = height - 1 - index;"),
    ("            .stack   = this->stack_top,",
     "            .stack   = (this->stack_top.sync(), this->stack_top),"),
    ("        this->stack_top = checkpoint.stack;",
     "        if (checkpoint.stack.height > 64) {\n"
     "            this->stack_top.height = checkpoint.stack.height;\n"
     "            this->stack_top.pending = &checkpoint.stack;\n"
     "        } else {\n"
     "            this->stack_top = checkpoint.stack;\n"
     "        }"),
)


def load_evaluator():
    sys.path.insert(0, str(EVM / "judge"))
    try:
        sys.modules.pop("submission_patch", None)
        spec = importlib.util.spec_from_file_location("evm_uniform_evaluator", EVM / "judge/evaluator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EVM / "judge"))


@unittest.skipUnless(GXX, "g++ is required for the EVM checkpoint driver")
class CheckpointMicroDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="evm-uniform-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)

    def tree(self, name, edit=None):
        tree = self.tmp / name
        shutil.copytree(EVM / "agent-base/frontier/task/repo", tree)
        if edit:
            edit(tree)
        return tree

    def run_driver(self, tree, seed=1907, rounds=20):
        binary = tree / "driver"
        subprocess.run([GXX, "-O2", "-DNDEBUG", "-std=c++20", f"-I{tree}",
                        f"-I{EVM / 'judge/judge_deps/include'}",
                        str(EVM / "judge/checkpoint_bench.cpp"), "-o", str(binary)],
                       check=True, capture_output=True, timeout=300)
        return subprocess.run([str(binary), str(rounds), str(seed)], capture_output=True,
                              text=True, timeout=120)

    def test_shipped_and_reference_report_the_full_metric_set(self):
        def reference(tree):
            subprocess.run(["git", "apply", "--whitespace=nowarn", str(EVM / "judge/reference.patch")],
                           cwd=tree, check=True, capture_output=True,
                           env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tree.parent)})
        for name, edit in (("naive", None), ("reference", reference)):
            with self.subTest(name):
                proc = self.run_driver(self.tree(name, edit))
                self.assertEqual(proc.returncode, 0, proc.stderr)
                create, restore, retained, live, _, end_to_end = map(int, proc.stdout.split())
                self.assertTrue(create > 0 and restore > 0 and retained > 0 and end_to_end > 0)
                self.assertEqual(live, 16)

    def test_restore_deferred_past_load_with_stale_top_fails(self):
        def lazy(tree):
            header = tree / HEADER
            text = header.read_text()
            for old, new in LAZY_EDITS:
                self.assertEqual(text.count(old), 1, old)
                text = text.replace(old, new)
            header.write_text(text)
        tree = self.tree("lazy", lazy)
        for seed in (1907, 3253):
            with self.subTest(seed=seed):
                self.assertNotEqual(self.run_driver(tree, seed=seed).returncode, 0)

    def test_create_and_restore_timers_are_unchanged_and_end_to_end_covers_the_mix(self):
        source = (EVM / "judge/checkpoint_bench.cpp").read_text()
        create = source[source.index("auto checkpoint = state->save_checkpoint();") - 60:]
        create = create[:create.index("snapshot_ns +=")]
        self.assertNotIn("get_mut", create)
        restore = source[source.index("state->load_checkpoint(moved);"):]
        self.assertIn("auto t1 = clock_type::now();", restore[:80])
        self.assertIn("access_mix", restore[:restore.index("end_to_end_ns +=")])


class UniformMetricEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator()

    def sample(self, throughput, create=100.0, restore=50.0, retained=1000.0):
        return {"throughput": throughput, "create_ns": create, "restore_ns": restore, "retained_bytes": retained}

    def test_every_workload_has_every_metric(self):
        e = self.evaluator
        for final, suite in ((False, e.DEV_SUITE), (True, e.FINAL_SUITE)):
            rows = e._beat_rows(final)
            self.assertEqual(len(rows), 2 * len(suite))
            self.assertEqual({row["workload"].split(":")[0] for row in rows} >= {e.MICRO_WORKLOAD}, True)
            for row in rows:
                self.assertEqual(set(row["metrics"]), set(e.BEAT_METRICS))

    def test_decisions_apply_strict_separation_to_spectrum_checkpoint_metrics(self):
        e = self.evaluator
        suite = e.DEV_SUITE
        better = [self.sample(200.0, 50.0, 20.0, 500.0) for _ in range(5)]
        worse = [self.sample(100.0) for _ in range(5)]
        samples = {"candidate": {w: list(better) for w, _ in suite},
                   "reference": {w: list(worse) for w, _ in suite}}
        # The Spectrum restore time of the candidate is worse on one workload only.
        slow_restore = [self.sample(200.0, 50.0, 80.0, 500.0) for _ in range(5)]
        samples["candidate"][suite[0][0]] = slow_restore
        profiles = {"candidate": {"per_seed": {s: list(better) for _, s in suite}},
                    "reference": {"per_seed": {s: list(worse) for _, s in suite}}}
        decisions = e._beat_decisions(suite, samples, profiles)
        self.assertFalse(decisions[e._workload_id(*suite[0])]["checkpoint_restore_ns"])
        self.assertTrue(decisions[e._workload_id(*suite[1])]["checkpoint_restore_ns"])
        for _, seed in suite:
            self.assertTrue(all(decisions[e._workload_id(e.MICRO_WORKLOAD, seed)].values()))

    def test_composite_counts_spectrum_checkpoint_costs_and_micro_throughput(self):
        e = self.evaluator
        suite = e.DEV_SUITE
        def quality(spectrum, micro):
            samples = {"x": {w: [spectrum] * 5 for w, _ in suite}}
            profiles = {"x": {"per_seed": {s: [micro] for _, s in suite}}}
            with mock.patch.object(e, "_one_benchmark", side_effect=lambda *a: spectrum):
                return e._qualities({"x": Path("/bin/true")}, Path("/"), suite, profiles, 5)[0]["x"]
        base = quality(self.sample(100.0), self.sample(10.0))
        self.assertLess(quality(self.sample(100.0, restore=400.0), self.sample(10.0)), base)
        self.assertLess(quality(self.sample(100.0), self.sample(5.0)), base)

    def test_spectrum_run_requires_the_checkpoint_line(self):
        e = self.evaluator
        good = ("commit        53721.0000 tx/s\n"
                "checkpoint create_ns 399.56 restore_ns 108.49 retained_bytes 83740.92 "
                "creates 274727 restores 2322 retained_samples 274727\n")
        for stderr, ok in ((good, True), ("commit        53721.0000 tx/s\n", False),
                           (good.replace("restores 2322", "restores 0"), False)):
            proc = subprocess.CompletedProcess([], 0, "", stderr)
            with self.subTest(ok=ok), mock.patch.object(e, "_run_checked", return_value=proc):
                if ok:
                    self.assertEqual(e._one_benchmark(Path("b"), Path("s"), "YCSB:1:0.9", 1)["restore_ns"], 108.49)
                else:
                    with self.assertRaises(ValueError):
                        e._one_benchmark(Path("b"), Path("s"), "YCSB:1:0.9", 1)

    def test_harness_tokens_are_forbidden_in_submissions(self):
        e = self.evaluator
        for token in ("mallctl", "jemalloc", "frontier_checkpoint_record_restore"):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(token=token):
                patch = Path(tmp) / "p.patch"
                patch.write_text(f"+++ b/{HEADER}\n+// {token}\n")
                self.assertIsNotNone(e._forbidden_patch(patch))


if __name__ == "__main__":
    unittest.main()
