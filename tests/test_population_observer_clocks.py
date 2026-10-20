"""Trusted observers enforce the selected suite's existing runtime budget."""

import deployment_test_guard  # noqa: F401

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/frontier-cs-2.0-demo/problems/Machine Learning/representation_frechet_loss"
DEPS = TASK / "judge/judge_deps"


class PopulationObserverClockTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse((DEPS / "queue_boundary/src/queue_observer_entry.py").read_text())
        function = next(node for node in self.tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "validate_deadline")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "observer", "exec"), namespace)
        self.validate = namespace["validate_deadline"]

    def test_gpu_bound_matches_each_published_suite(self):
        runtime = yaml.safe_load((TASK / "judge/config.yaml").read_text())["runtime"]
        for suite, field in (("development", "timeout_seconds"), ("final", "final_timeout_seconds")):
            ceiling = runtime[field]
            with self.subTest(suite=suite):
                self.validate(1, "cuda:0", suite)
                self.validate(ceiling, "cuda:0", suite)
                with self.assertRaises(ValueError):
                    self.validate(ceiling + 1, "cuda:0", suite)
                with self.assertRaises(ValueError):
                    self.validate(0, "cuda:0", suite)

    def test_real_final_remaining_grant_does_not_use_development_ceiling(self):
        self.validate(10680, "cuda:0", "final")
        with self.assertRaises(ValueError):
            self.validate(10680, "cuda:0", "development")

    def test_cpu_boundary_remains_120_seconds(self):
        for suite in ("development", "final"):
            self.validate(120, "cpu", suite)
            with self.assertRaises(ValueError):
                self.validate(121, "cpu", suite)

    def test_unknown_contract_is_rejected(self):
        for device, suite in (("cuda:0", "unknown"), ("cuda:1", "final")):
            with self.assertRaises(ValueError):
                self.validate(1, device, suite)

    def test_trusted_launcher_passes_suite_independently_of_phase(self):
        tree = ast.parse((DEPS / "run_workload.py").read_text())
        argv = next(node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "argv" for target in node.targets))
        values = argv.elts
        suite_flag = next(index for index, node in enumerate(values)
                          if isinstance(node, ast.Constant) and node.value == "--suite")
        self.assertIsInstance(values[suite_flag + 1], ast.Name)
        self.assertEqual(values[suite_flag + 1].id, "suite")
        main = next(node for node in self.tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        suite_arg = next(node for node in ast.walk(main)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                         and node.func.attr == "add_argument" and node.args
                         and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--suite")
        self.assertTrue(next(node.value.value for node in suite_arg.keywords if node.arg == "required"))

    def run_main(self, suite, seconds, phase="train"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = root / "observe.py"
            observer.write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "def main():\n"
                "    Path(sys.argv[sys.argv.index('--report') + 1]).write_text(json.dumps({'observed': True}))\n"
            )
            report = root / "report.json"
            queue_import = mock.Mock(return_value=contextlib.nullcontext())
            spec = importlib.util.spec_from_file_location(
                "population_actual_observer_main", DEPS / "queue_boundary/src/queue_observer_entry.py"
            )
            module = importlib.util.module_from_spec(spec)
            argv = [str(spec.origin), "--phase", phase, "--trusted-repo", str(root),
                    "--spine-manifest", str(root / "spine.json"), "--queue-source", str(root),
                    "--queue-scratch", str(root), "--queue-python", sys.executable,
                    "--observer", str(observer), "--observer-sha256",
                    hashlib.sha256(observer.read_bytes()).hexdigest(), "--report", str(report),
                    "--seconds", str(seconds), "--device", "cuda:0"]
            if suite is not None:
                argv += ["--suite", suite]
            original_path = sys.path[:]
            try:
                with mock.patch.dict(sys.modules, {"observer_queue": SimpleNamespace(
                        trusted_queue_import=queue_import)}):
                    spec.loader.exec_module(module)
                with mock.patch.object(module, "bind_parent_lifetime"), \
                     mock.patch.object(module.time, "monotonic", return_value=1000), \
                     mock.patch.object(sys, "argv", argv):
                    try:
                        module.main()
                    except BaseException:
                        self.assertFalse(queue_import.called)
                        self.assertFalse(report.exists())
                        raise
                self.assertEqual(queue_import.call_args.kwargs["deadline"], 1000 + seconds)
                self.assertTrue(queue_import.call_args.kwargs["allow_cuda"])
                self.assertTrue(json.loads(report.read_text())["observed"])
            finally:
                sys.path[:] = original_path

    def test_actual_main_accepts_final_remaining_clock_in_both_phases(self):
        for phase in ("train", "eval"):
            with self.subTest(phase=phase):
                self.run_main("final", 10680, phase)

    def test_actual_main_rejects_development_overrun_before_loading_observer(self):
        with self.assertRaisesRegex(ValueError, "deadline exceeds"):
            self.run_main("development", 10680)

    def test_actual_parser_requires_explicit_suite(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self.run_main(None, 100)
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
