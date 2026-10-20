"""Remaining GPU grants bound provisioning, upload and remote execution."""

import deployment_test_guard  # noqa: F401  # isort: skip

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "tasks/frontier-cs-2.0-demo/problems"
# Population's scientific handoff and setup classification have a separate suite.
ADAPTERS = sorted(
    p for p in PROBLEMS.rglob("verifier/fcs2_verifier_gpu.py")
    if p.parent.parent.name != "representation_frechet_loss"
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RemainingEnvelopeTests(unittest.TestCase):
    def test_all_published_adapters_inventory(self):
        self.assertEqual(len(ADAPTERS), 14)
        for index, path in enumerate(ADAPTERS):
            adapter = load(path, f"deadline_adapter_{index}")
            with self.subTest(task=path.parent.parent.name):
                self.assertEqual(
                    adapter._runner_deadline(
                        {"granted_seconds": 600}, 7200, anchor=100
                    ),
                    699,
                )
                for invalid in (True, False, None, 0, -1, 7201, 1.5, "600"):
                    with self.assertRaises(adapter.VerifierGpuError):
                        adapter._runner_deadline(
                            {"granted_seconds": invalid}, 7200, anchor=100
                        )

    def test_provision_and_upload_spend_same_envelope(self):
        for index, path in enumerate(ADAPTERS):
            if path.parent.parent.name == "nano_world_model":
                continue  # Its existing phase-control suite covers this variant.
            adapter = load(path, f"spend_adapter_{index}")
            with (
                self.subTest(task=path.parent.parent.name),
                tempfile.TemporaryDirectory() as temp,
            ):
                inputs = Path(temp) / "inputs.tar"
                inputs.write_bytes(b"x")
                with mock.patch.object(adapter.time, "monotonic", return_value=300):
                    deadline = adapter._runner_deadline(
                        {"granted_seconds": 600}, 7200, anchor=100
                    )
                    self.assertEqual(
                        adapter._WaitWindow(deadline, "phase_timeout").remaining()
                        if path.parent.parent.name in {"flow_opd", "lrr"}
                        else adapter._remaining(deadline), 399
                    )
                    with mock.patch.object(
                        adapter if path.parent.parent.name in {"flow_opd", "lrr"} else adapter.subprocess,
                        "_upload_process" if path.parent.parent.name in {"flow_opd", "lrr"} else "run",
                        return_value=SimpleNamespace(returncode=0, stderr=b""),
                    ) as child:
                        adapter._upload_inputs(
                            ["ssh"], inputs, 600, {}, runner_deadline=deadline,
                            **({"window": adapter._WaitWindow(deadline, "phase_timeout")}
                               if path.parent.parent.name in {"flow_opd", "lrr"} else {})
                        )
                        self.assertLessEqual((child.call_args.args[3] if path.parent.parent.name in {"flow_opd", "lrr"}
                                              else child.call_args.kwargs["timeout"]), 120)
                with (
                    mock.patch.object(adapter.time, "monotonic", return_value=680),
                    mock.patch.object(
                        adapter if path.parent.parent.name in {"flow_opd", "lrr"} else adapter.subprocess,
                        "_upload_process" if path.parent.parent.name in {"flow_opd", "lrr"} else "run",
                        return_value=SimpleNamespace(returncode=0, stderr=b""),
                    ) as child,
                ):
                    adapter._upload_inputs(
                        ["ssh"], inputs, 600, {}, runner_deadline=deadline,
                            **({"window": adapter._WaitWindow(deadline, "phase_timeout")}
                               if path.parent.parent.name in {"flow_opd", "lrr"} else {})
                    )
                    self.assertLessEqual((child.call_args.args[3] if path.parent.parent.name in {"flow_opd", "lrr"}
                                              else child.call_args.kwargs["timeout"]), 19)
                with (
                    mock.patch.object(adapter.time, "monotonic", return_value=700),
                    mock.patch.object(adapter.subprocess, "Popen") as child,
                ):
                    with self.assertRaises(adapter.VerifierGpuError):
                        adapter._run_remote_verifier(
                            ["ssh"],
                            "work",
                            Path(temp) / "output",
                            600,
                            {},
                            runner_deadline=deadline,
                        )
                    child.assert_not_called()

    def test_invalid_or_consumed_grant_still_cleans_owned_runner(self):
        for index, path in enumerate(ADAPTERS):
            if path.parent.parent.name == "nano_world_model":
                continue
            adapter = load(path, f"cleanup_adapter_{index}")
            if path.parent.parent.name in {"flow_opd", "lrr"}:
                from test_flow_gpu_phase_control import assert_grant_cleanup
                # Flow reserves cleanup inside the grant: consume its work
                # window while retaining the actual deletion reserve.
                for grant, elapsed in ((True, 0), (100, 50)):
                    with self.subTest(task=path.parent.parent.name, grant=grant):
                        assert_grant_cleanup(self, adapter, grant, elapsed)
            else:
                self.check_legacy_grant_cleanup(adapter, path)

    def check_legacy_grant_cleanup(self, adapter, path):
        for grant, elapsed in ((True, 0), (100, 200)):
            runner = {"runner_id": "owned", "granted_seconds": grant}
            with (
                self.subTest(task=path.parent.parent.name, grant=grant),
                tempfile.TemporaryDirectory() as temp,
                mock.patch.dict(
                    os.environ, {"FRONTIER_MAX_EVIDENCE_BYTES": "8192"}
                ),
                mock.patch.object(
                    adapter,
                    "_config",
                    return_value=("url", "verification", "token", "gpu", 7200, 1),
                ),
                mock.patch.object(adapter, "_input_archive"),
                mock.patch.object(adapter, "_create_runner", return_value=runner),
                mock.patch.object(
                    adapter.time, "monotonic", side_effect=[100, 100 + elapsed]
                ),
                mock.patch.object(
                    adapter, "_terminate_runner", return_value=True
                ) as delete,
                mock.patch.object(adapter, "_upload_inputs") as upload,
            ):
                with self.assertRaises(adapter.VerifierGpuError) as caught:
                    adapter.run(
                        payload=Path(temp) / "payload",
                        dataset=Path(temp) / "dataset",
                        mode="eval",
                        result_path=Path(temp) / "result",
                        evidence_path=Path(temp) / "evidence",
                    )
                self.assertIn(
                    "invalid GPU granted duration"
                    if grant is True
                    else "deadline is exhausted",
                    str(caught.exception),
                )
                delete.assert_called_once_with(
                    "url", "verification", "token", "owned"
                )
                upload.assert_not_called()

if __name__ == "__main__":
    unittest.main()
