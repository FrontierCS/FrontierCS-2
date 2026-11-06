"""The production Verifier layout (scripts/verifier_layout.py) mirrors the Driver's Pod."""

from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verifier_layout as layout  # noqa: E402

VERIFIER = {"command": ["/opt/frontier/verifier"], "cpu_millis": 4000,
            "memory_bytes": 8 << 30, "max_evidence_bytes": 4 << 20}
# The variables driver-k8s build_verifier_job sets on the Verifier container, and the five the
# verifier wrapper adds for the task command (FrontierCS core main 3adac45).
DRIVER_POD = {"FRONTIER_RUN_ID", "FRONTIER_ATTEMPT_ID", "FRONTIER_TASK_CONFIG_DIGEST",
              "FRONTIER_VERIFIER_IMAGE_DIGEST", "FRONTIER_VERIFICATION_ID",
              "FRONTIER_VERIFICATION_MODE", "FRONTIER_DRIVER_INTERNAL_URL",
              "FRONTIER_VERIFIER_TOKEN", "FRONTIER_VERIFIER_COMMAND_JSON",
              "FRONTIER_VERIFIER_RESULT_PATH", "FRONTIER_VERIFIER_EVIDENCE_PATH",
              "FRONTIER_MAX_EVIDENCE_BYTES", "FRONTIER_DATASET_VERSION"}
WRAPPER = {"FRONTIER_PAYLOAD_PATH", "FRONTIER_DATASET_PATH", "FRONTIER_RESULT_PATH",
           "FRONTIER_EVIDENCE_PATH"}


class VerifierLayoutTests(unittest.TestCase):
    def test_environment_is_exactly_the_driver_and_wrapper_variables(self):
        environment = layout.environment(VERIFIER, "train")
        self.assertEqual(set(environment), DRIVER_POD | WRAPPER)
        # Production sets no scratch, TMPDIR or HOME: the image and the shared verifier decide.
        for name in ("FRONTIER_VERIFIER_SCRATCH", "TMPDIR", "HOME"):
            self.assertNotIn(name, environment)
        self.assertEqual(environment["FRONTIER_PAYLOAD_PATH"], "/frontier/input/payload.tar")
        self.assertEqual(environment["FRONTIER_DATASET_PATH"], "/frontier/input/dataset")
        self.assertEqual(environment["FRONTIER_RESULT_PATH"], "/frontier/output/result.json")
        self.assertEqual(json.loads(environment["FRONTIER_VERIFIER_COMMAND_JSON"]),
                         VERIFIER["command"])
        with self.assertRaises(ValueError):
            layout.environment(VERIFIER, "final")

    def test_staged_work_matches_what_the_wrapper_leaves_in_the_empty_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "frontier"
            layout.stage_work(work, b"tar", b"data")
            self.assertEqual(sorted(p.relative_to(work).as_posix() for p in work.rglob("*")),
                             ["input", "input/dataset", "input/payload.tar", "output"])
            self.assertEqual((work / "input/payload.tar").read_bytes(), b"tar")
            self.assertEqual(work.stat().st_mode & 0o777, 0o777)

    def test_pod_matches_the_driver_verifier_pod(self):
        verifier = {**VERIFIER, "workspace_bytes": 4 << 30}
        pod = layout.pod_manifest("p", "image", verifier, "train",
                                  node_selector={"pool": "eval16"})
        spec = pod["spec"]
        self.assertIs(spec["hostUsers"], False)
        self.assertEqual(spec["securityContext"],
                         {"runAsNonRoot": True, "runAsUser": 65532, "runAsGroup": 65532,
                          "fsGroup": 65532, "seccompProfile": {"type": "RuntimeDefault"}})
        [container] = spec["containers"]
        self.assertEqual(container["securityContext"],
                         {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]},
                          "readOnlyRootFilesystem": True})
        # One emptyDir at /frontier, nothing mounted below it (each mount would be its own
        # filesystem, which production does not have).
        self.assertEqual(container["volumeMounts"], [{"name": "work", "mountPath": "/frontier"}])
        self.assertEqual(spec["volumes"], [{"name": "work",
                                            "emptyDir": {"sizeLimit": str(4 << 30)}}])
        self.assertEqual({item["name"] for item in container["env"]}, DRIVER_POD | WRAPPER)
        self.assertEqual(container["resources"]["limits"],
                         {"cpu": "4000m", "memory": str(8 << 30)})
        self.assertEqual(spec["nodeSelector"], {"pool": "eval16"})

    def test_pod_cli_prints_the_manifest(self):
        import contextlib
        import io
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            layout.main(["pod", "--name", "p", "--image", "i", "--mode", "eval",
                         "--cpu-millis", "2000", "--memory-bytes", "1024",
                         "--node-selector", "a=b"])
        pod = json.loads(stream.getvalue())
        self.assertEqual(pod["spec"]["nodeSelector"], {"a": "b"})
        self.assertEqual(pod["spec"]["containers"][0]["command"], ["sleep", "86400"])
        # The Driver's default Verifier workspace when the task sets none.
        self.assertEqual(pod["spec"]["volumes"],
                         [{"name": "work", "emptyDir": {"sizeLimit": str(4 << 30)}}])


if __name__ == "__main__":
    unittest.main()
