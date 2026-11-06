"""No platform credential reaches task code (decision 34).

The Verifier wrapper starts the shared verifier with FRONTIER_VERIFIER_TOKEN, the Driver bearer
that also authorizes the result and evidence PUTs. These tests run the real shared verifier
(tasks/shared/verifier/fcs2_verifier.main) in its own process, as the wrapper does, with a judge
whose evaluate() looks for every FRONTIER_* secret: in os.environ, in the process's initial
environment block, in the shared verifier's handle, and from a child process (its environment,
the verifier's /proc/<pid>/environ and /proc/<pid>/mem). Any secret it can see fails the test.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import task_manifest  # noqa: E402

VERIFIER_DIR = REPO / "tasks/shared/verifier"
TEMPLATE = REPO / "tasks/template/Theory/template_task"
SECRETS = {"FRONTIER_VERIFIER_TOKEN": "drv-bearer-0d3c9e",
           "FRONTIER_MODEL_PROXY_TOKEN": "proxy-bearer-51aa",
           "FRONTIER_EXAMPLE_API_KEY": "api-key-77b2"}

PROBE = r'''
import ctypes, json, os, subprocess, sys
from pathlib import Path

SECRETS = json.loads(Path(os.environ["PROBE_SECRETS_FILE"]).read_text())
CHILD = r"""
import json, os, sys
secrets, parent = list(json.load(open(os.environ["PROBE_SECRETS_FILE"])).values()), int(sys.argv[1])
found = sorted(k for k, v in os.environ.items() if any(s in v for s in secrets))
readable = {}
for name in ("environ", "mem", "cmdline"):
    try:
        body = open(f"/proc/{parent}/{name}", "rb").read(1 << 20) if name != "mem" else b""
        if name == "mem":
            open(f"/proc/{parent}/mem", "rb").close()
        readable[name] = [s for s in secrets if s.encode() in body] or True
    except OSError:
        readable[name] = False
print(json.dumps({"env": found, "proc": readable}))
"""

def evaluate(tree):
    stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    start, end = int(stat[47]), int(stat[48])
    block = ctypes.string_at(start, end - start)
    child = subprocess.run([sys.executable, "-c", CHILD, str(os.getpid())],
                           capture_output=True, text=True, check=True)
    shell = subprocess.run(["/bin/sh", "-c", "env"], capture_output=True, text=True, check=True)
    verifier = sys.modules["fcs2_verifier"]
    report = {
        "env": sorted(k for k, v in os.environ.items() if any(s in v for s in SECRETS.values())),
        "env_names": sorted(k for k in os.environ if k in SECRETS),
        "initial_block": [s for s in SECRETS.values() if s.encode() in block],
        "handle": getattr(verifier, "verifier_token", lambda: None)(),
        "child": json.loads(child.stdout),
        "shell": [s for s in SECRETS.values() if s in shell.stdout],
    }
    Path(os.environ["PROBE_OUT"]).write_text(json.dumps(report))
    return 1.0, 1.0, "probe", {}
'''

RUN = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fcs2_verifier import main
raise SystemExit(main(Path(sys.argv[2]), "payload/solution.patch"))
"""


class SecretsNeverReachTaskCode(unittest.TestCase):
    def setUp(self):
        self.scratch = Path(tempfile.mkdtemp(prefix="verifier-credentials-"))
        self.addCleanup(shutil.rmtree, self.scratch, True)
        self.bundle = self.scratch / "task"
        shutil.copytree(task_manifest.judge_bundle(TEMPLATE), self.bundle)
        (self.bundle / "evaluator.py").chmod(0o644)
        (self.bundle / "evaluator.py").write_text(textwrap.dedent(PROBE))
        (self.bundle / "check_tree.py").unlink(missing_ok=True)
        self.out = self.scratch / "out"
        self.out.mkdir()
        self.secrets = self.scratch / "secrets.json"  # not in the environment: it would leak them
        self.secrets.write_text(json.dumps(SECRETS))
        body = (TEMPLATE / "judge/reference.patch").read_bytes()
        self.payload = self.scratch / "payload.tar"
        with tarfile.open(self.payload, "w", format=tarfile.USTAR_FORMAT) as archive:
            folder = tarfile.TarInfo("payload")
            folder.type, folder.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(folder)
            entry = tarfile.TarInfo("payload/solution.patch")
            entry.size = len(body)
            archive.addfile(entry, io.BytesIO(body))

    def verify(self, **extra) -> dict:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8",
            "FRONTIER_PAYLOAD_PATH": str(self.payload),
            "FRONTIER_DATASET_PATH": str(TEMPLATE / "datasets/train.json"),
            "FRONTIER_VERIFICATION_MODE": "train",
            "FRONTIER_RESULT_PATH": str(self.out / "result.json"),
            "FRONTIER_EVIDENCE_PATH": str(self.out / "evidence.tar.zst"),
            "FRONTIER_VERIFIER_SCRATCH": str(self.scratch),
            "PROBE_OUT": str(self.out / "probe.json"),
            "PROBE_SECRETS_FILE": str(self.secrets),
            **SECRETS, **extra,
        }
        completed = subprocess.run([sys.executable, "-I", "-c", RUN, str(VERIFIER_DIR), str(self.bundle)],
                                   env=environment, capture_output=True, text=True, timeout=300)
        result = json.loads((self.out / "result.json").read_text())
        self.assertEqual((completed.returncode, result["status"]), (0, "passed"),
                         completed.stderr + json.dumps(result))
        return json.loads((self.out / "probe.json").read_text())

    def assert_unreachable_outside_the_handle(self, report: dict):
        self.assertEqual(report["env"], [])
        self.assertEqual(report["env_names"], [])
        self.assertEqual(report["initial_block"], [])
        self.assertEqual(report["child"]["env"], [])
        self.assertEqual(report["shell"], [])
        self.assertEqual(report["child"]["proc"]["environ"], False)
        self.assertEqual(report["child"]["proc"]["mem"], False)
        self.assertNotIn(report["child"]["proc"]["cmdline"], [[s] for s in SECRETS.values()])

    def test_cpu_task_code_sees_no_secret_at_all(self):
        report = self.verify()
        self.assert_unreachable_outside_the_handle(report)
        self.assertIsNone(report["handle"])  # nothing in the process needs it: discarded

    def test_model_gateway_task_keeps_only_the_sealed_bearer(self):
        inventory = self.scratch / "inventory.json"
        inventory.write_text("{}")
        report = self.verify(FRONTIER_LITELLM_INVENTORY=str(inventory))
        self.assert_unreachable_outside_the_handle(report)
        # The judge's model-gateway header; with the core token split this is the scoped token.
        self.assertEqual(report["handle"], SECRETS["FRONTIER_VERIFIER_TOKEN"])


if __name__ == "__main__":
    unittest.main()
