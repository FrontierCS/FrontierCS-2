"""Diffusion admits hosts by the sealed hardware class."""
import deployment_test_guard  # noqa: F401 # isort: skip

import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

TASK = Path(__file__).resolve().parents[1] / (
    "tasks/frontier-cs-2.0-demo/problems/Robotics/diffusion_policy_cpu")
PATCHED = "6.12.111-136.204.amzn2023.x86_64"


class HardwareClassTests(unittest.TestCase):
    def setUp(self):
        old_modules, old_path = dict(sys.modules), list(sys.path)

        def restore():
            for name in set(sys.modules) - set(old_modules):
                del sys.modules[name]
            sys.modules.update(old_modules)
            sys.path[:] = old_path

        self.addCleanup(restore)
        for name in ("hardware_admission", "native_data", "private_context", "task_guards"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(TASK / "judge"))
        from hardware_admission import admits
        from native_data import read_dataset
        self.admits, self.read_dataset = admits, read_dataset
        command = json.loads((TASK / "policy.json").read_bytes())["verifier"]["command"][-1]
        self.selector = {"__name__": "diffusion_hardware_class_selector"}
        exec(compile(command, "policy-selector", "exec"), self.selector)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.scratch = Path(directory.name)

    def sealed(self, native):
        selector = json.loads((TASK / "datasets" / f"{native}.json").read_bytes())
        body = base64.b64decode(selector["archive_base64"], validate=True)
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:") as composite:
            manifest = json.load(composite.extractfile("manifest.json"))
        (identity,) = [entry["identity"] for entry in manifest["datasets"].values()]
        return identity

    def select(self, native, identity):
        scratch = Path(tempfile.mkdtemp(dir=self.scratch))
        return self.selector["select_dataset"](TASK / "datasets" / f"{native}.json", native, identity,
                                               scratch, self.read_dataset)

    def admitted(self, identity):
        return [dict(identity), dict(identity, kernel=PATCHED),
                dict(identity, kernel="6.12.110-135.202.amzn2023.x86_64"), dict(identity, affinity=[2, 3])]

    def rejected(self, identity):
        return [dict(identity, cpu_models=["Intel(R) Xeon(R) Platinum 8175M CPU @ 2.50GHz"]),
                dict(identity, kernel="6.1.155-176.282.amzn2023.x86_64"),
                dict(identity, kernel="6.12.111-136.204.amzn2.x86_64"), dict(identity, kernel="7.0.0-1014-aws"),
                dict(identity, architecture="aarch64"), dict(identity, affinity=[0]),
                dict(identity, affinity=[0, 1, 2])]

    def test_rule(self):
        sealed = self.sealed("train")
        for actual in self.admitted(sealed):
            self.assertTrue(self.admits(sealed, actual), actual)
        for actual in self.rejected(sealed):
            self.assertFalse(self.admits(sealed, actual), actual)
        self.assertTrue(self.admits({"synthetic": True}, {"synthetic": True}))
        self.assertFalse(self.admits({"synthetic": True}, sealed))

    def test_selector_uses_the_class_over_the_released_datasets(self):
        for native in ("train", "eval"):
            sealed = self.sealed(native)
            key = self.selector["hardware_key"](sealed)
            for actual in self.admitted(sealed):
                self.assertEqual(self.select(native, actual).name, key + ".tar")
            for actual in self.rejected(sealed):
                with self.assertRaisesRegex(ValueError, "no accepted dataset"):
                    self.select(native, actual)

    def test_runtime_check_admits_the_class_and_keeps_the_sealed_constants(self):
        import private_context as pc
        from native_data import materialize_dataset
        # The image inventory: COPY judge, then chmod -R a+rX,a-w (harbor .sh files 0555).
        source = {}
        for path in sorted((TASK / "judge").rglob("*")):
            if path.is_file():
                name = path.relative_to(TASK / "judge").as_posix()
                executable = path.stat().st_mode & 0o111 or (name.startswith("harbor/app/") and name.endswith(".sh"))
                source[name] = {"sha256": pc.file_sha256(path), "mode": 0o555 if executable else 0o444}
        runtime, real = object(), pc.canonical_sha256
        sealed = self.sealed("train")
        with materialize_dataset(self.select("train", sealed), "dev", self.scratch) as root:
            seal = json.loads((root / "seal.json").read_bytes())
            with mock.patch.object(pc, "inventory", lambda root: source), \
                    mock.patch.object(pc, "runtime_identity", lambda prefix: runtime), \
                    mock.patch.object(pc, "canonical_sha256", lambda value: (
                        seal["runtime"]["identity_sha256"] if value is runtime else real(value))), \
                    mock.patch.object(pc, "verify_cgroup", lambda policy: None):
                for actual in self.admitted(sealed):
                    with mock.patch.object(pc, "hardware_identity", lambda: actual):
                        context = pc.load_context(root=root, mode="dev")
                        constants = pc.accepted_calibrations(context)["dev"]["constants"]
                        self.assertEqual(constants["public0"]["mean_predict_action_seconds"], 0.20545783716411198)
                for actual in self.rejected(sealed):
                    with mock.patch.object(pc, "hardware_identity", lambda: actual):
                        with self.assertRaisesRegex(Exception, "sealed hardware class"):
                            pc.load_context(root=root, mode="dev")


if __name__ == "__main__":
    unittest.main()
