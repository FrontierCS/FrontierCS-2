"""CPU immutable-source and Driver identity binding; no platform calls."""
import deployment_test_guard  # noqa: F401
from pathlib import Path
import sys
import tempfile
import unittest

JUDGE = Path(__file__).resolve().parents[1] / 'tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization/judge'
sys.path.insert(0, str(JUDGE))
from controller_identity import content_contract, load_context, load_limits
from controller_contract_build import emit
from terminal_contract import Slot, canonical


class ControllerIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'judge').mkdir()
        (self.root / 'judge/controller.py').write_bytes(b'fixed trusted source\n')
        self.slot = Slot('run', 'attempt', 'verification', 'operation', 'eval', 'swe/public',
                         'candidate', 'case', 'sha256:' + 'a' * 64, 'sha256:' + 'b' * 64)
        self.contract = content_contract(self.root, ['judge/controller.py'],
            maximum_contract_bytes=8192, maximum_source_bytes=8192)
        self.path = self.root / 'controller-contract.json'
        self.path.write_bytes(canonical(self.contract))
        self.environment = {'FRONTIER_RUN_ID': 'run', 'FRONTIER_ATTEMPT_ID': 'attempt',
            'FRONTIER_VERIFICATION_ID': 'verification',
            'FRONTIER_TASK_CONFIG_DIGEST': 'sha256:' + 'c' * 64,
            'FRONTIER_VERIFIER_IMAGE_DIGEST': 'sha256:' + 'd' * 64}

    def load(self, **changes):
        args = dict(environment=self.environment, contract_path=self.path,
            source_root=self.root, slots=[self.slot],
            maximum_contract_bytes=8192, maximum_source_bytes=8192)
        return load_context(**dict(args, **changes))

    def test_separate_source_contract_and_executable_image_binding(self):
        context = self.load()
        self.assertEqual(context.executable_image_digest, self.environment['FRONTIER_VERIFIER_IMAGE_DIGEST'])
        self.assertNotEqual(context.executable_image_digest, context.source_content_digest)
        self.assertNotIn(context.executable_image_digest.encode(), self.path.read_bytes())
        self.assertNotEqual(context.source_content_digest, context.contract_digest)

    def test_missing_or_malformed_driver_identity_is_release_blocker(self):
        for field in self.environment:
            for value in [None, '', 'x' * 257]:
                with self.subTest(field=field, value=value), self.assertRaises((KeyError, ValueError)):
                    environment = dict(self.environment)
                    if value is None:
                        del environment[field]
                    else:
                        environment[field] = value
                    self.load(environment=environment)

    def test_modified_missing_and_symlinked_source_is_rejected(self):
        source = self.root / 'judge/controller.py'
        source.write_bytes(b'modified')
        with self.assertRaises(ValueError):
            self.load()
        source.unlink()
        with self.assertRaises(ValueError):
            self.load()
        source.symlink_to(self.path)
        with self.assertRaises(ValueError):
            self.load()

    def test_path_escape_duplicate_keys_and_allocation_caps_rejected(self):
        for path in ['../escape', '/absolute', 'judge//controller.py']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                content_contract(self.root, [path], maximum_contract_bytes=8192, maximum_source_bytes=8192)
        with self.assertRaises(ValueError):
            self.load(maximum_source_bytes=1)
        with self.assertRaises(ValueError):
            self.load(maximum_contract_bytes=1)
        self.path.write_bytes(b'{"files":{},"files":{}}')
        with self.assertRaises(ValueError):
            self.load()

    def test_build_artifacts_measure_exact_source_and_contract_allocations(self):
        (self.root / 'judge/__pycache__').mkdir()
        (self.root / 'judge/__pycache__/mutable.py').write_bytes(b'not shipped source')
        contract = self.root / 'built-contract.json'
        limits = self.root / 'built-limits.json'
        emitted = emit(self.root, ['judge'], [], contract, limits)
        loaded = load_limits(limits)
        self.assertEqual(loaded['maximum_contract_bytes'], contract.stat().st_size)
        self.assertEqual(loaded['maximum_source_bytes'], len(b'fixed trusted source\n'))
        self.assertEqual(loaded['maximum_source_bytes'], emitted['maximum_source_bytes'])
        self.assertEqual(self.load(contract_path=contract, **loaded), self.load())
        self.assertEqual(contract.stat().st_mode & 0o777, 0o444)
        with self.assertRaises(ValueError):
            emit(self.root, ['judge'], [], contract, limits)

    def test_build_contract_binds_fixed_json_and_excludes_mutable_bytecode(self):
        (self.root / 'judge/manifest.json').write_bytes(b'{"fixed":true}')
        contract, limits = self.root / 'json-contract.json', self.root / 'json-limits.json'
        emit(self.root, ['judge'], [], contract, limits)
        bounds = load_limits(limits)
        self.load(contract_path=contract, **bounds)
        (self.root / 'judge/manifest.json').write_bytes(b'{"fixed":false}')
        with self.assertRaises(ValueError):
            self.load(contract_path=contract, **bounds)

    def test_allocation_noncanonical_duplicates_and_impossible_sizes_rejected(self):
        path = self.root / 'limits.json'
        for raw in [b'{"maximum_source_bytes":1,"maximum_source_bytes":2}',
                    b' ' * 4096, b'{}', b'{"schema":"unknown"}']:
            path.write_bytes(raw)
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                load_limits(path)
