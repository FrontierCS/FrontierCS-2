"""Identical captures reuse only a completed oracle check in the same workload."""
import deployment_test_guard  # noqa: F401 # isort: skip

import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

JUDGE = Path(__file__).resolve().parents[1] / (
    'tasks/frontier-cs-2.0-demo/problems/Arch/clusterwise_spgemm/judge')


def exact_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class CaptureIdentityTests(unittest.TestCase):
    def test_cache_reads_all_bytes_and_never_accepts_same_size_changed_bytes(self):
        scope = exact_definitions(JUDGE / 'product_oracle.py',
            {'_capture_unchanged', '_validated_capture'}, {'hashlib': hashlib, 'os': os})
        body = b'a' * (2 * 1024 * 1024 + 7)
        expected = SimpleNamespace(_validated_products={(len(body), hashlib.sha256(body).hexdigest())})
        with tempfile.TemporaryFile() as stream:
            stream.write(body); stream.flush()
            calls = []
            result = scope['_validated_capture'](stream, expected, os.fstat(stream.fileno()),
                                                  lambda: calls.append(stream.tell()))
            self.assertEqual(result['product_bytes'], len(body))
            self.assertEqual(result['product_sha256'], hashlib.sha256(body).hexdigest())
            self.assertGreaterEqual(len(calls), 5)
            stream.seek(-1, 2); stream.write(b'b'); stream.flush()
            self.assertIsNone(scope['_validated_capture'](
                stream, expected, os.fstat(stream.fileno()), lambda: None))
            with self.assertRaisesRegex(RuntimeError, 'trusted deadline'):
                scope['_validated_capture'](stream, expected, os.fstat(stream.fileno()),
                    lambda: (_ for _ in ()).throw(RuntimeError('trusted deadline')))

    def test_trusted_budget_is_distinct_from_native_child_time_limit(self):
        scope = exact_definitions(JUDGE / 'trusted_runner.py',
            {'TrustedAggregateDeadline', 'remaining'}, {'time': SimpleNamespace(monotonic=lambda: 10)})
        self.assertEqual(scope['remaining'](20, 'validation'), 10)
        with self.assertRaises(scope['TrustedAggregateDeadline']):
            scope['remaining'](10, 'validation')
        with self.assertRaises(TimeoutError):
            scope['remaining'](10, 'process', child=True)
        self.assertFalse(issubclass(scope['TrustedAggregateDeadline'], (ValueError, TimeoutError)))

    def test_aggregate_deadline_in_candidate_phase_emits_native_evaluator_failure(self):
        runner = exact_definitions(JUDGE / 'trusted_runner.py',
            {'TrustedAggregateDeadline', 'remaining'}, {'time': SimpleNamespace(monotonic=lambda: 10)})
        evaluator = exact_definitions(JUDGE / 'evaluator.py', {'evaluate', '_is_resource_failure'},
            {'time': SimpleNamespace(monotonic=lambda: 0), 'tempfile': tempfile, 'Path': Path,
             'os': os, 'json': json})
        evaluator.update({'Context': lambda **kwargs: SimpleNamespace(receipt={}),
            'ROOT': JUDGE, 'LIMITS': SimpleNamespace(evaluator_seconds=60), 'CONFIG': {},
            'install': lambda *args: None, '_workloads': lambda final: [],
            'sha': lambda path: str(path), 'require_disk_capture': lambda *args: None,
            'remaining': lambda *args, **kwargs: runner['remaining'](10, 'validation')})
        # Setup uses no remaining() call until candidate patch application.
        feedback_spec = importlib.util.spec_from_file_location('beat_spec_feedback',
            JUDGE.parent / 'verifier/beat_spec_feedback.py')
        feedback = importlib.util.module_from_spec(feedback_spec); feedback_spec.loader.exec_module(feedback)
        adapter_spec = importlib.util.spec_from_file_location('clusterwise_cache_adapter',
            JUDGE.parent / 'verifier/fcs2_verifier.py')
        adapter = importlib.util.module_from_spec(adapter_spec)
        with patch.dict(sys.modules, {'beat_spec_feedback': feedback}):
            adapter_spec.loader.exec_module(adapter)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            root = Path(directory); payload = root / 'payload.tar'; dataset = root / 'dataset.tar'
            dataset.write_bytes(b'dataset')
            with tarfile.open(payload, 'w') as archive:
                member = tarfile.TarInfo('payload/solution.patch'); member.size = 5
                archive.addfile(member, io.BytesIO(b'patch'))
            with patch.object(adapter, 'load_evaluator', return_value=evaluator['evaluate']):
                result = adapter.verify(bundle=JUDGE, entrypoint='payload/solution.patch',
                    payload=payload, dataset=dataset, mode='eval', result_path=root / 'result.json',
                    evidence_path=root / 'evidence.tar.zst')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['failure']['code'], 'evaluator_error')
            self.assertNotIn('raw_score', result)
            self.assertNotIn('private_metrics', result)
            self.assertEqual(json.loads((root / 'result.json').read_bytes()), result)


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'native oracle equivalence needs numpy')
class CompleteOracleCacheTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        spec = importlib.util.spec_from_file_location('clusterwise_product_oracle_test', JUDGE / 'product_oracle.py')
        self.oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.oracle)
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.root = root
        self.rows = 16
        self.ids = [0, 3, 11]
        self.values = np.array([[row + column / 10 + 1 for column in self.ids]
                                for row in range(self.rows)], dtype='<f8')
        self.write_expected(root / 'expected', self.values)
        self.expected = self.oracle.Expected(root / 'expected',
            {'max_matrix_rows': 32, 'max_oracle_product_nnz': 1000})
        self.limits = {'max_product_bytes': 1024 * 1024}

    def tearDown(self):
        self.expected.close(); self.directory.cleanup()

    def write_expected(self, root, values):
        root.mkdir()
        (root / 'metadata.json').write_text(json.dumps(
            {'rows': self.rows, 'groups': 2, 'width': 8, 'numeric_nnz': self.rows * 3}))
        self.np.arange(0, self.rows * 3 + 1, 3, dtype='<i8').tofile(root / 'rowptr.i64le')
        self.np.tile(self.ids, self.rows).astype('<i8').tofile(root / 'columns.i64le')
        values.astype('<f8').tofile(root / 'values.f64le')

    def capture(self, order=(11, 0, 3)):
        np = self.np
        pointers = np.array([0, len(order), 2 * len(order)], dtype='<i8')
        values = [self.values[group * 8:(group + 1) * 8, self.ids.index(column)]
                  for group in range(2) for column in order]
        return (struct.pack('<Q5q', self.oracle.MAGIC, self.rows, self.rows, 2, 8, 2 * len(order))
            + pointers.tobytes() + np.tile(order, 2).astype('<i8').tobytes()
            + np.asarray(values, dtype='<f8').tobytes())

    def validate(self, body, expected=None):
        with tempfile.TemporaryFile() as stream:
            stream.write(body); stream.flush()
            return self.oracle.validate_product(stream, expected or self.expected, self.limits)

    def test_cached_and_full_oracle_are_equivalent_for_arbitrary_column_order(self):
        for order in ((11, 0, 3), (0, 3, 11), (3, 11, 0)):
            body = self.capture(order)
            self.expected._validated_products.clear()
            full = self.validate(body)
            with patch.object(self.oracle.np, 'sort', side_effect=AssertionError('cache missed')):
                self.assertEqual(self.validate(body), full)
            self.expected._validated_products.clear()
            self.assertEqual(self.validate(body), full)

    def test_changed_same_length_capture_must_be_revalidated_and_rejected(self):
        body = self.capture(); self.validate(body)
        corrupted = body[:-8] + struct.pack('<d', -123.0)
        with self.assertRaisesRegex(ValueError, 'incorrect numeric capture'):
            self.validate(corrupted)
        self.assertEqual(len(self.expected._validated_products), 1)

    def test_cache_is_never_shared_between_workloads(self):
        body = self.capture(); self.validate(body)
        self.write_expected(self.root / 'different', self.values + 5)
        different = self.oracle.Expected(self.root / 'different',
            {'max_matrix_rows': 32, 'max_oracle_product_nnz': 1000})
        try:
            with self.assertRaisesRegex(ValueError, 'incorrect numeric capture'):
                self.validate(body, different)
        finally:
            different.close()

    def test_missing_duplicate_and_changed_during_hash_are_rejected(self):
        self.validate(self.capture())
        with self.assertRaisesRegex(ValueError, 'missing significant expected value'):
            self.validate(self.capture((0, 3)))
        with self.assertRaisesRegex(ValueError, 'duplicate capture columns'):
            self.validate(self.capture((0, 0, 3)))
        with patch.object(self.oracle, '_capture_unchanged', return_value=False):
            with self.assertRaisesRegex(ValueError, 'capture changed during validation'):
                self.validate(self.capture())


if __name__ == '__main__':
    unittest.main()
