"""Whole-batch files with the existing real-localhost model protocol fixture."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from batch_coordinator import BackendError
from batch_run import SCHEMA, execute, main, prepared_group
from instance_manifest import expected_groups
from serving_eval import bfcl
from test_batch_coordinator import Fixture


class BatchFileTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()
        self.addCleanup(self.fixture.close)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / 'prepared.json'
        self.output = self.root / 'result'

    def document(self, group=None):
        return {'schema': SCHEMA, 'group': asdict(group or self.fixture.group('bfcl'))}

    def write(self, group=None):
        self.prepared.write_text(json.dumps(self.document(group)))

    def test_cli_retains_all_public_instances_and_original_schedule_without_score(self):
        self.write()
        ids = expected_groups('train')['bfcl/public']
        answers = bfcl.load_memory_answers(ids)
        self.fixture.answers = {ident: answers[ident][0] for ident in ids}
        main(['--prepared', str(self.prepared), '--output', str(self.output)])
        result = json.loads((self.output / 'batch.json').read_text())
        self.assertEqual([row['instance_id'] for row in result['episodes']],
                         list(expected_groups('train')['bfcl/public']))
        self.assertEqual((result['jps'], result['arrival_seed']), (1.0, 20260605))
        self.assertTrue(all('correct' not in row['episode'] for row in result['episodes']))
        self.assertTrue(all(row['episode']['output'] == self.fixture.answers[row['instance_id']]
                            for row in result['episodes']))
        self.assertFalse(result['score_ready'])
        self.assertNotIn('score', result)
        self.assertEqual((self.output / 'prepared.json').read_bytes(), self.prepared.read_bytes())
        before = len(self.fixture.requests)
        with self.assertRaises(FileExistsError):
            execute(self.prepared, self.output)
        self.assertEqual(len(self.fixture.requests), before)

    def test_bad_readiness_retains_execution_error_and_never_scores(self):
        self.write(self.fixture.group('swe'))
        self.fixture.bad_ready = True
        with self.assertRaises(BackendError):
            execute(self.prepared, self.output)
        error = json.loads((self.output / 'error.json').read_text())
        self.assertEqual(error['origin'], 'batch_setup_or_runtime_unclassified')
        self.assertFalse(error['score_ready'])
        self.assertFalse((self.output / 'batch.json').exists())
        self.assertEqual(self.fixture.requests, [])

    def test_closed_prepared_interface_rejects_secret_and_runtime_override_fields(self):
        document = self.document()
        for key in ('token', 'env', 'args', 'image'):
            changed = json.loads(json.dumps(document))
            changed['group'][key] = 'not-accepted'
            with self.assertRaises(BackendError):
                prepared_group(changed)
        document['group']['model_endpoint'] = 'https://user:password@example.invalid/v1'
        with self.assertRaises(BackendError):
            prepared_group(document)
        self.assertEqual(self.fixture.requests, [])

    def test_hidden_requires_the_configured_final_rate_and_never_new_jps(self):
        self.write(replace(self.fixture.group('bfcl'), mode='eval', rate='hidden'))
        for index, settings in enumerate(({}, {'jps': 1.0}, {'jps': 1.5},
                                          {'jps': 1.5, 'calibration': True})):
            with self.assertRaises(BackendError):
                execute(self.prepared, self.root / str(index), **settings)
        self.assertEqual(self.fixture.requests, [])


if __name__ == '__main__':
    unittest.main()
