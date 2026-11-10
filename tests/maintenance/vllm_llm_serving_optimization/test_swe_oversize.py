"""Bounded model-patch dispositions retain the existing scheduler trust chain."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import attestation
import scheduler_service
import test_scheduler_service as scheduler_fixture
import test_correctness_phase as phase_fixture
import test_swe_correctness_service as worker_fixture
import test_batch_measurements as measurement_fixture
import batch_measurements
import correctness_phase
from batch_run import SCHEMA as PREPARED_SCHEMA
from types import SimpleNamespace
import tempfile
from instance_manifest import expected_groups
from serving_eval import accuracy, swe_test_worker
from sidecar_catalog import VARIANTS, check_name, instance_index


class OversizeTests(unittest.TestCase):
    def setUp(self):
        self.ident = accuracy.load_swe_cases('train')[0]['instance_id']
        self.identity = {'operation_id': 'runner_one', 'group_id': 'runner_one',
                         'placement_id': 'runner_one', 'instance_id': self.ident,
                         'child_id': check_name(instance_index(self.ident)), 'phase': 'train'}
        self.large = 'x' * accuracy.SWE_CORRECTNESS_REQUEST_BYTES

    def request(self, text=None, variant='vanilla', identity=None):
        return accuracy.build_correctness_request(self.ident, self.large if text is None else text,
            identity=self.identity if identity is None else identity, variant=variant)

    def receipt(self, request):
        identity = {'base_commit': accuracy._case(request['instance_id'])['base_commit'],
                    'observed_head': 'base', 'content_matches_fixed_base': True, 'fresh_worktree': True}
        worker = swe_test_worker.PreparedSWEWorker(instance_id=request['instance_id'], max_output_bytes=65536)
        with patch.object(swe_test_worker, 'inspect_testbed', return_value=identity), \
                patch.object(worker, 'execute', side_effect=AssertionError('oversize patch must never execute')):
            receipt = worker.run_correctness(request)
        self.assertTrue(worker.used)
        return receipt

    def test_exact_full_http_envelope_boundary_and_json_escaping(self):
        empty = accuracy.build_test_request(self.ident, '')
        budget = accuracy.SWE_CORRECTNESS_REQUEST_BYTES
        spare = budget - len(accuracy.correctness_envelope(empty, self.identity, 'vanilla'))
        exact = self.request('x' * spare)
        self.assertNotEqual(exact['schema'], accuracy.OVERSIZE_REQUEST_SCHEMA)
        self.assertEqual(len(accuracy.correctness_envelope(exact, self.identity, 'vanilla')), budget)
        over = self.request('x' * (spare + 1))
        self.assertEqual(over['original_request_bytes'], budget + 1)
        self.assertLess(len(accuracy.correctness_envelope(over, self.identity, 'vanilla')), budget)
        escaped = self.request('"\\\n' * (budget // 4))
        original = accuracy.build_test_request(self.ident, '"\\\n' * (budget // 4))
        self.assertEqual(escaped['original_request_bytes'],
                         len(accuracy.correctness_envelope(original, self.identity, 'vanilla')))
        self.assertNotIn('candidate_patch', escaped)

    def test_all_variants_are_incorrect_with_bound_hash_size_and_real_environment_inspection(self):
        for variant in VARIANTS:
            request = self.request(variant=variant)
            receipt = self.receipt(request)
            parsed = accuracy.parse_test_receipt(request, receipt)
            self.assertFalse(parsed['correct'])
            self.assertEqual(parsed['reason'], 'candidate_patch_oversize')
            self.assertEqual(parsed['oversize_evidence']['candidate_patch_bytes'], len(self.large))
            self.assertEqual(parsed['oversize_evidence']['candidate_patch_sha256'],
                             hashlib.sha256(self.large.encode()).hexdigest())
            self.assertEqual(request['variant'], variant)
        worker = swe_test_worker.PreparedSWEWorker(instance_id=self.ident, max_output_bytes=65536)
        with patch.object(swe_test_worker, 'inspect_testbed',
                          side_effect=accuracy.CorrectnessRuntimeError('bad fixed base')):
            with self.assertRaises(accuracy.CorrectnessRuntimeError):
                worker.run_correctness(self.request())

    def test_unbound_or_incomplete_receipt_remains_runtime_failure(self):
        request = self.request()
        receipt = self.receipt(request)
        mutations = [lambda r: r.update(request_sha256='0' * 64),
                     lambda r: r.update(instance_id='other'),
                     lambda r: r['oversize_evidence'].update(candidate_patch_bytes=1),
                     lambda r: r['oversize_evidence'].update(candidate_patch_sha256='0' * 64),
                     lambda r: r.update(execution='setup_failed'),
                     lambda r: r.update(tests_started=True),
                     lambda r: r['testbed'].update(content_matches_fixed_base=False)]
        for mutate in mutations:
            broken = copy.deepcopy(receipt); mutate(broken)
            with self.subTest(mutation=mutate), self.assertRaises(accuracy.CorrectnessRuntimeError):
                accuracy.parse_test_receipt(request, broken)
        for key, value in [('original_request_bytes', 1), ('request_limit_bytes', 2**30),
                           ('candidate_patch_bytes', -1), ('candidate_patch_sha256', 'bad')]:
            broken = copy.deepcopy(request); broken[key] = value
            # Even a freshly hashed forged marker cannot relax the fixed bound or schema.
            broken['request_sha256'] = accuracy._json_sha({k:v for k,v in broken.items() if k!='request_sha256'})
            with self.subTest(field=key), self.assertRaises(accuracy.CorrectnessRuntimeError):
                accuracy.parse_test_receipt(broken, receipt)

    def scheduler(self):
        fixture = scheduler_fixture.SchedulerServiceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = fixture.run_group
        def run(group, **kwargs):
            batch = original(group, **kwargs)
            row = batch['episodes'][0]
            row['episode']['patch'] = self.large
            row['patch_sha256'] = 'sha256:' + hashlib.sha256(self.large.encode()).hexdigest()
            return batch
        fixture.service.run_group = run
        for sequence, variant in enumerate(VARIANTS):
            fixture.complete(fixture.batch_job('swe', variant, sequence), sequence)
        return fixture

    def document(self, fixture, variant='vanilla'):
        group, batch = fixture.service.batches[variant]
        requests = {}
        for row in batch['episodes']:
            ident = row['instance_id']
            identity = {**group.identity(), 'instance_id': ident,
                        'child_id': check_name(instance_index(ident)), 'phase': group.mode}
            requests[ident] = accuracy.build_correctness_request(ident, row['episode']['patch'],
                                                               identity=identity, variant=variant)
        return {'schema': scheduler_service.JOB_SCHEMA, 'kind': 'correctness', 'sequence': 3,
                'nonce': 'cd' * 8, 'variant': variant, 'requests': requests}

    def test_scheduler_recomputes_marker_from_its_owned_batch_and_rejects_forgery(self):
        fixture = self.scheduler()
        good = self.document(fixture)
        self.assertIn(self.ident, fixture.service.validate(good)[1])
        # A forged per-case marker is dropped from the accepted set (the case is incorrect);
        # it cannot invalidate sibling cases (test_scheduler_service preserve_siblings).
        for field, value in [('candidate_patch_bytes', 1), ('candidate_patch_sha256', '0'*64),
                             ('original_request_bytes', 1), ('candidate_patch_json_bytes', 1),
                             ('instance_id', 'other'), ('variant', 'candidate')]:
            forged = copy.deepcopy(good); marker = forged['requests'][self.ident]; marker[field] = value
            marker['request_sha256'] = accuracy._json_sha({k:v for k,v in marker.items() if k!='request_sha256'})
            with self.subTest(field=field):
                self.assertNotIn(self.ident, fixture.service.validate(forged)[1])
        # A structurally valid marker for a real small patch is not accepted either.
        fixture.service.batches['vanilla'][1]['episodes'][0]['episode']['patch'] = ''
        self.assertNotIn(self.ident, fixture.service.validate(good)[1])
        fixture.service.batches.clear()
        with self.assertRaises(scheduler_service.JobError):
            fixture.service.validate(good)

    def test_worker_rejects_forged_size_for_small_patch_even_when_metadata_is_well_formed(self):
        from serving_eval.swe_patch_disposition import oversize_request
        marker = oversize_request(self.ident, 'small', identity=self.identity, variant='vanilla',
                                  original_request_bytes=accuracy.SWE_CORRECTNESS_REQUEST_BYTES + 1)
        with self.assertRaises(accuracy.CorrectnessRuntimeError):
            self.receipt(marker)

    def test_authenticated_worker_keeps_fixed_order_and_reset_failure_is_not_incorrect(self):
        fixture = worker_fixture.CorrectnessServiceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.service.prepare()
        test = self

        class Worker:
            def __init__(self, **kwargs):
                pass

            def run_correctness(self, request):
                return test.receipt(request)

        fixture.service.worker_factory = Worker
        def body(variant):
            return {'identity': fixture.identity, 'variant': variant,
                    'request': accuracy.build_correctness_request(fixture.identity['instance_id'], self.large,
                                identity=fixture.identity, variant=variant)}

        self.assertEqual(fixture.request('POST', '/correctness', body('vanilla'), signed=False)[0], 401)
        self.assertEqual(fixture.request('POST', '/correctness', body('reference'))[0], 409)
        wrong = body('vanilla'); wrong['request']['variant'] = 'candidate'
        self.assertEqual(fixture.request('POST', '/correctness', wrong)[0], 409)
        status, result = fixture.request('POST', '/correctness', body('vanilla'))
        self.assertEqual((status, result['run_index'], result['reset']), (200, 0, None))
        self.assertEqual(result['receipt']['execution'], 'candidate_patch_oversize')
        fixture.reset.fail = True
        status, result = fixture.request('POST', '/correctness', body('reference'))
        self.assertEqual((status, result['error_type']), (503, 'ResetError'))
        self.assertEqual(fixture.service.completed, ['vanilla'])
        fixture.reset.fail = False
        for index, variant in enumerate(VARIANTS[1:], 1):
            status, result = fixture.request('POST', '/correctness', body(variant))
            self.assertEqual((status, result['run_index']), (200, index))
            self.assertIsInstance(result['reset'], dict)
        self.assertEqual(fixture.reset.resets, 2)

    def test_existing_scheduler_signature_covers_disposition_receipt_and_tampering_fails(self):
        fixture = self.scheduler()
        original = scheduler_fixture.FakeCheckClient.correctness
        test = self
        def result(client, variant, request):
            if request.get('schema') == accuracy.OVERSIZE_REQUEST_SCHEMA:
                return {'receipt': test.receipt(request), 'run_index': VARIANTS.index(variant),
                        'reset': None if variant == 'vanilla' else {'head': 'base'}}
            return original(client, variant, request)
        with patch.object(scheduler_fixture.FakeCheckClient, 'correctness', result):
            document = self.document(fixture)
            payload = fixture.complete(json.dumps(document).encode(), 3)
        summary = payload['receipts'][self.ident]
        raw = fixture.service.receipts[('vanilla', instance_index(self.ident))]
        self.assertEqual(summary['receipt_sha256'], 'sha256:' + hashlib.sha256(raw).hexdigest())
        self.assertFalse(accuracy.parse_test_receipt(document['requests'][self.ident], json.loads(raw))['correct'])
        signed = copy.deepcopy(fixture.service.jobs[3].signed)
        signed['envelope']['payload']['receipts'][self.ident]['receipt_bytes'] += 1
        with self.assertRaises(attestation.AttestationError):
            attestation.verify(signed, fixture.public)
        unsigned = copy.deepcopy(fixture.service.jobs[3].signed)
        unsigned.pop('signature')
        with self.assertRaises(attestation.AttestationError):
            attestation.verify(unsigned, fixture.public)

    def test_complete_projection_keeps_remaining_instances_latency_and_original_denominator(self):
        fixture = phase_fixture.CorrectnessPhaseTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        row = fixture.batch['episodes'][0]
        row['episode']['patch'] = self.large
        row['patch_sha256'] = 'sha256:' + hashlib.sha256(self.large.encode()).hexdigest()
        original = fixture.relay.receipt_for
        fixture.relay.receipt_for = lambda request: (
            self.receipt(request) if request.get('schema') == accuracy.OVERSIZE_REQUEST_SCHEMA else original(request))
        for variant in VARIANTS:
            observations = fixture.invoke(variant=variant, evidence=fixture.evidence / variant,
                                          request_bytes=accuracy.SWE_CORRECTNESS_REQUEST_BYTES)
            self.assertEqual(len(observations), 5)
            self.assertTrue(all(row.latency_seconds == 32 for row in observations))
            self.assertFalse(observations[0].correct)
            disposition = json.loads((fixture.evidence / variant / 'parsed-000.json').read_text())
            self.assertEqual(disposition['reason'], 'candidate_patch_oversize')

    def test_full_final_signed_pipeline_keeps_all_48_instances(self):
        fixture = scheduler_fixture.SchedulerServiceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.service.identity['phase'] = 'eval'
        fixture.service.signer.identity['phase'] = 'eval'
        fixture.service.max_response_bytes = 64 * 1024**2
        ids = expected_groups('eval')['swe/public']
        test = self

        def run(group, **kwargs):
            if hasattr(test, 'accepted_batches'):
                variant = next(v for v in VARIANTS if group.model_source_digest == 'sha256:' + v.ljust(64, '0'))
                batch = copy.deepcopy(test.accepted_batches[variant])
            else:
                _, batch = measurement_fixture.MeasurementProjectionTests().make('swe')
                template = batch['episodes'][0]
                batch['episodes'] = []
                for ident in ids:
                    row = copy.deepcopy(template)
                    row['instance_id'] = row['episode']['instance_id'] = ident
                    batch['episodes'].append(row)
                batch.update(mode='eval', expected_instances=len(ids))
            batch.update(identity=group.identity(), model_source_digest=group.model_source_digest,
                         group='swe/public', jps=0.5, input_digest=batch_measurements.expected_input_digest(group))
            if not hasattr(test, 'accepted_batches') or variant == 'vanilla':
                row = next(r for r in batch['episodes'] if r['instance_id'] == self.ident)
                row['episode']['patch'] = self.large
                row['patch_sha256'] = 'sha256:' + hashlib.sha256(self.large.encode()).hexdigest()
            return batch

        fixture.service.run_group = run

        def completed(raw, sequence):
            status, data = fixture.http('POST', '/jobs', raw)
            self.assertEqual(status, 202, data)
            view = fixture.wait(sequence)
            self.assertEqual(view['state'], 'complete')
            envelope = attestation.verify(view['signed'], fixture.public)
            return attestation.bind(envelope, operation_id='runner_one', phase='eval', name='scheduler',
                kind=json.loads(raw)['kind'], sequence=sequence,
                request_sha256='sha256:' + hashlib.sha256(raw).hexdigest())

        groups = fixture.groups('swe', mode='eval')
        for sequence, variant in enumerate(VARIANTS):
            completed(fixture.batch_job('swe', variant, sequence,
                prepared={'schema': PREPARED_SCHEMA, 'group': groups[variant]},
                all_prepared=[{'schema': PREPARED_SCHEMA, 'group': groups[v]}
                              for v in VARIANTS]), sequence)

        class Relay:
            session = SimpleNamespace(runner_id='runner_one', mode='eval')

            def job(self, kind, variant, fields, **kwargs):
                raw = json.dumps({'schema': scheduler_service.JOB_SCHEMA, 'kind': kind,
                    'sequence': 3 + VARIANTS.index(variant), 'nonce': 'ef'*8, 'variant': variant, **fields}).encode()
                return completed(raw, 3 + VARIANTS.index(variant))

            def receipt(self, variant, index, *, expected_sha256, expected_bytes, **kwargs):
                status, raw = fixture.http('GET', f'/receipts/{variant}/{index}')
                self_status = status == 200 and len(raw) == expected_bytes
                if not self_status or 'sha256:' + hashlib.sha256(raw).hexdigest() != expected_sha256:
                    raise ValueError('receipt differs from signed summary')
                return json.loads(raw)

        def worker(client, variant, request):
            receipt = (test.receipt(request) if request.get('schema') == accuracy.OVERSIZE_REQUEST_SCHEMA else
                {'instance_id': request['instance_id'], 'request_sha256': request['request_sha256'],
                 'execution': 'candidate_patch_rejected', 'candidate_patch_applied': False,
                 'patch_apply_log': 'synthetic patch rejection for unrelated instances'})
            return {'receipt': receipt, 'run_index': VARIANTS.index(variant),
                    'reset': None if variant == 'vanilla' else {'head': 'base'}}

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(scheduler_fixture.FakeCheckClient, 'correctness', worker):
            for variant in VARIANTS:
                group, batch = fixture.service.batches[variant]
                observations = correctness_phase.run_swe_correctness(batch, group,
                    relay=Relay(), variant=variant, evidence=Path(directory)/variant,
                    request_bytes=accuracy.SWE_CORRECTNESS_REQUEST_BYTES, timeout=10800)
                self.assertEqual(len(observations), 48)
                self.assertEqual([o.instance_id for o in observations], list(ids))
                self.assertEqual([o.latency_seconds for o in observations],
                                 [r['episode']['latency_seconds'] for r in batch['episodes']])
                self.assertFalse(observations[list(ids).index(self.ident)].correct)


if __name__ == '__main__':
    unittest.main()
