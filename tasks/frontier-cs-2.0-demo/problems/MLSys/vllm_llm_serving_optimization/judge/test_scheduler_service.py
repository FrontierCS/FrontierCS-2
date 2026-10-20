"""Scheduler sidecar over real localhost HTTP with synthetic batches and workers."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import attestation
import scheduler_service as service
import scheduler_relay
from batch_coordinator import BackendError, PreparedGroup, TestbedTransportError
from batch_run import SCHEMA as PREPARED_SCHEMA
from dataclasses import asdict
from instance_manifest import expected_groups
from serving_eval.tool_auth import ToolAuthSigner, _key_id
from sidecar_catalog import SCHEDULER, TESTBED_PORT, VARIANTS, check_name, instance_index, testbed_name
import test_batch_measurements
from test_attestation import keypair

IDENTITY = {'run_id': 'run', 'attempt_id': 'attempt', 'operation_id': 'runner_one', 'phase': 'train',
            'name': SCHEDULER, 'profile_digest': 'sha256:' + '1' * 64}


class FakeTestbedClient:
    ready_calls = []
    shutdowns = []
    refuse_shutdown = set()

    def __init__(self, group, descriptor, trace, *, signer=None):
        self.descriptor = descriptor

    def ready(self, **kwargs):
        FakeTestbedClient.ready_calls.append(self.descriptor.child_id)

    def shutdown(self):
        if self.descriptor.child_id in FakeTestbedClient.refuse_shutdown:
            raise BackendError('synthetic shutdown refusal')
        FakeTestbedClient.shutdowns.append(self.descriptor.child_id)


class FakeCheckClient:
    calls = []
    fail = set()

    def __init__(self, group, instance_id, *, signer):
        self.name = check_name(instance_index(instance_id))
        self.instance_id = instance_id

    def correctness(self, variant, request):
        FakeCheckClient.calls.append((variant, self.instance_id, threading.current_thread().name))
        if self.instance_id in FakeCheckClient.fail:
            raise BackendError('synthetic worker failure')
        time.sleep(0.01)
        return {'identity': {}, 'variant': variant, 'run_index': VARIANTS.index(variant),
                'reset': None if variant == 'vanilla' else {'head': 'x'},
                'receipt': {'instance_id': self.instance_id, 'request_sha256': request['request_sha256'],
                            'execution': 'complete', 'log': 'PASSED\n'}}


class SchedulerServiceTests(unittest.TestCase):
    def test_candidate_only_scheduler_accepts_one_variant_barrier(self):
        self.service.variants = ('candidate',)
        raw = json.loads(self.batch_job('bfcl', 'candidate', 0))
        raw['all_prepared'] = [raw['prepared']]
        result = self.complete(json.dumps(raw).encode(), 0)
        self.assertEqual(result['variant'], 'candidate')
        self.assertEqual(len(self.batches), 1)
        self.assertEqual(self.service.expected_steps(), [('batch', 'candidate')])
        self.assertEqual(self.http('POST', '/jobs', self.batch_job('bfcl', 'vanilla', 1))[0], 400)

    def test_preinput_readiness_failure_keeps_closed_stage_and_declared_sidecar(self):
        with patch.object(FakeTestbedClient, 'ready', side_effect=TestbedTransportError('connection')):
            status, _ = self.http('POST', '/jobs', self.batch_job('swe', 'vanilla', 0))
            self.assertEqual(status, 202)
            view = self.wait(0)
        expected = {'stage': 'preinput_sidecar_ready',
                    'sidecar_id': testbed_name(0, 'vanilla'), 'candidate_input_forwarded': False}
        self.assertEqual(view['failure'], expected)
        self.assertEqual(self.service.event_view(0, 0)['failure'], expected)
        self.assertNotIn('private body', json.dumps(view))
        self.assertEqual(self.batches, [])

    def test_unclassified_readiness_barrier_retains_signed_generic_terminal(self):
        for error in (BackendError('HTTP 401 private body'), ValueError('identity mismatch')):
            with self.subTest(error=type(error).__name__), \
                 patch.object(FakeTestbedClient, 'ready', side_effect=error):
                document = self.batch_job('swe', 'vanilla', 0)
                prepared = self.service.validate(json.loads(document))
                job = service.Job(0, json.loads(document), document)
                self.service.execute(job, prepared)
                signed = attestation.verify(job.signed, self.public)
                self.assertEqual(signed['payload']['failure'], {
                    'stage': 'scheduler_control', 'cause': 'unclassified',
                    'sidecar_id': None, 'candidate_input_forwarded': None})
                self.assertEqual(job.stream_signed, job.signed)
                self.assertNotIn('private body', json.dumps(job.view()))
        self.assertEqual(self.batches, [])

    def test_per_case_missing_request_does_not_reclassify_timed_terminal(self):
        from dataclasses import replace
        from containment_policy import Boundary, child_failure_metadata
        from terminal_contract import Slot, terminal
        self.service.workload = 'swe'
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.service.journal = Path(temporary.name)
        limits = {'verification_id': 'verification', 'owner_seconds': 10,
                  'cleanup_seconds': 1, 'frame_bytes': 100000, 'stderr_bytes': 16000}
        for variant in VARIANTS:
            descriptor = self.groups('swe')[variant]
            group = replace(service.prepared_group({'schema': PREPARED_SCHEMA, 'group': descriptor}),
                            model_source_digest='sha256:' + 'a' * 64)
            _, batch = test_batch_measurements.MeasurementProjectionTests().make('swe')
            original = service.Job(0, {'variant': variant, 'containment': limits}, b'original-job')
            ids = expected_groups('train')['swe/public']
            slot = Slot('run', 'attempt', 'verification', 'runner_one', 'train', 'swe/public',
                        variant, ids[0], group.model_source_digest, original.request_sha256)
            record = terminal(slot, status='runtime_error', cause='child_exit', elapsed=1.25, cap=10)
            record['candidate_input_dispatched'] = True
            record['failure_context'] = child_failure_metadata(slot, Boundary('swe_case', 'execution'), True)
            batch['episodes'][0].update(episode=None, execution_error={'origin': 'supervised_episode'},
                                        supervised_terminal=record)
            self.service.batches[variant] = group, batch
            self.service.batch_jobs[variant] = original
            requests = {ident: {'instance_id': ident, 'candidate_patch': '', 'timeout_seconds': 1200,
                'request_sha256': 'sha256:' + 'b' * 64} for ident in ids[1:]}
            document = {'schema': service.JOB_SCHEMA, 'kind': 'correctness', 'sequence': 3,
                'nonce': 'cd' * 8, 'variant': variant, 'requests': requests, 'containment': limits}
            self.service.validate(document)
            record['slot']['request_digest'] = 'sha256:' + 'f' * 64
            # Correctness depends on its own measured tests, not inferred row shape.
            self.assertEqual(set(self.service.validate(document)[1]), set(ids[1:]))

    def test_full_selector_subset_and_unknown_worker_fault_preserve_siblings(self):
        from dataclasses import replace
        for mode in ('train', 'eval'):
            for variant in VARIANTS:
                with self.subTest(mode=mode, variant=variant):
                    FakeCheckClient.calls, FakeCheckClient.fail = [], set()
                    group = replace(self.service.validate(json.loads(self.batch_job('swe', variant, 0)))[0], mode=mode)
                    ids = expected_groups(mode)['swe/public']
                    rows = [{'instance_id': ident, 'episode': {'patch': ''}} for ident in ids]
                    # Reproduce a non-null partial episode that old admission
                    # wrongly treated as requiring a correctness request.
                    rows[0]['episode'].update(error='BadRequestError', exit_status='limit_with_patch')
                    self.service.workload = 'swe'
                    self.service.batches[variant] = group, {'episodes': rows}
                    requests = {ident: {'instance_id': ident, 'candidate_patch': '',
                        'timeout_seconds': 1200, 'request_sha256': 'sha256:' + 'a' * 64}
                        for ident in ids[1:]}
                    requests[ids[1]]['candidate_patch'] = 'foreign patch'
                    document = {'schema': service.JOB_SCHEMA, 'kind': 'correctness',
                        'sequence': 10, 'nonce': 'cd' * 8, 'variant': variant, 'requests': requests}
                    prepared = self.service.validate(document)
                    self.assertEqual(set(prepared[1]), set(ids[2:]))
                    raw = json.dumps(document).encode()
                    original = FakeCheckClient.correctness
                    def unknown(client, role, request):
                        if client.instance_id == ids[2]:
                            raise RuntimeError('unknown per-case fault')
                        return original(client, role, request)
                    with patch.object(FakeCheckClient, 'correctness', unknown):
                        result = self.service.correctness(service.Job(10, document, raw), *prepared)
                    self.assertEqual(set(result['receipts']), set(ids[3:]))
                    self.assertEqual({call[1] for call in FakeCheckClient.calls}, set(ids[3:]))

    def setUp(self):
        FakeTestbedClient.ready_calls, FakeTestbedClient.shutdowns, FakeTestbedClient.refuse_shutdown = [], [], set()
        FakeCheckClient.calls, FakeCheckClient.fail = [], set()
        private, self.public = keypair()
        signer = attestation.Signer(private, self.public, IDENTITY)
        tool_signer = ToolAuthSigner(private, self.public, _key_id(self.public),
                                     {'schema_version': 'frontier.tool-auth-context/v1', 'run_id': 'run',
                                      'attempt_id': 'attempt', 'operation_id': 'runner_one', 'sidecar_id': SCHEDULER})
        self.batches = []
        self.service = service.SchedulerService(IDENTITY, signer=signer, tool_signer=tool_signer,
                                                max_response_bytes=4_000_000, run=self.run_group,
                                                testbed_client=FakeTestbedClient, check_client=FakeCheckClient)
        self.server = service.make_server(self.service, host='127.0.0.1', port=0, max_request_bytes=4_000_000)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01})
        self.thread.start()
        self.addCleanup(self.close)
        self.endpoint = 'http://127.0.0.1:' + str(self.server.server_port)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def run_group(self, group, *, jps=None, calibration=False, tool_signer=None):
        self.batches.append((group, jps, calibration, tool_signer is not None))
        _, batch = test_batch_measurements.MeasurementProjectionTests().make(group.workload)
        batch['identity'] = group.identity()
        batch['model_source_digest'] = group.model_source_digest
        return batch

    def http(self, method, route, body=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        request = Request(self.endpoint + route, data=data, method=method,
                          headers={'Content-Type': 'application/json'} if data else {})
        try:
            response = build_opener(ProxyHandler({})).open(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, response.read()

    def groups(self, workload, mode='train'):
        groups = {}
        for variant in VARIANTS:
            beds = []
            if workload == 'swe':
                for ident in expected_groups(mode)['swe/public']:
                    name = testbed_name(instance_index(ident), variant)
                    beds.append({'instance_id': ident, 'endpoint': f'http://{name}:{TESTBED_PORT}',
                                 'child_id': name, 'phase': mode})
            group = PreparedGroup('runner_one', 'runner_one', 'runner_one', mode, workload, 'public',
                                  'http://main:8000/v1', 'sha256:' + variant.ljust(64, '0'))
            groups[variant] = {**asdict(group), 'testbeds': beds}
        return groups

    def batch_job(self, workload, variant, sequence, **overrides):
        groups = self.groups(workload)
        document = {'schema': service.JOB_SCHEMA, 'kind': 'batch', 'sequence': sequence, 'nonce': 'ab' * 8,
                    'variant': variant, 'prepared': {'schema': PREPARED_SCHEMA, 'group': groups[variant]},
                    'all_prepared': [{'schema': PREPARED_SCHEMA, 'group': groups[v]} for v in VARIANTS],
                    'jps': None, 'calibration': False}
        document.update(overrides)
        return json.dumps(document, sort_keys=True, separators=(',', ':')).encode()

    def wait(self, sequence, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, data = self.http('GET', f'/jobs/{sequence}')
            view = json.loads(data)
            if view['state'] != 'running':
                return view
            time.sleep(0.02)
        raise AssertionError('job did not finish')

    def complete(self, raw, sequence):
        status, data = self.http('POST', '/jobs', raw)
        self.assertEqual(status, 202, data)
        view = self.wait(sequence)
        self.assertEqual(view['state'], 'complete', view)
        envelope = attestation.verify(view['signed'], self.public)
        return attestation.bind(envelope, operation_id='runner_one', phase='train', name=SCHEDULER,
                                kind=json.loads(raw)['kind'], sequence=sequence,
                                request_sha256='sha256:' + hashlib.sha256(raw).hexdigest())

    def test_incremental_progress_is_signed_and_durable_before_http_visibility(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.service.journal = Path(temporary.name)
        def progressing(group, *, on_progress, **kwargs):
            batch = self.run_group(group, **kwargs)
            for row in batch['episodes']:
                on_progress('dispatch', {'instance_id': row['instance_id']})
                on_progress('terminal', row)
            return batch
        self.service.run_group = progressing
        raw = self.batch_job('bfcl', 'vanilla', 0)
        self.complete(raw, 0)
        status, encoded = self.http('GET', '/jobs/0/events/0')
        self.assertEqual(status, 200)
        view = json.loads(encoded)
        retained = (Path(temporary.name) / 'job-00-event-00000.json').read_bytes()
        self.assertEqual(json.loads(retained), view['event'])
        event = attestation.verify(view['event'], self.public)
        payload = attestation.bind(event, operation_id='runner_one', phase='train', name=SCHEDULER,
            kind='batch', sequence=0, request_sha256='sha256:' + hashlib.sha256(raw).hexdigest())
        self.assertEqual(payload['event'], 'dispatch')
        self.assertEqual(view['event_count'], 16)
        self.assertEqual(len(list(Path(temporary.name).glob('*event*.json'))), 16)

    def test_swe_phase_runs_three_batches_shutdowns_and_three_correctness_jobs_in_order(self):
        ids = expected_groups('train')['swe/public']
        self.assertEqual(json.loads(self.http('GET', '/ready')[1])['ready'], True)
        for sequence, variant in enumerate(VARIANTS):
            payload = self.complete(self.batch_job('swe', variant, sequence), sequence)
            self.assertEqual(payload['variant'], variant)
            self.assertEqual(set(payload['testbed_shutdown'].values()), {'confirmed'})
            self.assertEqual(set(payload['testbed_shutdown']), {testbed_name(instance_index(i), variant) for i in ids})
        self.assertEqual(len(FakeTestbedClient.ready_calls), 15 + 5 + 5, 'barrier covers all sets once, then own set')
        self.assertEqual(len(self.batches), 3)
        self.assertTrue(all(signed for *_, signed in self.batches))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.service.journal = Path(temporary.name)
        for sequence, variant in enumerate(VARIANTS, start=3):
            requests = {ident: {'instance_id': ident, 'candidate_patch': '', 'timeout_seconds': 1200,
                                'request_sha256': 'sha256:' + ident.encode().hex().ljust(64, '0')[:64]}
                        for ident in ids}
            raw = json.dumps({'schema': service.JOB_SCHEMA, 'kind': 'correctness', 'sequence': sequence,
                              'nonce': 'cd' * 8, 'variant': variant, 'requests': requests},
                             sort_keys=True, separators=(',', ':')).encode()
            payload = self.complete(raw, sequence)
            self.assertEqual(payload['instance_ids'], list(ids))
            self.assertEqual(payload['workers'], 8)
            selector = json.dumps({'sequence': sequence, 'cursor': 0,
                'request_sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
                'variant': variant, 'expected_count': len(ids)}).encode()
            with patch.object(scheduler_relay, 'SCHEDULER', self.endpoint):
                body, metadata = scheduler_relay.correctness_next(
                    selector, time.monotonic() + 5, 4_000_000, 16000)
                replay_body, replay_metadata = scheduler_relay.correctness_next(
                    selector, time.monotonic() + 5, 4_000_000, 16000)
            self.assertEqual((body, metadata), (replay_body, replay_metadata))
            cursor_view = json.loads(metadata)
            cursor_event = attestation.verify(cursor_view['event'], self.public)['payload']
            self.assertEqual(cursor_event['record']['correctness_cursor'], 0)
            self.assertEqual(cursor_event['record']['episode']['receipt_sha256'],
                             'sha256:' + hashlib.sha256(body).hexdigest())
            self.assertIsNone(json.loads(self.http('GET',
                f'/jobs/{sequence}/correctness/{len(ids)}')[1])['event'])
            for ident in ids:
                entry = payload['receipts'][ident]
                status, data = self.http('GET', f'/receipts/{variant}/{instance_index(ident)}')
                self.assertEqual(status, 200)
                self.assertEqual('sha256:' + hashlib.sha256(data).hexdigest(), entry['receipt_sha256'])
                self.assertEqual(len(data), entry['receipt_bytes'])
                self.assertEqual(entry['worker'], check_name(instance_index(ident)))
                self.assertEqual(entry['run_index'], VARIANTS.index(variant))
        self.assertEqual(len(FakeCheckClient.calls), 15)
        self.assertEqual(self.http('POST', '/jobs', self.batch_job('swe', 'vanilla', 6))[0], 409)

    def test_jobs_are_idempotent_by_digest_and_refuse_out_of_order_or_foreign_requests(self):
        raw = self.batch_job('bfcl', 'vanilla', 0)
        self.assertEqual(self.http('POST', '/jobs', raw)[0], 202)
        self.assertEqual(self.http('POST', '/jobs', raw)[0], 202)
        self.wait(0)
        self.assertEqual(len(self.batches), 1)
        self.assertEqual(self.http('POST', '/jobs', self.batch_job('bfcl', 'vanilla', 0, nonce='ef' * 8))[0], 409)
        self.assertEqual(self.http('POST', '/jobs', self.batch_job('bfcl', 'candidate', 1))[0], 409)
        self.assertEqual(self.http('POST', '/jobs', self.batch_job('swe', 'reference', 1))[0], 400)
        foreign = self.groups('bfcl')
        for variant in foreign:
            foreign[variant]['operation_id'] = 'runner_two'
        groups = {'prepared': {'schema': PREPARED_SCHEMA, 'group': foreign['reference']},
                  'all_prepared': [{'schema': PREPARED_SCHEMA, 'group': foreign[v]} for v in VARIANTS]}
        self.assertEqual(self.http('POST', '/jobs', self.batch_job('bfcl', 'reference', 1, **groups))[0], 400)
        self.assertEqual(self.http('POST', '/jobs', self.batch_job('bfcl', 'reference', 1))[0], 202)
        self.wait(1)
        self.assertEqual(self.http('POST', '/jobs', self.batch_job('bfcl', 'candidate', 2))[0], 202)
        self.wait(2)
        correctness = json.dumps({'schema': service.JOB_SCHEMA, 'kind': 'correctness', 'sequence': 3,
                                  'nonce': 'cd' * 8, 'variant': 'vanilla', 'requests': {}}).encode()
        self.assertEqual(self.http('POST', '/jobs', correctness)[0], 409, 'BFCL parents have no correctness jobs')
        self.assertEqual(self.http('GET', '/jobs/9')[0], 404)
        self.assertEqual(self.http('GET', '/receipts/vanilla/0')[0], 404)
        self.assertEqual(self.http('POST', '/jobs', b'{')[0], 400)

    def test_c2_timed_failure_does_not_invent_patch_or_call_worker(self):
        descriptor = self.groups('swe')['candidate']
        descriptor['model_source_digest'] = 'sha256:' + 'a' * 64
        group = service.prepared_group({'schema': PREPARED_SCHEMA,
            'group': descriptor})
        _, batch = test_batch_measurements.MeasurementProjectionTests().make('swe')
        batch.update(identity=group.identity(), model_source_digest=group.model_source_digest)
        ids = expected_groups('train')['swe/public']
        batch['episodes'][0].update(episode=None, execution_error={'origin': 'supervised_episode'},
            failure_context={'boundary': {'kind': 'swe_case', 'stage': 'tool'}})
        self.service.workload = 'swe'
        self.service.batches['candidate'] = (group, batch)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.service.journal = Path(temporary.name)
        requests = {ident: {'instance_id': ident, 'candidate_patch': '', 'timeout_seconds': 1200,
            'request_sha256': 'sha256:' + 'a' * 64} for ident in ids[1:]}
        document = {'schema': service.JOB_SCHEMA, 'kind': 'correctness', 'sequence': 5,
            'nonce': 'cd' * 8, 'variant': 'candidate', 'requests': requests,
            'containment': {'verification_id': 'verification', 'owner_seconds': 10,
                'cleanup_seconds': 1, 'frame_bytes': 100000, 'stderr_bytes': 16000}}
        prepared = self.service.validate(document)
        raw = json.dumps(document, sort_keys=True, separators=(',', ':')).encode()
        job = service.Job(5, document, raw)
        self.service.correctness(job, *prepared)
        self.assertEqual({call[1] for call in FakeCheckClient.calls}, set(ids[1:]))
        self.assertEqual(len(job.correctness_events), len(ids))
        terminals = [attestation.verify(event, self.public)['payload']['record']
                     for event in job.correctness_events]
        self.assertEqual({record['instance_id'] for record in terminals}, set(ids))
        failed = next(record for record in terminals if record['instance_id'] == ids[0])
        self.assertIsNone(failed['episode'])
        self.assertEqual(failed['execution_error']['origin'], 'timed_case_boundary')
        del batch['episodes'][0]['failure_context']
        self.assertEqual(set(self.service.validate(document)[1]), set(ids[1:]))

    def test_unconfirmed_shutdown_and_worker_failure_preserve_siblings(self):
        ids = expected_groups('train')['swe/public']
        FakeTestbedClient.refuse_shutdown = {testbed_name(0, 'vanilla')}
        payload = self.complete(self.batch_job('swe', 'vanilla', 0), 0)
        self.assertEqual(payload['testbed_shutdown'][testbed_name(0, 'vanilla')], 'uncertain:BackendError')
        self.complete(self.batch_job('swe', 'reference', 1), 1)
        self.complete(self.batch_job('swe', 'candidate', 2), 2)
        FakeCheckClient.fail = {ids[1]}
        requests = {ident: {'instance_id': ident, 'candidate_patch': '', 'timeout_seconds': 1200,
                            'request_sha256': 'sha256:' + '0' * 64} for ident in ids}
        raw = json.dumps({'schema': service.JOB_SCHEMA, 'kind': 'correctness', 'sequence': 3,
                          'nonce': 'cd' * 8, 'variant': 'vanilla', 'requests': requests}).encode()
        self.assertEqual(self.http('POST', '/jobs', raw)[0], 202)
        view = self.wait(3)
        self.assertEqual(view['state'], 'complete')
        result = attestation.verify(view['signed'], self.public)['payload']
        self.assertEqual(set(result['receipts']), set(ids) - {ids[1]})
        bad = dict(requests)
        bad[ids[0]] = dict(requests[ids[0]], candidate_patch='other')
        raw = json.dumps({'schema': service.JOB_SCHEMA, 'kind': 'correctness', 'sequence': 4,
                          'nonce': 'cd' * 8, 'variant': 'reference', 'requests': bad}).encode()
        self.assertEqual(self.http('POST', '/jobs', raw)[0], 202)
        result = attestation.verify(self.wait(4)['signed'], self.public)['payload']
        self.assertEqual(set(result['receipts']), set(ids) - {ids[0], ids[1]})


if __name__ == '__main__':
    unittest.main()
