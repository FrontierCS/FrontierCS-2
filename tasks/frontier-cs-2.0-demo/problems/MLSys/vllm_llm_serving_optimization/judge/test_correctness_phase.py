"""Join tests use a synthetic signed relay; real wire/worker tests are separate."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'verifier'))
import correctness_phase
from batch_transport import prepared_testbeds
from driver_client import SidecarExecutionError
from sidecar_catalog import check_name, expected_sidecars, instance_index
import test_batch_measurements


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


class SyntheticRelay:
    """Answers correctness jobs the way the scheduler would, without signatures."""

    def __init__(self, owner):
        self.owner = owner
        self.session = SimpleNamespace(runner_id='runner_one', mode='train')
        self.jobs = []
        self.fetched = []
        self.fail_after = None
        self.receipts = {}

    def receipt_for(self, request):
        return {'instance_id': request['instance_id'], 'request_sha256': request['request_sha256'],
                'execution': 'candidate_patch_rejected', 'candidate_patch_applied': False,
                'patch_apply_log': 'synthetic rejected-patch receipt; no real worker execution'}

    def job(self, kind, variant, fields, *, timeout, name):
        self.owner.assertEqual(kind, 'correctness')
        self.jobs.append((variant, fields, timeout, name))
        summaries = {}
        for ident, request in fields['requests'].items():
            raw = canonical(self.receipt_for(request))
            self.receipts[(variant, instance_index(ident))] = raw
            summaries[ident] = {'request_sha256': request['request_sha256'],
                                'receipt_sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
                                'receipt_bytes': len(raw), 'run_index': ['vanilla', 'reference', 'candidate'].index(variant),
                                'reset': None if variant == 'vanilla' else {'head': 'x', 'removed_changed_or_new_ignored': 0},
                                'worker': check_name(instance_index(ident))}
        self.owner.mutate_summaries(summaries)
        return {'variant': variant, 'instance_ids': list(fields['requests']), 'workers': 8, 'receipts': summaries}

    def receipt(self, variant, index, *, expected_sha256, expected_bytes, name):
        if self.fail_after is not None and len(self.fetched) >= self.fail_after:
            raise SidecarExecutionError('synthetic transport failure')
        self.fetched.append((variant, index, name))
        raw = self.receipts[(variant, index)]
        if len(raw) != expected_bytes or 'sha256:' + hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise SidecarExecutionError('fetched correctness receipt differs from its signed summary')
        return json.loads(raw)


class CorrectnessPhaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.evidence = Path(self.temporary.name) / 'evidence'
        self.group, self.batch = test_batch_measurements.MeasurementProjectionTests().make('swe')
        beds = prepared_testbeds('train', 'candidate', expected_sidecars('train'))
        self.group = replace(self.group, operation_id='runner_one', group_id='runner_one',
                             placement_id='runner_one', testbeds=beds)
        self.batch['identity'] = self.group.identity()
        self.relay = SyntheticRelay(self)
        self.summary_change = None

    def mutate_summaries(self, summaries):
        if self.summary_change is not None:
            self.summary_change(summaries)

    def test_unknown_request_error_is_per_case_for_all_variants(self):
        import cpu_swe_accuracy as accuracy
        original = accuracy._build_test_request
        first = self.batch['episodes'][0]['instance_id']
        for variant in ('vanilla', 'reference', 'candidate'):
            failures = set()
            def build(ident, *args, **kwargs):
                if ident == first:
                    raise RuntimeError('previously unknown model request error')
                return original(ident, *args, **kwargs)
            with self.subTest(variant=variant), patch.object(accuracy, '_build_test_request', build):
                ids, requests = correctness_phase.correctness_requests(self.batch, self.group,
                    variant=variant, calibration=False, request_bytes=1_000_000,
                    construction_failures=failures)
                self.assertEqual(failures, {first})
                self.assertEqual(set(requests), set(ids) - {first})

    def test_request_input_failure_is_incorrect_for_all_variants(self):
        from cpu_swe_accuracy import TrustedCorrectnessError
        for variant in ('vanilla', 'reference', 'candidate'):
            failures = set()
            with self.subTest(variant=variant), patch.object(correctness_phase, 'build_correctness_request',
                    side_effect=TrustedCorrectnessError('fixed CPU data digest mismatch')):
                ids, requests = correctness_phase.correctness_requests(self.batch, self.group,
                    variant=variant, calibration=False, request_bytes=1_000_000,
                    construction_failures=failures)
                self.assertEqual(failures, set(ids))
                self.assertEqual(requests, {})

    def invoke(self, **changes):
        kwargs = dict(relay=self.relay, variant='candidate', evidence=self.evidence,
                      request_bytes=1_000_000, timeout=1500)
        kwargs.update(changes)
        return correctness_phase.run_swe_correctness(self.batch, self.group, **kwargs)

    def test_trusted_parsed_incorrect_results_keep_measured_time(self):
        for variant in ('vanilla', 'reference'):
            with self.subTest(variant=variant):
                rows = self.invoke(variant=variant, evidence=self.evidence / variant)
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(row.correct is False and row.latency_seconds == 32 for row in rows))

    def test_missing_trusted_correctness_receipt_is_typed_unavailable(self):
        self.relay.fail_after = 0
        for variant in ('vanilla', 'reference'):
            with self.subTest(variant=variant), self.assertRaises(correctness_phase.TrustedMeasurementError):
                self.invoke(variant=variant, evidence=self.evidence / variant)

    def test_all_original_patches_flow_through_one_signed_job_and_bound_receipts(self):
        observations = self.invoke()
        ids = [row.instance_id for row in observations]
        self.assertEqual(ids, [bed.instance_id for bed in self.group.testbeds])
        self.assertTrue(all(not row.correct and row.latency_seconds == 32 for row in observations))
        self.assertEqual(len(self.relay.jobs), 1)
        variant, fields, timeout, name = self.relay.jobs[0]
        self.assertEqual((variant, timeout, name), ('candidate', 1500, 'candidate-correctness'))
        self.assertEqual(set(fields['requests']), set(ids))
        for ident, request in fields['requests'].items():
            self.assertEqual(request['candidate_patch'], '')
            self.assertEqual(request['instance_id'], ident)
        self.assertEqual([index for _, index, _ in self.relay.fetched], [instance_index(i) for i in ids])
        summary = json.loads((self.evidence / 'summary.json').read_text())
        self.assertEqual(summary['correct'], 0)
        self.assertFalse(summary['score_ready'])
        self.assertEqual(set(summary['resets']), set(ids))

    def test_original_eight_worker_setting_is_carried_in_the_signed_summary(self):
        self.assertEqual(correctness_phase.SWE_CORRECTNESS_WORKERS, 8)
        original = self.relay.job

        def fewer(kind, variant, fields, **kwargs):
            payload = original(kind, variant, fields, **kwargs)
            payload['workers'] = 4
            return payload
        with patch.object(self.relay, 'job', fewer):
            with self.assertRaises(SidecarExecutionError):
                self.invoke()
        self.assertEqual(self.relay.fetched, [])

    def test_candidate_http_error_keeps_all_correctness_requests(self):
        row = self.batch['episodes'][0]
        row['episode'].update(exit_status='api_error', error='BadRequestError')
        row['trace'] = [{'kind': 'model', 'error_origin': 'model_api_response',
                         'error_type': 'BadRequestError'}]
        row['requires_error_attribution'] = True

        def request(ident, candidate_patch, **kwargs):
            return {'instance_id': ident, 'candidate_patch': candidate_patch,
                    'request_sha256': 'sha256:' + ident.encode().hex().ljust(64, '0')[:64]}

        def parsed(request_value, receipt):
            return {'correct': False}

        with patch.object(correctness_phase, 'build_correctness_request', request), \
                patch.object(correctness_phase, 'parse_test_receipt', parsed), \
                patch('batch_measurements.build_correctness_request', request), \
                patch('batch_measurements.parse_test_receipt', parsed):
            observations = self.invoke()
        self.assertEqual(len(self.relay.jobs[0][1]['requests']), 5)
        self.assertEqual(observations[0].execution, 'candidate_error')
        self.assertFalse(observations[0].correct)

    def test_summary_must_bind_request_worker_order_and_reset(self):
        for change in ('request', 'worker', 'order', 'reset', 'missing'):
            def mutate(summaries, change=change):
                first = next(iter(summaries))
                if change == 'request':
                    summaries[first]['request_sha256'] = 'sha256:' + '0' * 64
                elif change == 'worker':
                    summaries[first]['worker'] = check_name(49)
                elif change == 'order':
                    summaries[first]['run_index'] = 0
                elif change == 'reset':
                    summaries[first]['reset'] = None
                else:
                    summaries.pop(first)
            self.summary_change = mutate
            self.relay.fetched = []
            with self.subTest(change=change), self.assertRaises(SidecarExecutionError):
                self.invoke(evidence=self.evidence / change)
            self.assertEqual(self.relay.fetched, [])

    def test_receipt_hash_mismatch_or_transport_failure_keeps_partial_evidence(self):
        self.relay.fail_after = 1
        with self.assertRaises(SidecarExecutionError):
            self.invoke()
        error = json.loads((self.evidence / 'error.json').read_text())
        self.assertEqual(len(error['completed_instance_ids']), 1)
        self.assertFalse(error['score_ready'])
        self.assertNotIn('score', error)
        self.assertFalse((self.evidence / 'summary.json').exists())
        self.relay.fail_after = None
        tampered = {key: value[:-1] + b' ' for key, value in self.relay.receipts.items()}

        def swapped(variant, index, **kwargs):
            self.relay.receipts[(variant, index)] = tampered[(variant, index)]
            return SyntheticRelay.receipt(self.relay, variant, index, **kwargs)
        with patch.object(self.relay, 'receipt', swapped):
            with self.assertRaisesRegex(SidecarExecutionError, 'differs from its signed summary'):
                self.invoke(evidence=self.evidence / 'tampered')

    def test_foreign_parent_or_unattributed_patch_fails_before_the_job(self):
        foreign = replace(self.group, operation_id='runner_other', group_id='runner_other',
                          placement_id='runner_other')
        with self.assertRaises(SidecarExecutionError):
            correctness_phase.run_swe_correctness(self.batch, foreign, relay=self.relay, variant='candidate',
                                                  evidence=self.evidence / 'foreign', request_bytes=1_000_000, timeout=1500)
        broken = copy.deepcopy(self.batch)
        broken['episodes'][2]['requires_error_attribution'] = True
        with self.assertRaisesRegex(ValueError, 'live CPU correctness ownership'):
            correctness_phase.run_swe_correctness(broken, self.group, relay=self.relay, variant='candidate',
                                                  evidence=self.evidence / 'broken', request_bytes=1_000_000, timeout=1500)
        with self.assertRaisesRegex(ValueError, 'live CPU correctness ownership'):
            self.invoke(request_bytes=16, evidence=self.evidence / 'small')
        self.assertEqual(self.relay.jobs, [])


if __name__ == '__main__':
    unittest.main()
