"""Bindings for complete measurement projection; no serving/transport acceptance."""
import copy
from dataclasses import asdict, replace
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from batch_coordinator import PreparedGroup
from batch_measurements import project_complete_batch, expected_input_digest
from instance_manifest import expected_groups
from paired_timing import paired_speedups
from reference_verdict import MeasurementError
from serving_eval import bfcl
from serving_eval.settings import EpisodeSettings
from serving_eval.accuracy import CorrectnessRuntimeError
from serving_eval.accuracy import build_correctness_request
from sidecar_catalog import check_name, instance_index


class MeasurementProjectionTests(unittest.TestCase):
    def make(self, workload='bfcl'):
        group = PreparedGroup('operation', 'group', 'placement', 'train', workload, 'public', 'http://model:8000/v1', 'sha256:model')
        ids = expected_groups('train')[workload + '/public']
        rows = []
        answers = bfcl.load_memory_answers(ids) if workload == 'bfcl' else None
        for index, ident in enumerate(ids):
            episode = {'instance_id': ident, 'started_monotonic': 10., 'ended_monotonic': 42.,
                       'latency_seconds': 32., 'exit_status': 'force_quit' if workload == 'bfcl' else 'incomplete',
                       'patch': '', 'error': ''}
            if workload == 'bfcl':
                episode['output'] = answers[ident][0] if index % 2 == 0 else '__known_wrong_answer__'
            else:
                episode['correct'] = index % 2 == 0
            rows.append({'instance_id': ident, 'episode': episode, 'trace': [], 'requires_error_attribution': False,
                         'patch_sha256': 'sha256:' + hashlib.sha256(b'').hexdigest()})
        batch = {'schema': 'frontier.vllm-local-batch/v1', 'identity': group.identity(), 'mode': 'train',
                 'group': workload + '/public', 'model_source_digest': group.model_source_digest,
                 'expected_instances': len(ids), 'episodes': rows, 'calibration_only': False,
                 'jps': {'swe': 0.5, 'bfcl': 1.0}[workload],
                 'arrival_seed': {'swe': 20260604, 'bfcl': 20260605}[workload],
                 'input_digest': expected_input_digest(group)}
        return group, batch

    def test_complete_wrong_and_force_quit_keep_real_latency_and_correctness(self):
        group, batch = self.make()
        observations = project_complete_batch(batch, group)
        self.assertEqual(len(observations), 8)
        self.assertEqual([row.correct for row in observations], [index % 2 == 0 for index in range(8)])
        self.assertTrue(all(row.latency_seconds == 32 and row.source_status == 'force_quit' for row in observations))

    def test_real_bfcl_max_steps_result_preserves_original_measured_semantics(self):
        group, batch = self.make()
        output = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='[core_memory_retrieve_all()]'))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: output)))
        instances = bfcl.load_memory_instances(expected_groups('train')['bfcl/public'])
        for row, instance in zip(batch['episodes'], instances):
            trace = []
            result = bfcl.run_memory_instance(instance, client=client,
                        settings=replace(EpisodeSettings(), memory_max_steps=1), tool_trace=trace)
            self.assertEqual((result.exit_status, result.error), ('force_quit', 'max_steps'))
            row['episode'], row['trace'] = asdict(result), trace
        projected = project_complete_batch(batch, group)
        self.assertEqual([row.latency_seconds for row in projected],
                         [row['episode']['latency_seconds'] for row in batch['episodes']])
        answers = bfcl.load_memory_answers(expected_groups('train')['bfcl/public'])
        self.assertEqual([row.correct for row in projected],
                         [bfcl.memory_correct(row['episode']['output'], answers[row['instance_id']])
                          for row in batch['episodes']])
        self.assertTrue(all('correct' not in row['episode'] for row in batch['episodes']))
        for edit in ('origin', 'attribution', 'other_error', 'other_status'):
            bad = copy.deepcopy(batch)
            row = bad['episodes'][0]
            if edit == 'origin': row['trace'].append({'error_origin': 'model_api_unclassified'})
            elif edit == 'attribution': row['requires_error_attribution'] = True
            elif edit == 'other_error': row['episode']['error'] = 'TimeoutError'
            else: row['episode']['exit_status'] = 'ok'
            with self.subTest(edit=edit), self.assertRaises(MeasurementError):
                project_complete_batch(bad, group)

    def test_missing_and_unclassified_failures_are_unscored_not_silently_dropped(self):
        group, batch = self.make()
        bad = copy.deepcopy(batch)
        bad['episodes'].pop()
        with self.assertRaises(MeasurementError):
            project_complete_batch(bad, group)
        for change in ({'requires_error_attribution': True}, {'trace': [{'error_origin': 'model_api_unclassified'}]}, {'episode': None}):
            bad = copy.deepcopy(batch)
            bad['episodes'][0].update(change)
            with self.assertRaises(MeasurementError):
                project_complete_batch(bad, group)

    def test_cpu_case_failures_keep_siblings_and_original_denominator(self):
        group, batch = self.make('swe')
        ids = expected_groups('train')['swe/public']
        failed = frozenset(ids[:2])
        receipts = {}
        for ident in ids[2:]:
            request = build_correctness_request(ident, '', variant='candidate', identity={
                **group.identity(), 'instance_id': ident, 'phase': 'train',
                'child_id': check_name(instance_index(ident))})
            receipts[ident] = {'instance_id': ident, 'request_sha256': request['request_sha256'],
                'execution': 'candidate_patch_rejected', 'candidate_patch_applied': False,
                'patch_apply_log': 'ordinary original parser receipt'}
        for row in batch['episodes'][:2]:
            row.update(episode=None, execution_error='remote case process killed',
                       requires_error_attribution=True)
        observations = project_complete_batch(batch, group, swe_receipts=receipts,
            variant='candidate', candidate_failures=failed)
        self.assertEqual(tuple(row.instance_id for row in observations), ids)
        self.assertTrue(all(row.execution == 'candidate_error' and row.latency_seconds is None
                            and row.correct is False for row in observations[:2]))
        self.assertTrue(all(row.execution == 'complete' and row.latency_seconds == 32
                            for row in observations[2:]))
        for variant in ['vanilla', 'reference', None]:
            with self.subTest(variant=variant), self.assertRaises(MeasurementError):
                project_complete_batch(batch, group, swe_receipts=receipts,
                    variant=variant, candidate_failures=failed)

    def test_cpu_model_case_failures_preserve_all_variants_original_timing_and_denominator(self):
        for variant in ('vanilla', 'reference', 'candidate'):
            group, batch = self.make('swe')
            ids = expected_groups('train')['swe/public']
            failed = frozenset(ids[:2])
            receipts = {}
            for ident in ids[2:]:
                request = build_correctness_request(ident, '', variant=variant, identity={
                    **group.identity(), 'instance_id': ident, 'phase': 'train',
                    'child_id': check_name(instance_index(ident))})
                receipts[ident] = {'instance_id': ident, 'request_sha256': request['request_sha256'],
                    'execution': 'candidate_patch_rejected', 'candidate_patch_applied': False,
                    'patch_apply_log': 'unchanged native parsed receipt'}
            with self.subTest(variant=variant):
                observations = project_complete_batch(batch, group, swe_receipts=receipts,
                    variant=variant, model_case_failures=failed)
                self.assertEqual(tuple(row.instance_id for row in observations), ids)
                self.assertTrue(all(row.latency_seconds == 32 and row.execution == 'complete'
                                    and row.correct is False for row in observations))
                self.assertTrue(all(row.source_status == 'cpu_observed_model_case_failure'
                                    for row in observations[:2]))
                self.assertTrue(all(row.source_status == 'incomplete' for row in observations[2:]))
                for invalid in (frozenset({'foreign'}), list(failed)):
                    with self.assertRaises(MeasurementError):
                        project_complete_batch(batch, group, swe_receipts=receipts,
                            variant=variant, model_case_failures=invalid)
        group, batch = self.make('bfcl')
        with self.assertRaises(MeasurementError):
            project_complete_batch(batch, group, variant='reference',
                model_case_failures=frozenset([batch['episodes'][0]['instance_id']]))

    def test_only_candidate_http_error_reaches_fixed_failure_penalty_input(self):
        group, batch = self.make()
        row = batch['episodes'][0]
        row['episode'].update(exit_status='api_error', error='BadRequestError', output='')
        row['trace'] = [{'kind': 'model', 'error_origin': 'model_api_response',
                         'error_type': 'BadRequestError'}]
        row['requires_error_attribution'] = True
        for variant in (None, 'vanilla', 'reference'):
            with self.subTest(variant=variant), self.assertRaises(MeasurementError):
                project_complete_batch(batch, group, variant=variant)
        observations = project_complete_batch(batch, group, variant='candidate')
        self.assertEqual(observations[0].execution, 'candidate_error')
        self.assertFalse(observations[0].correct)
        self.assertEqual(observations[0].latency_seconds, 32)
        timing = paired_speedups(
            [observations[0].instance_id],
            {observations[0].instance_id: 32},
            {observations[0].instance_id: observations[0].latency_seconds},
            candidate_api_error_ids=[observations[0].instance_id],
        )
        self.assertEqual(timing.ratios[observations[0].instance_id], 1 / 8)
        bad = copy.deepcopy(batch)
        bad['episodes'][0]['trace'][0]['error_origin'] = 'model_transport_or_client_unclassified'
        with self.assertRaises(MeasurementError):
            project_complete_batch(bad, group, variant='candidate')

    def test_group_model_time_and_correctness_bindings(self):
        group, batch = self.make()
        for changed in (replace(group, group_id='replacement'), replace(group, model_source_digest='other')):
            with self.assertRaises(MeasurementError):
                project_complete_batch(batch, changed)
        for field, value in [('ended_monotonic', 50), ('latency_seconds', 0), ('correct', 1), ('exit_status', 'api_error')]:
            bad = copy.deepcopy(batch)
            bad['episodes'][0]['episode'][field] = value
            with self.assertRaises(MeasurementError):
                project_complete_batch(bad, group)

    def test_plan_binding_rejects_other_rate_seed_inputs_or_calibration(self):
        group, batch = self.make()
        for field, value in (('jps', 1.25), ('jps', True), ('arrival_seed', 1),
                             ('input_digest', 'sha256:other'), ('calibration_only', True)):
            wrong = copy.deepcopy(batch)
            wrong[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(MeasurementError):
                project_complete_batch(wrong, group)
        calibrated = copy.deepcopy(batch)
        calibrated['calibration_only'] = True
        self.assertEqual(len(project_complete_batch(calibrated, group, calibration=True)), 8)

    def test_swe_requires_exact_patch_bound_original_parser_receipts(self):
        group, batch = self.make('swe')
        ids = expected_groups('train')['swe/public']
        receipts = {ident: {'instance_id': ident, 'request_sha256': 'wrong'} for ident in ids}
        with self.assertRaises(MeasurementError):
            project_complete_batch(batch, group)
        with self.assertRaises(CorrectnessRuntimeError):
            project_complete_batch(batch, group, swe_receipts=receipts)
        bad = copy.deepcopy(batch)
        bad['episodes'][0]['episode']['patch'] = 'changed after measurement'
        with self.assertRaises(MeasurementError):
            project_complete_batch(bad, group, swe_receipts=receipts)
        # This fixture checks invocation/binding only; parser execution has its
        # own real cached-testbed evidence and original-harness regression tests.
        seen = []
        def parse(request, receipt):
            seen.append((request['instance_id'], request['candidate_patch'], receipt))
            return {'correct': False}
        with patch('batch_measurements.parse_test_receipt', parse):
            result = project_complete_batch(batch, group, swe_receipts=receipts)
        self.assertEqual([row[0] for row in seen], list(ids))
        self.assertTrue(all(row[1] == '' for row in seen))
        self.assertTrue(all(not row.correct and row.source_status == 'incomplete' for row in result))


if __name__ == '__main__':
    unittest.main()
