"""Synthetic trusted-scheduler phase joins; no vLLM, SWE, Driver or Modal."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / 'verifier'))
import phase_run
from batch_run import prepared_group
from driver_client import SidecarExecutionError
from instance_manifest import expected_groups
from reference_verdict import InstanceObservation
from sidecar_catalog import TESTBED_PORT, expected_sidecars, instance_index, testbed_name
import test_batch_measurements as measurement_fixtures
from test_runtime_plan import funded_fixture


class PhaseJoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = []
        self.fail = False
        self.unconfirmed_shutdown = False
        self.plan, self.config = funded_fixture()
        self.session = SimpleNamespace(
            mode='train', runner_id='runner_one', run=self.run_parent,
            sidecars={'public_key': 'ssh-ed25519 ' + 'A' * 68, 'names': list(expected_sidecars('train')),
                      'states': {}},
            client=SimpleNamespace(driver_url='http://driver',
                verification_id='verification_one', _token='synthetic-only',
                deadline=999999999))
        owner = self

        class Relay:
            def __init__(self, session, *, public_key, relay_limits, max_stderr_bytes, evidence,
                         payload_validator=None):
                owner.assertEqual(public_key, session.sidecars['public_key'])
                owner.assertEqual(relay_limits, owner.plan['relay'])
                Path(evidence).mkdir(parents=True, exist_ok=False)
                self.sequence = 0
                self.session = session
                self.payload_validator = payload_validator

            def job(self, kind, variant, fields, *, timeout, name):
                owner.assertEqual(kind, 'batch')
                owner.assertEqual(self.sequence, phase_run.execution_variants(owner.config).index(variant))
                owner.assertEqual(timeout, owner.plan['relay']['batch_timeout_seconds']['train'])
                group = prepared_group(json.loads(json.dumps(fields['prepared'])))
                owner.assertEqual(len(fields['all_prepared']), len(phase_run.execution_variants(owner.config)))
                owner.events.append('scheduler-' + group.workload + '-' + variant)
                if owner.fail:
                    raise SidecarExecutionError('synthetic scheduler failure')
                _, batch = measurement_fixtures.MeasurementProjectionTests().make(group.workload)
                batch['identity'] = group.identity()
                batch['model_source_digest'] = group.model_source_digest
                shutdown = {bed.child_id: 'confirmed' for bed in group.testbeds}
                if owner.unconfirmed_shutdown and shutdown:
                    shutdown[next(iter(shutdown))] = 'uncertain:TimeoutError'
                payload = {'variant': variant, 'batch': batch, 'testbed_shutdown': shutdown}
                if self.payload_validator is not None:
                    self.payload_validator(kind, variant, payload)
                self.sequence += 1
                return payload
        self.relay = Relay

    def run_parent(self, argv, **kwargs):
        action = argv[3]
        request = json.loads(kwargs['stdin'])
        variant = request['variant']
        self.events.append('parent-' + action + '-' + variant)
        output = kwargs['output']
        output.mkdir(parents=True)
        if action == 'start':
            result = {'schema': 'frontier.vllm-parent-server/v1', 'state': 'ready',
                'variant': variant, 'model_source_digest': request['model_source_digest'],
                'pid': 123, 'process_start_ticks': 456,
                'model_endpoint': 'http://main:8000/v1'}
        else:
            result = {'schema': 'frontier.vllm-parent-server/v1', 'state': 'terminated',
                'variant': variant, 'model_source_digest': request['model_source_digest'],
                'remote_artifacts_removed': True}
        path = output / 'stdout'
        path.write_text(json.dumps(result))
        return path

    def correctness(self, batch, group, **kwargs):
        self.events.append('correctness-' + kwargs['variant'])
        self.assertEqual(kwargs['request_bytes'], self.plan['correctness']['request_bytes'])
        self.assertEqual(kwargs['timeout'], self.plan['relay']['correctness_timeout_seconds']['train'])
        self.assertEqual(len([e for e in self.events if e.startswith('parent-stop-')]),
                         len(phase_run.execution_variants(self.config)),
                         'correctness starts only after every variant server stopped')
        return [InstanceObservation(i, 32, False, source_status='incomplete')
                for i in expected_groups('train')['swe/public']]

    def invoke(self, **overrides):
        kwargs = dict(candidate_patch=b'', evidence=self.root / 'phase',
            startup_seconds=30, stop_seconds=5, max_stdout_bytes=1_000_000,
            max_stderr_bytes=1000)
        kwargs.update(overrides)
        with patch.object(phase_run, 'validate_projection', lambda value: value), \
             patch.object(phase_run, 'SchedulerRelay', self.relay), \
             patch.object(phase_run, 'run_swe_correctness', self.correctness):
            return phase_run.run_phase(self.session, self.config, self.plan, **kwargs)

    def test_candidate_only_group_has_no_baseline_servers_episodes_or_testbeds(self):
        self.config['sidecars'] = {k: v for k, v in self.config['sidecars'].items()
                                   if not k.endswith(('-vanilla', '-reference'))}
        self.config['sidecars']['scheduler']['args'].extend(['--variants', 'candidate'])
        self.session.sidecars['names'] = list(expected_sidecars('train', variants=('candidate',)))
        result = self.invoke(selected_group='swe/public')
        self.assertEqual(set(result), {'candidate'})
        self.assertEqual(self.events, ['parent-start-candidate', 'scheduler-swe-candidate',
                                      'parent-stop-candidate', 'correctness-candidate'])
        self.assertFalse(any('vanilla' in name or 'reference' in name
                             for name in self.session.sidecars['names']))

    def test_complete_public_three_variant_batches_and_operator_scoring(self):
        result = self.invoke()
        self.assertEqual(len([e for e in self.events if e.startswith('scheduler-')]), 6)
        self.assertEqual(len([e for e in self.events if e.startswith('parent-start-')]), 6)
        self.assertEqual(len([e for e in self.events if e.startswith('parent-stop-')]), 6)
        self.assertEqual(len([e for e in self.events if e.startswith('correctness-')]), 3)
        self.assertEqual(result['reference_verdict']['expected_groups'], 2)
        self.assertFalse(result['pass'])
        self.assertEqual(result['score'], 0)
        self.assertTrue((self.root / 'phase/operator-result.json').is_file())

    def test_declared_testbeds_are_described_by_name_for_each_variant(self):
        seen = {}
        original = self.relay.job

        def observed(relay, kind, variant, fields, **kwargs):
            group = prepared_group(json.loads(json.dumps(fields['prepared'])))
            if group.workload == 'swe':
                seen[variant] = group.testbeds
            return original(relay, kind, variant, fields, **kwargs)
        with patch.object(self.relay, 'job', observed):
            self.invoke(selected_group='swe/public')
        for variant, beds in seen.items():
            ids = expected_groups('train')['swe/public']
            self.assertEqual([bed.instance_id for bed in beds], list(ids))
            for bed in beds:
                name = testbed_name(instance_index(bed.instance_id), variant)
                self.assertEqual((bed.child_id, bed.endpoint, bed.phase),
                                 (name, f'http://{name}:{TESTBED_PORT}', 'train'))

    def test_selected_bfcl_group_runs_three_variants_without_operator_score(self):
        observations = self.invoke(selected_group='bfcl/public')
        self.assertEqual({variant: set(groups) for variant, groups in observations.items()},
                         {variant: {'bfcl/public'} for variant in phase_run.VARIANTS})
        self.assertEqual(len([event for event in self.events if event.startswith('scheduler-')]), 3)
        self.assertFalse(any(event.startswith('correctness-') for event in self.events))
        self.assertTrue((self.root / 'phase/group-result.json').is_file())
        self.assertFalse((self.root / 'phase/operator-result.json').exists())

    def test_failed_scheduler_stops_parent_and_never_scores(self):
        self.fail = True
        with self.assertRaises(SidecarExecutionError):
            self.invoke()
        self.assertIn('parent-stop-vanilla', self.events)
        self.assertFalse(any(e.startswith('correctness-') for e in self.events))
        self.assertFalse((self.root / 'phase/operator-result.json').exists())

    def test_lifecycle_duplicate_and_nonfinite_json_are_remote_protocol_failures(self):
        original = self.session.run
        for index, suffix in enumerate([b',"state":"ready"}', b',"untrusted_extra":NaN}']):
            def malformed(*args, **kwargs):
                path = original(*args, **kwargs)
                path.write_bytes(path.read_bytes().rstrip()[:-1] + suffix)
                return path
            self.session.run = malformed
            with self.subTest(suffix=suffix), self.assertRaises(SidecarExecutionError):
                self.invoke(evidence=self.root / ('malformed-' + str(index)),
                            selected_group='bfcl/public')

    def test_unconfirmed_testbed_shutdown_blocks_correctness(self):
        self.unconfirmed_shutdown = True
        with self.assertRaisesRegex(SidecarExecutionError, 'confirm shutdown'):
            self.invoke(selected_group='swe/public')
        self.assertIn('parent-stop-vanilla', self.events)
        self.assertFalse(any(e.startswith('correctness-') for e in self.events))

    def test_missing_sidecar_view_or_hidden_rates_fail_before_execution(self):
        self.session.sidecars = None
        with self.assertRaises(SidecarExecutionError):
            self.invoke()
        self.session.sidecars = {'public_key': 'ssh-ed25519 ' + 'A' * 68, 'names': [], 'states': {}}
        with self.assertRaises(SidecarExecutionError):
            self.invoke(selected_group='swe/public')
        self.session.sidecars['names'] = list(expected_sidecars('train'))
        with self.assertRaises(SidecarExecutionError):
            self.invoke(hidden_jps={'swe': 1.0, 'bfcl': 1.25})
        self.assertEqual(self.events, [])

    def test_final_plans_four_groups_with_public_and_configured_hidden_rates(self):
        from batch_coordinator import FINAL_HIDDEN_JPS
        self.session.mode = 'eval'
        self.session.sidecars['names'] = list(expected_sidecars('eval'))
        documents = []

        class Relay:
            def __init__(self, session, **kwargs):
                Path(kwargs['evidence']).mkdir(parents=True, exist_ok=False)

            def job(self, kind, variant, fields, *, timeout, name):
                group = prepared_group(json.loads(json.dumps(fields['prepared'])))
                documents.append((group.workload + '/' + group.rate, variant,
                                  fields['jps'], fields['calibration'], timeout))
                return {'variant': variant, 'batch': {'group': group.workload + '/' + group.rate},
                        'testbed_shutdown': {bed.child_id: 'confirmed' for bed in group.testbeds}}

        def rows(batch, group, **kwargs):
            return [InstanceObservation(i, 1.0, True) for i in expected_groups('eval')[batch['group']]]

        def final(vanilla, candidate, reference):
            for observed in (vanilla, candidate, reference):
                self.assertEqual(set(observed), set(expected_groups('eval')))
            return {'score': 0.0, 'pass': False}
        with patch.object(phase_run, 'validate_projection', lambda value: value), \
             patch.object(phase_run, 'SchedulerRelay', Relay), \
             patch.object(phase_run, 'validate_batch_plan'), \
             patch.object(phase_run, 'project_complete_batch', rows), \
             patch.object(phase_run, 'run_swe_correctness', rows), \
             patch.object(phase_run, 'evaluate_final', final):
            result = phase_run.run_phase(self.session, self.config, self.plan,
                candidate_patch=b'', evidence=self.root / 'phase', startup_seconds=30,
                stop_seconds=5, max_stdout_bytes=1_000_000, max_stderr_bytes=1000,
                hidden_jps=FINAL_HIDDEN_JPS)
        self.assertEqual(result, {'score': 0.0, 'pass': False})
        timeout = self.plan['relay']['batch_timeout_seconds']['eval']
        expected = [(group, variant, FINAL_HIDDEN_JPS[group.split('/')[0]]
                     if group.endswith('/hidden') else None, False, timeout)
                    for group in expected_groups('eval') for variant in phase_run.VARIANTS]
        self.assertEqual(documents, expected)
        self.assertEqual({group for group, *_ in documents},
                         {'swe/public', 'bfcl/public', 'swe/hidden', 'bfcl/hidden'})
        operator = json.loads((self.root / 'phase/operator-result.json').read_text())
        self.assertEqual((operator['mode'], operator['calibration_only']), ('eval', False))

    def test_final_refuses_missing_or_other_hidden_rates(self):
        self.session.mode = 'eval'
        self.session.sidecars['names'] = list(expected_sidecars('eval'))
        for hidden in (None, {'swe': 1.0, 'bfcl': 1.5}, {'swe': 0.5, 'bfcl': 1.0}):
            with self.subTest(hidden=hidden), self.assertRaisesRegex(
                    SidecarExecutionError, 'configured hidden arrival rates'):
                self.invoke(hidden_jps=hidden, evidence=self.root / 'refused')
        self.assertEqual(self.events, [])


if __name__ == '__main__':
    unittest.main()
