"""CPU-only synthetic fixtures; never production calibration evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from build_fixed_targets import materialize
from fixed_targets import (CALIBRATION_SCHEMA, ENVIRONMENT_FIELDS, EXCLUDED_RUN,
    build_targets, canonical_bytes, correctness_allowance, digest, evaluate_fixed, load_targets, validate_targets)
from instance_manifest import expected_groups
from reference_verdict import InstanceObservation, MeasurementError, evaluate_final, evaluate_train


ARTIFACT_PROVENANCE = {}
ENV = {field: hashlib.sha256(field.encode()).hexdigest() for field in ENVIRONMENT_FIELDS}
for artifact in ('model', 'tokenizer'):
    key = artifact+'_artifact_binding_sha256'
    descriptor = {'schema': 'frontier.vllm-model-artifact-binding/v1', 'artifact_kind': artifact,
                  'pinned_runtime_image_digest': 'sha256:'+'1'*64,
                  'scientific_model_identity': 'SYNTHETIC_CPU_FIXTURE_MODEL',
                  'tokenizer_identity': 'SYNTHETIC_CPU_FIXTURE_TOKENIZER'}
    ENV[key] = digest(descriptor)
    ARTIFACT_PROVENANCE[key] = {'binding_kind': 'source/image-declaration',
        'descriptor': descriptor, 'direct_file_fingerprints': None,
        'original_source': {'sha256': '8'*64, 'path_or_member': 'SYNTHETIC_CPU_FIXTURE_SOURCE'}}


def fixtures():
    captures = []
    for run in range(3):
        phases = {}
        for mode in ('train', 'eval'):
            groups = {}
            for group, ids in expected_groups(mode).items():
                groups[group] = {variant: {ident: {
                    'latency_seconds': 20.0 if variant == 'vanilla' else 10.0,
                    'correct': True, 'evidence_member': 'synthetic-fixture/'+ident+'.json',
                    'terminal_sha256': 'b'*64} for ident in ids}
                    for variant in ('vanilla', 'reference')}
            phases[mode] = {'evidence_sha256': str(run+1)*64,
                'consumer_receipt_sha256': 'c'*64, 'groups': groups}
        captures.append({'schema': CALIBRATION_SCHEMA, 'run_id': 'run_SYNTHETIC_FIXTURE_'+str(run),
                         'environment': copy.deepcopy(ENV),
                         'environment_provenance': copy.deepcopy(ARTIFACT_PROVENANCE), 'phases': phases})
    return captures


def target(captures=None):
    return build_targets(fixtures() if captures is None else captures, environment=ENV)


def observations(mode, time=6.0, correct=True):
    return {group: [InstanceObservation(ident, time, correct) for ident in ids]
            for group, ids in expected_groups(mode).items()}


class FixedTargetsTests(unittest.TestCase):
    def test_immutable_release_binding_candidate_only_and_missing_baseline(self):
        from fixed_target_release import load_release, FixedTargetReleaseError
        from runtime_config import project_task_config
        from test_runtime_config import synthetic_task, SYNTHETIC_LIMITS
        from sidecar_catalog import render_profiles
        task = synthetic_task()
        task['images']['gpu_sidecars'] = render_profiles(SYNTHETIC_LIMITS, variants=('candidate',))
        config = project_task_config(task)
        self.assertEqual(len(config['sidecars']), 97)
        self.assertFalse(any(k.endswith(('-vanilla', '-reference')) for k in config['sidecars']))
        fixed = target()
        binding = {'schema': 'frontier.vllm-fixed-target-release/v1',
                   'variants': ['candidate'], 'manifest_sha256': digest(fixed), 'environment': ENV}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(FixedTargetReleaseError): load_release(root, config)
            (root/'fixed-target-release.json').write_bytes(canonical_bytes(binding))
            with self.assertRaises(FixedTargetReleaseError): load_release(root, config)
            (root/'fixed-targets.json').write_bytes(canonical_bytes(fixed))
            self.assertEqual(load_release(root, config), fixed)
            with self.assertRaises(FixedTargetReleaseError):
                load_release(root, project_task_config(synthetic_task()))
            (root/'fixed-targets.json').write_bytes(canonical_bytes(fixed)+b'\n')
            with self.assertRaises(FixedTargetReleaseError): load_release(root, config)

    def test_candidate_service_runtime_failure_keeps_affected_workload_zero(self):
        from cpu_reference_verdict import RuntimeGroupFailure
        rows = observations('eval')
        rows['swe/hidden'] = RuntimeGroupFailure('candidate', 'swe/hidden', expected_groups('eval')['swe/hidden'])
        result = evaluate_fixed(rows, target(), mode='eval')
        self.assertFalse(result['pass'])
        self.assertEqual(result['execution'], 'runtime_error')
        self.assertEqual(result['continuous']['swe']['score'], 0)
        self.assertEqual(result['continuous']['swe']['policy'], 'affected_workload_no_credit')

    def test_each_original_reference_capture_fails_by_strict_envelope_arithmetic(self):
        captures = fixtures()
        for k, capture in enumerate(captures):
            for group in capture['phases']['train']['groups'].values():
                for j, row in enumerate(group['reference'].values()):
                    row['latency_seconds'] += (j+k) % 3
        fixed = target(captures)
        for capture in captures:
            candidate = {g: [InstanceObservation(i, r['latency_seconds'], r['correct'])
                            for i, r in group['reference'].items()]
                         for g, group in capture['phases']['train']['groups'].items()}
            result = evaluate_fixed(candidate, fixed, mode='train')
            self.assertFalse(result['pass'])
            self.assertTrue(all(not g['comparisons']['paired_timing']
                                for g in result['reference_verdict']['groups'].values()))

    def test_candidate_only_parallel_join_rejects_baseline_observations(self):
        from parallel_phase import join_groups
        groups = observations('train')
        completed = {g: {'candidate': {g: rows}} for g, rows in groups.items()}
        self.assertTrue(join_groups('train', completed, fixed_targets=target())['pass'])
        completed['swe/public']['reference'] = {'swe/public': groups['swe/public']}
        with self.assertRaises(MeasurementError):
            join_groups('train', completed, fixed_targets=target())

    def test_fixed_binding_failure_is_before_candidate_input_and_gpu(self):
        import os
        import verify
        from fixed_target_release import FixedTargetReleaseError
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'scratch').mkdir()
            (root/'config.json').write_text('{}')
            paths = {'/opt/frontier/runtime-config.json': root/'config.json',
                     '/opt/frontier/runtime-plan.json': root/'config.json'}
            env = {'FRONTIER_RESULT_PATH': str(root/'result.json'),
                   'FRONTIER_EVIDENCE_PATH': str(root/'evidence'),
                   'FRONTIER_VERIFICATION_MODE': 'train',
                   'FRONTIER_MAX_EVIDENCE_BYTES': '10000'}
            with patch.dict(os.environ, env), \
                 patch.object(verify, 'Path', side_effect=lambda v: Path(paths.get(str(v), v))), \
                 patch.object(verify.tempfile, 'mkdtemp', return_value=str(root/'scratch')), \
                 patch.object(verify, 'validate_plan', return_value={}), \
                 patch.object(verify, 'load_release', side_effect=FixedTargetReleaseError()), \
                 patch.object(verify, 'submission_input') as payload, \
                 patch.object(verify, 'run_parallel_phase') as gpu, \
                 patch.object(verify, 'write_evidence'):
                self.assertEqual(verify.main(), 0)
            payload.assert_not_called()
            gpu.assert_not_called()
            result = json.loads((root/'result.json').read_text())
            self.assertEqual(result['failure']['code'], 'vllm_trusted_measurement_unavailable')
            self.assertNotIn('raw_score', result)

    def test_deterministic_capture_order_and_schema(self):
        captures = fixtures()
        first = target(captures)
        self.assertEqual(canonical_bytes(first), canonical_bytes(target(captures[::-1])))
        try:
            import jsonschema
        except ImportError:
            self.skipTest('jsonschema is a repository validation dependency')
        jsonschema.validate(first, json.loads(Path(__file__).with_name('fixed_targets.schema.json').read_text()))

    def test_minimum_per_case_and_tied_provenance(self):
        captures = fixtures()
        ids = expected_groups('eval')['swe/public']
        captures[1]['phases']['eval']['groups']['swe/public']['reference'][ids[0]]['latency_seconds'] = 3
        fixed = target(captures)['phases']['eval']['swe/public']['reference']
        self.assertEqual(fixed['latency_seconds'][ids[0]], 3)
        self.assertEqual(len(fixed['timing_sources'][ids[0]]), 1)
        self.assertEqual(len(fixed['timing_sources'][ids[1]]), 3)

    def test_bfcl_majority_same_count_no_extra_validity_gate(self):
        captures = fixtures()
        ids = expected_groups('train')['bfcl/public']
        for k, capture in enumerate(captures):
            for ident in ids:
                capture['phases']['train']['groups']['bfcl/public']['vanilla'][ident]['correct'] = ident == ids[k]
        fixed = target(captures)
        vanilla = fixed['phases']['train']['bfcl/public']['vanilla']
        self.assertEqual(vanilla['correct'], 0)
        self.assertEqual(sum(vanilla['correct_labels'].values()), 0)
        result = evaluate_fixed(observations('train', correct=False), fixed, mode='train')
        self.assertEqual(result['validity']['groups'], {})
        self.assertGreater(result['score'], 0)
        self.assertTrue(result['valid'])
        self.assertFalse(result['pass'])

    def test_vanilla_median_vector_majority_and_tied_sources_all_groups(self):
        captures = fixtures()
        for mode in ('train', 'eval'):
            for group, ids in expected_groups(mode).items():
                for k, capture in enumerate(captures):
                    for j, ident in enumerate(ids):
                        row = capture['phases'][mode]['groups'][group]['vanilla'][ident]
                        row['latency_seconds'] = ((1, 20, 100) if j == 0 else (30, 30, 2))[k]
                        row['correct'] = k < (2 if j == 0 else 1)
        fixed = target(captures)
        for mode in ('train', 'eval'):
            for group, ids in expected_groups(mode).items():
                vanilla = fixed['phases'][mode][group]['vanilla']
                self.assertEqual(vanilla['latency_seconds'],
                                 {i: 20 if j == 0 else 30 for j, i in enumerate(ids)})
                self.assertEqual(vanilla['correct'], 1)
                self.assertEqual(vanilla['correct_labels'], {i: j == 0 for j, i in enumerate(ids)})
                self.assertEqual(len(vanilla['timing_sources'][ids[0]]), 1)
                self.assertEqual(len(vanilla['timing_sources'][ids[1]]), 2)
                self.assertEqual(vanilla['tails_seconds'], {'p90': 30, 'p95': 30, 'p99': 30})
                self.assertEqual(vanilla['correctness_policy'], 'per_case_majority_of_three')
        self.assertEqual(canonical_bytes(fixed), canonical_bytes(target(captures[::-1])))

    def test_old_flip_threshold_never_called_or_enforced(self):
        for mode in ('train','eval'):
            captures=fixtures()
            c=observations(mode)
            for group,ids in expected_groups(mode).items():
                if not group.startswith('bfcl/'):
                    continue
                count=len(ids)-correctness_allowance(len(ids))-1
                for capture in captures:
                    for j,ident in enumerate(ids):
                        capture['phases'][mode]['groups'][group]['reference'][ident]['correct']=j<count
                c[group]=[InstanceObservation(i,6,j<count) for j,i in enumerate(ids)]
            with patch('cpu_reference_verdict._bfcl_regression_gate',side_effect=AssertionError('obsolete flip gate invoked')):
                result=evaluate_fixed(c,target(captures),mode=mode)
            self.assertTrue(result['pass'])
            self.assertTrue(result['valid'])
            self.assertGreater(result['score'],0)
            self.assertEqual(result['score'],result['continuous']['score'])

    def test_vanilla_tails_come_from_median_vector_not_capture_percentiles(self):
        captures = fixtures()
        ids = expected_groups('train')['swe/public']
        for k, capture in enumerate(captures):
            for j, ident in enumerate(ids):
                capture['phases']['train']['groups']['swe/public']['vanilla'][ident]['latency_seconds'] = (
                    100 if j == k else 1)
        vanilla = target(captures)['phases']['train']['swe/public']['vanilla']
        self.assertTrue(all(t['p99'] == 100 for t in vanilla['observed_group_tails_seconds'].values()))
        self.assertEqual(vanilla['tails_seconds'], {'p90': 1, 'p95': 1, 'p99': 1})

    def test_vanilla_majority_is_not_median_or_max_whole_group_count(self):
        captures = fixtures()
        ids = expected_groups('train')['swe/public']
        for k, capture in enumerate(captures):
            for ident in ids:
                capture['phases']['train']['groups']['swe/public']['vanilla'][ident]['correct'] = ident == ids[k]
        vanilla = target(captures)['phases']['train']['swe/public']['vanilla']
        self.assertEqual(list(vanilla['observed_correct_counts'].values()), [1, 1, 1])
        self.assertEqual(vanilla['correct'], 0)

    def test_many_flips_equal_correct_total_same_score(self):
        captures=fixtures()
        ids=expected_groups('eval')['bfcl/public']
        for capture in captures:
            for variant in ('vanilla','reference'):
                for j,ident in enumerate(ids):
                    capture['phases']['eval']['groups']['bfcl/public'][variant][ident]['correct']=j<70
        fixed=target(captures)
        results=[]
        for selected in (set(ids[:70]),set(ids[-70:])):
            c=observations('eval')
            c['bfcl/public']=[InstanceObservation(i,6,i in selected) for i in ids]
            results.append(evaluate_fixed(c,fixed,mode='eval'))
        self.assertTrue(all(r['pass'] for r in results))
        self.assertEqual(results[0]['score'],results[1]['score'])
        self.assertEqual(results[0]['continuous'],results[1]['continuous'])

    def test_no_flip_policy_preserves_runtime_group_no_credit(self):
        from cpu_reference_verdict import RuntimeGroupFailure
        from verify import result_projection
        for mode in ('train','eval'):
            for group,ids in expected_groups(mode).items():
                c=observations(mode)
                c[group]=RuntimeGroupFailure('candidate',group,tuple(ids))
                result=evaluate_fixed(c,target(),mode=mode)
                self.assertFalse(result['pass'])
                self.assertEqual(result['execution'],'runtime_error')
                workload=group.split('/')[0]
                self.assertEqual(result['continuous'][workload]['score'],0)
                self.assertGreater(result['score'],0)
                self.assertEqual(result_projection(result,mode)['raw_score'],result['score'])

    def test_reference_median_group_total_not_max_or_majority(self):
        captures = fixtures()
        ids = expected_groups('train')['swe/public']
        for k, capture in enumerate(captures):
            for ident in ids:
                capture['phases']['train']['groups']['swe/public']['reference'][ident]['correct'] = (
                    ident in ((ids[0],), (ids[1], ids[2]), tuple(ids))[k])
        fixed = target(captures)['phases']['train']['swe/public']['reference']
        self.assertEqual(fixed['correct'], 2)
        self.assertEqual(sum(fixed['correct_labels'].values()), 2)
        self.assertEqual(sorted(fixed['observed_correct_counts'].values()), [1, 2, 5])
        self.assertEqual(fixed['median_correct_count'], 2)
        self.assertEqual(fixed['required_correct_count'], 1)
        self.assertEqual(fixed['selected_count_source_runs'], [captures[1]['run_id']])
        self.assertNotIn('max_count_source_runs', fixed)

    def test_allowance_ceiling_discontinuities(self):
        for n, allowance in ((5,1),(8,1),(20,1),(21,2),(40,2),(41,3),(48,3),(155,8)):
            self.assertEqual(correctness_allowance(n), allowance)

    def test_exact_threshold_and_one_below_all_groups(self):
        fixed = target()
        for mode in ('train', 'eval'):
            for group, ids in expected_groups(mode).items():
                threshold = fixed['phases'][mode][group]['reference']['required_correct_count']
                for count in (threshold, threshold-1):
                    c = observations(mode)
                    c[group] = [InstanceObservation(i,6,j<count) for j,i in enumerate(ids)]
                    result = evaluate_fixed(c, fixed, mode=mode)
                    verdict = result['reference_verdict']['groups'][group]
                    self.assertEqual(verdict['comparisons']['correctness'],count==threshold)
                    self.assertEqual(verdict['expected_instances'],len(ids))
                    self.assertEqual(verdict['required_correct_count'],threshold)

    def test_negative_raw_threshold_and_missing_denominator(self):
        captures = fixtures()
        for capture in captures:
            for variants in capture['phases']['train']['groups'].values():
                for row in variants['reference'].values():
                    row['correct'] = False
        fixed = target(captures)
        self.assertEqual(fixed['phases']['train']['swe/public']['reference']['required_correct_count'],-1)
        c = observations('train',correct=False)
        c['swe/public'].pop()
        verdict = evaluate_fixed(c,fixed,mode='train')['reference_verdict']['groups']['swe/public']
        self.assertEqual(verdict['expected_instances'],5)
        self.assertTrue(verdict['comparisons']['correctness'])
        self.assertFalse(verdict['pass'])

    def test_reference_allowance_does_not_change_vanilla_continuous(self):
        captures = fixtures()
        for k, capture in enumerate(captures):
            for variants in capture['phases']['eval']['groups'].values():
                for j, row in enumerate(variants['reference'].values()):
                    row['correct'] = j < k
        c = observations('eval')
        original = evaluate_fixed(c,target(),mode='eval')
        revised = evaluate_fixed(c,target(captures),mode='eval')
        self.assertEqual(original['continuous'],revised['continuous'])
        self.assertEqual(original['score'],revised['score'])
        self.assertEqual(original['validity'],revised['validity'])

    def test_old_rmax_policy_rejected(self):
        fixed = target()
        fixed['policy'] = 'min-case-time-min-observed-group-tails-strict-max-reference-count/v2'
        with self.assertRaises(MeasurementError):
            validate_targets(fixed,expected_environment=ENV)

    def test_reused_calibration_samepatch_never_beats_fastest_envelope(self):
        captures = fixtures()
        for k,capture in enumerate(captures):
            for mode in ('train','eval'):
                for variants in capture['phases'][mode]['groups'].values():
                    for j,row in enumerate(variants['reference'].values()):
                        row['latency_seconds'] = 2 + (j+k)%3
        fixed = target(captures)
        for capture in captures:
            for mode in ('train','eval'):
                c = {group:[InstanceObservation(i,row['latency_seconds'],row['correct'])
                            for i,row in variants['reference'].items()]
                     for group,variants in capture['phases'][mode]['groups'].items()}
                result = evaluate_fixed(c,fixed,mode=mode)
                self.assertFalse(result['pass'])
                self.assertTrue(all(not g['comparisons']['paired_timing']
                    for g in result['reference_verdict']['groups'].values()))

    def test_original_continuous_arithmetic_parity_both_phases(self):
        fixed = target()
        for mode in ('train', 'eval'):
            c = observations(mode)
            legacy = (evaluate_train if mode == 'train' else evaluate_final)(
                observations(mode, 20), c, observations(mode, 10))
            new = evaluate_fixed(c, fixed, mode=mode)
            self.assertEqual(new['continuous'], legacy['continuous'])
            self.assertEqual(new['score'], legacy['score'])

    def test_unchanged_reference_fails_but_preserves_score(self):
        result = evaluate_fixed(observations('eval', 10), target(), mode='eval')
        self.assertFalse(result['pass'])
        self.assertGreater(result['score'], 0)
        self.assertTrue(all(not g['comparisons']['paired_timing'] for g in result['reference_verdict']['groups'].values()))

    def test_strict_zero_margin_and_correctness_equality(self):
        fixed = target()
        self.assertTrue(evaluate_fixed(observations('train', 9.999999), fixed, mode='train')['pass'])
        for time in (10, 10.000001):
            result = evaluate_fixed(observations('train', time), fixed, mode='train')
            self.assertFalse(result['pass'])
            self.assertTrue(all(g['comparisons']['correctness'] for g in result['reference_verdict']['groups'].values()))
            self.assertTrue(all(not g['comparisons']['paired_timing'] for g in result['reference_verdict']['groups'].values()))

    def test_tail_targets_are_min_observed_group_tail_not_envelope_tail(self):
        captures = fixtures()
        ids = expected_groups('train')['swe/public']
        for k, capture in enumerate(captures):
            for j, ident in enumerate(ids):
                capture['phases']['train']['groups']['swe/public']['reference'][ident]['latency_seconds'] = 100 if j == k else 1
        fixed = target(captures)['phases']['train']['swe/public']['reference']
        self.assertEqual(fixed['per_case_envelope_tails_diagnostic_seconds']['p99'], 1)
        self.assertEqual(fixed['tails_seconds']['p99'], 100)
        self.assertEqual(fixed['tail_sources']['p99'], sorted(c['run_id'] for c in captures))
        self.assertEqual(set(fixed['observed_group_tails_seconds']), set(c['run_id'] for c in captures))

    def test_missing_and_candidate_runtime_error_not_infra(self):
        c = observations('train')
        c['swe/public'].pop()
        c['bfcl/public'][0] = InstanceObservation(c['bfcl/public'][0].instance_id, None, False, 'candidate_error')
        result = evaluate_fixed(c, target(), mode='train')
        self.assertFalse(result['pass'])
        self.assertEqual(result['continuous']['bfcl']['groups']['public']['unusable'],
                         {c['bfcl/public'][0].instance_id: 'api_error'})

    def test_infra_observation_does_not_become_scientific_fail(self):
        c = observations('train')
        c['swe/public'][0] = InstanceObservation(c['swe/public'][0].instance_id, None, False, 'infra_error')
        with self.assertRaises(MeasurementError):
            evaluate_fixed(c, target(), mode='train')

    def test_phase_separation_no_final_prefix_as_train(self):
        captures = fixtures()
        first_id = expected_groups('train')['swe/public'][0]
        for capture in captures:
            capture['phases']['train']['groups']['swe/public']['vanilla'][first_id]['latency_seconds'] = 99
        fixed = target(captures)
        self.assertEqual(fixed['phases']['train']['swe/public']['vanilla']['latency_seconds'][first_id], 99)
        self.assertEqual(fixed['phases']['eval']['swe/public']['vanilla']['latency_seconds'][first_id], 20)
        self.assertEqual(len(fixed['phases']['train']['swe/public']['expected_ids']), 5)
        self.assertEqual(len(fixed['phases']['eval']['swe/public']['expected_ids']), 48)
        del captures[0]['phases']['train']
        with self.assertRaises(MeasurementError): target(captures)

    def test_invalid_calibrations_fail_closed(self):
        mutations = [
            lambda c: c.pop(),
            lambda c: c[0].update(run_id=EXCLUDED_RUN),
            lambda c: c[0].update(run_id=c[1]['run_id']),
            lambda c: c[0]['environment'].update(tokenizer_artifact_binding_sha256='d'*64),
            lambda c: c[0]['phases']['eval']['groups']['swe/public']['vanilla'].pop(next(iter(c[0]['phases']['eval']['groups']['swe/public']['vanilla']))),
        ]
        for mutate in mutations:
            c = fixtures()
            mutate(c)
            with self.subTest(mutate=mutate), self.assertRaises(MeasurementError): target(c)

    def test_invalid_timing_correctness_and_provenance(self):
        for field, value in [('latency_seconds', float('nan')), ('latency_seconds', True),
                             ('latency_seconds', 0), ('correct', None),
                             ('evidence_member', '../untrusted'), ('terminal_sha256', '')]:
            c = fixtures()
            row = next(iter(c[0]['phases']['train']['groups']['swe/public']['vanilla'].values()))
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(MeasurementError): target(c)

    def test_positive_margin_rejected_and_environment(self):
        for margin in [.04, .3, 1, True, float('inf')]:
            fixed = target()
            fixed['margin_fraction'] = margin
            with self.subTest(margin=margin), self.assertRaises(MeasurementError):
                validate_targets(fixed, expected_environment=ENV)
        wrong = dict(ENV, tokenizer_artifact_binding_sha256='f'*64)
        with self.assertRaises(MeasurementError): validate_targets(target(), expected_environment=wrong)

    def test_target_derived_fields_are_not_trusted(self):
        fixed = target()
        fixed['phases']['train']['bfcl/public']['vanilla']['correct'] = 0
        with self.assertRaises(MeasurementError): validate_targets(fixed, expected_environment=ENV)

    def test_loader_pin_duplicate_keys_and_noncanonical(self):
        fixed = target()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'targets.json'
            raw = canonical_bytes(fixed)
            path.write_bytes(raw)
            loaded = load_targets(path, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_environment=ENV)
            self.assertEqual(digest(loaded), hashlib.sha256(raw).hexdigest())
            with self.assertRaises(MeasurementError): load_targets(path, expected_sha256='0'*64, expected_environment=ENV)
            for raw in [b'{"schema":1,"schema":2}', b'{"schema":NaN}', canonical_bytes(fixed)+b'\n']:
                path.write_bytes(raw)
                with self.assertRaises(MeasurementError): load_targets(path, expected_sha256=hashlib.sha256(raw).hexdigest(), expected_environment=ENV)

    def test_offline_plan_receipt_binding(self):
        captures = fixtures()
        final_only = copy.deepcopy(captures[2])
        final_only['run_id'] = 'run_SYNTHETIC_FINAL_ONLY'
        final_only['phases'].pop('train')
        captures[2]['phases'].pop('eval')
        captures.append(final_only)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = []
            for k, capture in enumerate(captures):
                item = {'path': f'capture-{k}.json', 'receipt_paths': {}}
                for mode in capture['phases']:
                    phase = capture['phases'][mode]
                    receipt = {'schema': 'frontier.vllm-fixed-calibration-consumer-receipt/v1',
                        'status': 'PASS', 'run_id': capture['run_id'], 'mode': mode,
                        'evidence_sha256': phase['evidence_sha256'],
                        'normalized_groups_sha256': digest(phase['groups']),
                        'environment_sha256': digest(capture['environment']),
                        'environment_provenance_sha256': digest(capture['environment_provenance'])}
                    raw = canonical_bytes(receipt)
                    name = f'receipt-{k}-{mode}.json'
                    (root/name).write_bytes(raw)
                    phase['consumer_receipt_sha256'] = hashlib.sha256(raw).hexdigest()
                    item['receipt_paths'][mode] = name
                raw = canonical_bytes(capture)
                (root/item['path']).write_bytes(raw)
                item['sha256'] = hashlib.sha256(raw).hexdigest()
                items.append(item)
            plan = {'schema': 'frontier.vllm-fixed-target-build-plan/v2', 'captures': items,
                    'environment': ENV}
            self.assertEqual(len(materialize(plan, root)['calibrations']), 4)
            invalid = copy.deepcopy(plan)
            invalid['captures'][2]['receipt_paths']['eval'] = items[0]['receipt_paths']['eval']
            with self.assertRaises(MeasurementError): materialize(invalid, root)
            (root/items[0]['receipt_paths']['train']).write_text('{}')
            with self.assertRaises(MeasurementError): materialize(plan, root)

    def test_phase_specific_three_retains_four_actual_run_identities(self):
        captures = fixtures()
        final_only = copy.deepcopy(captures[2])
        final_only['run_id'] = 'run_SYNTHETIC_FINAL_ONLY'
        final_only['phases'].pop('train')
        captures[2]['phases'].pop('eval')
        captures.append(final_only)
        fixed = target(captures)
        self.assertEqual(len(fixed['calibrations']), 4)
        for mode in ('train', 'eval'):
            expected_runs = {c['run_id'] for c in captures if mode in c['phases']}
            self.assertEqual(len(expected_runs), 3)
            observed = fixed['phases'][mode]['swe/public']['reference']['observed_correct_counts']
            self.assertEqual(set(observed), expected_runs)
        validate_targets(fixed, expected_environment=ENV)
        import jsonschema
        jsonschema.validate(fixed, json.loads(
            Path(__file__).with_name('fixed_targets.schema.json').read_text()))
        with self.assertRaises(MeasurementError): target(captures[:-1])
        with self.assertRaises(MeasurementError): target(captures + [copy.deepcopy(captures[2])])
        wrong = copy.deepcopy(captures)
        wrong[2]['phases']['eval'] = copy.deepcopy(final_only['phases']['eval'])
        with self.assertRaises(MeasurementError): target(wrong)


if __name__ == '__main__':
    unittest.main()
