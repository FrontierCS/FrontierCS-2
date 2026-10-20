"""Bounded synthetic tests of operator projection, not serving qualification."""
import copy
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'verifier'))
from verify import beat_spec_progress, result_projection
from instance_manifest import expected_groups
from reference_verdict import InstanceObservation, evaluate_final, evaluate_train


def observations(latency, mode='eval'):
    return {group: [InstanceObservation(i, latency, True) for i in ids]
            for group, ids in expected_groups(mode).items()}


def operator(mode='eval'):
    evaluate = evaluate_train if mode == 'train' else evaluate_final
    return evaluate(observations(4, mode), observations(1, mode), observations(2, mode))


class BeatSpecProjectionTests(unittest.TestCase):
    def assert_counts(self, row, passed, failed):
        total = passed + failed
        self.assertEqual(row, {'passed': passed, 'failed': failed, 'total': total,
                              'pass_percent': round(100 * passed / total, 2),
                              'fail_percent': round(100 * failed / total, 2)})

    def test_final_counts_and_train_disclosure_boundary(self):
        for mode, count, bfcl_count in (('train', 2, 1), ('eval', 4, 2)):
            with self.subTest(mode=mode):
                internal = operator(mode)
                progress = beat_spec_progress(internal, mode)
                self.assertEqual(set(progress), {'paired_timing', 'p90', 'p95',
                                                 'p99', 'correctness', 'bfcl_regressions'})
                for metric, row in progress.items():
                    self.assert_counts(row, bfcl_count if metric == 'bfcl_regressions' else count, 0)
                projected = result_projection(internal, mode)
                if mode == 'eval':
                    self.assertEqual(projected['metrics'], {})
                else:
                    self.assertEqual(set(projected['metrics']['beat_spec_progress']),
                                     {'paired_timing', 'p90', 'p95', 'p99', 'correctness'})
                self.assertEqual(projected['raw_score'], internal['score'])
                if mode == 'train':
                    self.assertNotIn('private_metrics', projected)
                else:
                    self.assertEqual(projected['private_metrics'], {
                        'beats_reference': 1, 'beat_spec_progress': progress,
                        'aggregation': {'kind': 'conjunction'}})

    def test_each_reference_predicate_failure_counts_only_its_group(self):
        # Projection of explicit operator booleans: this does not retest timing arithmetic.
        for group in expected_groups('eval'):
            for metric in ('paired_timing', 'p90', 'p95', 'p99', 'correctness'):
                with self.subTest(group=group, metric=metric):
                    internal = operator()
                    internal['reference_verdict']['groups'][group]['comparisons'][metric] = False
                    internal['reference_verdict']['groups'][group]['pass'] = False
                    internal['reference_verdict']['pass'] = False
                    internal['reference_verdict']['passed_groups'] = 3
                    internal['pass'] = False
                    private = result_projection(internal, 'eval')['private_metrics']
                    self.assertEqual(private['beats_reference'], 0)
                    for name, row in private['beat_spec_progress'].items():
                        self.assert_counts(row, 2 if name == 'bfcl_regressions' else (3 if name == metric else 4),
                                           1 if name == metric else 0)

    def test_existing_bfcl_gate_failure_is_counted_without_swe_padding(self):
        for group in ('bfcl/public', 'bfcl/hidden'):
            with self.subTest(group=group):
                vanilla, candidate, reference = [observations(t) for t in (4, 1, 2)]
                for i in range(9):
                    candidate[group][i] = replace(candidate[group][i], correct=False)
                    reference[group][i + 9] = replace(reference[group][i + 9], correct=False)
                internal = evaluate_final(vanilla, candidate, reference)
                private = result_projection(internal, 'eval')['private_metrics']
                self.assert_counts(private['beat_spec_progress']['bfcl_regressions'], 1, 1)
                for name in ('paired_timing', 'p90', 'p95', 'p99', 'correctness'):
                    self.assert_counts(private['beat_spec_progress'][name], 4, 0)
                self.assertEqual(private['beats_reference'], 0)
                self.assertEqual(internal['score'], 0)

    def test_rounding_and_correctness_equality_do_not_recompute_predicates(self):
        candidate = observations(1)
        for group in candidate:
            candidate[group][0] = replace(candidate[group][0], latency_seconds=math.nextafter(1.0, 0.0))
        internal = evaluate_final(observations(4), candidate, observations(1))
        self.assertTrue(all(row['paired_speedup_over_reference'] == 1.0
                            for row in internal['reference_verdict']['groups'].values()))
        progress = result_projection(internal, 'eval')['private_metrics']['beat_spec_progress']
        self.assert_counts(progress['paired_timing'], 4, 0)
        self.assert_counts(progress['correctness'], 4, 0)
        for name in ('p90', 'p95', 'p99'):
            self.assert_counts(progress[name], 0, 4)

    def test_evaluated_candidate_omission_is_distinct_from_missing_result_evidence(self):
        candidate = observations(1)
        del candidate['bfcl/hidden']
        internal = evaluate_final(observations(4), candidate, observations(2))
        private = result_projection(internal, 'eval')['private_metrics']
        for name in ('paired_timing', 'p90', 'p95', 'p99', 'correctness'):
            self.assert_counts(private['beat_spec_progress'][name], 3, 1)
        self.assert_counts(private['beat_spec_progress']['bfcl_regressions'], 1, 1)
        self.assertEqual(private['beats_reference'], 0)
        variants = [{'score': 0.0, 'pass': False}]
        for section in ('reference_verdict', 'validity'):
            missing = copy.deepcopy(internal)
            del missing[section]['groups']['bfcl/hidden']
            variants.append(missing)
        missing_boolean = copy.deepcopy(internal)
        missing_boolean['reference_verdict']['groups']['swe/public']['comparisons']['p95'] = None
        variants.append(missing_boolean)
        for missing in variants:
            self.assertIsNone(beat_spec_progress(missing, 'eval'))
            self.assertEqual(result_projection(missing, 'eval')['private_metrics'], {'beats_reference': 0})
        # Null historical scientific scores cannot become a projected scientific zero.
        with self.assertRaises(ValueError):
            result_projection({'score': None, 'pass': False}, 'eval')

    def test_aggregate_and_completeness_must_agree_with_existing_booleans(self):
        complete = operator()
        for section, key in ((None, 'pass'), (None, 'valid'),
                             ('reference_verdict', 'pass'), ('validity', 'pass')):
            internal = copy.deepcopy(complete)
            (internal if section is None else internal[section])[key] = False
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                result_projection(internal, 'eval')
        incomplete = copy.deepcopy(complete)
        incomplete['reference_verdict']['groups']['swe/public']['complete'] = False
        with self.assertRaises(ValueError):
            result_projection(incomplete, 'eval')
        with self.assertRaises(ValueError):
            result_projection({'score': 12.0, 'pass': True}, 'eval')

    def test_projection_contains_counts_only_and_never_copies_private_detail(self):
        internal = operator()
        internal['operator_note'] = 'synthetic-private-detail'
        internal['reference_verdict']['groups']['swe/hidden']['unusable_instances'] = ['synthetic-case-id']
        for mode in ('train', 'eval'):
            phase_result = operator(mode)
            phase_result['operator_note'] = internal['operator_note']
            projected = result_projection(phase_result, mode)
            encoded = json.dumps(projected, allow_nan=False)
            for withheld in ('synthetic-private-detail', 'synthetic-case-id', 'reference_tails',
                             'candidate_tails', 'paired_speedup_over_reference', 'swe/hidden'):
                self.assertNotIn(withheld, encoded)
            if mode == 'eval':
                self.assertEqual(projected['metrics'], {})
            if mode == 'train':
                self.assertNotIn('private_metrics', projected)
        spec = json.loads((Path(__file__).parent / 'beat_spec.json').read_text())
        self.assertNotIn('beat_spec_progress', {row['name'] for row in spec['solver_feedback']})
        self.assertIn('beat_spec_progress', spec['aggregate_progress'])


if __name__ == '__main__':
    unittest.main()
