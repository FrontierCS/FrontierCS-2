import copy
import unittest
from dataclasses import replace

from instance_manifest import expected_groups
from reference_verdict import InstanceObservation, MeasurementError, bfcl_regression_gate, evaluate_final, evaluate_train


def observations(latency, mode='eval'):
    return {group: [InstanceObservation(ident, latency, True) for ident in ids] for group, ids in expected_groups(mode).items()}


class ReferenceVerdictTests(unittest.TestCase):
    def setUp(self):
        self.vanilla = observations(4)
        self.reference = observations(2)
        self.candidate = observations(1)

    def test_original_manifest_and_all_four_groups_pass(self):
        train = expected_groups('train')
        final = expected_groups('eval')
        self.assertEqual({k: len(v) for k, v in train.items()}, {'swe/public': 5, 'bfcl/public': 8})
        self.assertEqual({k: len(v) for k, v in final.items()}, {'swe/public': 48, 'swe/hidden': 48, 'bfcl/public': 155, 'bfcl/hidden': 155})
        for workload in ('swe', 'bfcl'):
            self.assertEqual(final[workload + '/public'], final[workload + '/hidden'])
            self.assertEqual(final[workload + '/public'][:len(train[workload + '/public'])], train[workload + '/public'])
        result = evaluate_final(self.vanilla, self.candidate, self.reference)
        self.assertTrue(result['reference_verdict']['pass'])
        self.assertEqual(result['reference_verdict']['passed_groups'], 4)
        self.assertEqual(result['continuous']['swe']['instance_evaluations'], 96)
        self.assertEqual(result['continuous']['bfcl']['instance_evaluations'], 310)

    def test_train_uses_only_original_public_counts_and_existing_weights(self):
        vanilla, reference, candidate = [observations(t, 'train') for t in (4, 2, 1)]
        result = evaluate_train(vanilla, candidate, reference)
        self.assertTrue(result['pass'])
        self.assertEqual(result['reference_verdict']['expected_groups'], 2)
        self.assertEqual(set(result['reference_verdict']['groups']), {'swe/public', 'bfcl/public'})
        self.assertEqual(result['continuous']['swe']['instance_evaluations'], 5)
        self.assertEqual(result['continuous']['bfcl']['instance_evaluations'], 8)
        self.assertAlmostEqual(result['score'], 0.8 * result['continuous']['swe']['score']
                               + 0.2 * result['continuous']['bfcl']['score'])
        candidate['swe/hidden'] = candidate['swe/public']
        with self.assertRaises(MeasurementError):
            evaluate_train(vanilla, candidate, reference)

    def test_train_reference_ties_do_not_change_independent_continuous(self):
        vanilla, reference, candidate = [observations(t, 'train') for t in (4, 2, 1)]
        score = evaluate_train(vanilla, candidate, reference)['score']
        reference['swe/public'] = copy.deepcopy(candidate['swe/public'])
        result = evaluate_train(vanilla, candidate, reference)
        self.assertEqual(result['score'], score)
        self.assertFalse(result['reference_verdict']['groups']['swe/public']['pass'])
        self.assertTrue(result['reference_verdict']['groups']['bfcl/public']['pass'])

    def test_train_original_bfcl_regression_gate_still_invalidates_all_scoring(self):
        vanilla, reference, candidate = [observations(t, 'train') for t in (4, 2, 1)]
        for index in range(2):
            candidate['bfcl/public'][index] = replace(candidate['bfcl/public'][index], correct=False)
        result = evaluate_train(vanilla, candidate, reference)
        self.assertFalse(result['valid'])
        self.assertFalse(result['pass'])
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['validity']['groups']['bfcl/public']['allowed_regressions'], 1)
        self.assertGreater(result['continuous']['score'], 0)

    def test_one_timing_tie_fails_without_other_group_compensation(self):
        self.candidate['swe/hidden'] = copy.deepcopy(self.reference['swe/hidden'])
        result = evaluate_final(self.vanilla, self.candidate, self.reference)['reference_verdict']
        self.assertFalse(result['pass'])
        self.assertEqual(result['passed_groups'], 3)
        group = result['groups']['swe/hidden']
        self.assertFalse(group['comparisons']['paired_timing'])
        self.assertFalse(group['comparisons']['p99'])
        self.assertTrue(group['comparisons']['correctness'])

    def test_one_tail_can_fail_despite_better_geomean(self):
        self.candidate['swe/hidden'][-1] = replace(self.candidate['swe/hidden'][-1], latency_seconds=3)
        group = evaluate_final(self.vanilla, self.candidate, self.reference)['reference_verdict']['groups']['swe/hidden']
        self.assertTrue(group['comparisons']['paired_timing'])
        self.assertTrue(group['comparisons']['p90'])
        self.assertFalse(group['comparisons']['p99'])
        self.assertFalse(group['pass'])

    def test_public_correctness_gain_does_not_cancel_hidden_regression(self):
        self.reference['bfcl/public'][0] = replace(self.reference['bfcl/public'][0], correct=False)
        self.candidate['bfcl/hidden'][0] = replace(self.candidate['bfcl/hidden'][0], correct=False)
        result = evaluate_final(self.vanilla, self.candidate, self.reference)
        self.assertFalse(result['reference_verdict']['pass'])
        self.assertTrue(result['reference_verdict']['groups']['bfcl/public']['comparisons']['correctness'])
        self.assertFalse(result['reference_verdict']['groups']['bfcl/hidden']['comparisons']['correctness'])
        self.assertTrue(result['reference_verdict']['groups']['bfcl/hidden']['complete'])

    def test_missing_instance_keeps_denominator_and_fails_group(self):
        missing = self.candidate['swe/hidden'].pop().instance_id
        result = evaluate_final(self.vanilla, self.candidate, self.reference)
        group = result['reference_verdict']['groups']['swe/hidden']
        self.assertFalse(group['pass'])
        self.assertEqual(group['expected_instances'], 48)
        self.assertEqual(group['observed_instances'], 47)
        self.assertIn(missing, group['unusable_instances'])
        self.assertEqual(result['continuous']['swe']['groups']['hidden']['instances'], 48)
        self.assertIsNone(group['candidate_tails'])

    def test_missing_candidate_group_does_not_renormalize_workloads(self):
        del self.candidate['bfcl/hidden']
        result = evaluate_final(self.vanilla, self.candidate, self.reference)
        self.assertFalse(result['reference_verdict']['pass'])
        self.assertEqual(result['continuous']['bfcl']['groups']['hidden']['instances'], 155)
        self.assertAlmostEqual(result['continuous']['score'], 0.8 * result['continuous']['swe']['score'] + 0.2 * result['continuous']['bfcl']['score'])
        self.assertFalse(result['valid'])
        self.assertEqual(result['score'], 0)

    def test_candidate_error_and_valid_step_limit_stay_distinct(self):
        self.candidate['swe/public'][0] = replace(self.candidate['swe/public'][0], execution='candidate_error', latency_seconds=None, correct=False, source_status='api_error')
        self.candidate['bfcl/public'][0] = replace(self.candidate['bfcl/public'][0], source_status='force_quit')
        result = evaluate_final(self.vanilla, self.candidate, self.reference)['reference_verdict']
        self.assertFalse(result['groups']['swe/public']['complete'])
        self.assertTrue(result['groups']['bfcl/public']['pass'])

    def test_failed_execution_cannot_claim_final_correctness_in_any_group(self):
        for group in expected_groups('eval'):
            with self.subTest(group=group):
                candidate = copy.deepcopy(self.candidate)
                candidate[group][0] = replace(candidate[group][0], execution='candidate_error', correct=True)
                with self.assertRaisesRegex(MeasurementError, 'cannot claim a successful correctness'):
                    evaluate_final(self.vanilla, candidate, self.reference)

    def test_failed_execution_cannot_bypass_bfcl_regression_gate(self):
        for mode, rate in (('train', 'public'), ('eval', 'public'), ('eval', 'hidden')):
            ids = expected_groups(mode)['bfcl/' + rate]
            vanilla = [InstanceObservation(i, 2, True) for i in ids]
            candidate = [InstanceObservation(i, 1, True) for i in ids]
            with self.subTest(mode=mode, rate=rate):
                candidate[0] = replace(candidate[0], execution='candidate_error', correct=True)
                with self.assertRaisesRegex(MeasurementError, 'cannot claim a successful correctness'):
                    bfcl_regression_gate(vanilla, candidate, mode=mode, rate=rate)
                for correctness in (False, None):
                    candidate[0] = replace(candidate[0], correct=correctness)
                    result = bfcl_regression_gate(vanilla, candidate, mode=mode, rate=rate)
                    self.assertEqual(result['regressions'], 1)

    def test_bfcl_original_threshold_allows_eight_final_regressions(self):
        for i in range(8):
            self.candidate['bfcl/public'][i] = replace(self.candidate['bfcl/public'][i], correct=False)
        result = evaluate_final(self.vanilla, self.candidate, self.reference)
        self.assertTrue(result['valid'])
        self.assertEqual(result['validity']['groups']['bfcl/public']['allowed_regressions'], 8)
        self.assertFalse(result['reference_verdict']['pass'])
        self.assertGreater(result['score'], 0)

    def test_bfcl_new_correct_answers_cannot_cancel_nine_regressions(self):
        for group in ('bfcl/public', 'bfcl/hidden'):
            with self.subTest(group=group):
                vanilla = copy.deepcopy(self.vanilla)
                reference = copy.deepcopy(self.reference)
                candidate = copy.deepcopy(self.candidate)
                for i in range(9):
                    candidate[group][i] = replace(candidate[group][i], correct=False)
                    vanilla[group][i + 9] = replace(vanilla[group][i + 9], correct=False)
                    reference[group][i + 9] = replace(reference[group][i + 9], correct=False)
                result = evaluate_final(vanilla, candidate, reference)
                self.assertTrue(result['reference_verdict']['pass'])
                self.assertGreater(result['continuous']['score'], 0)
                self.assertFalse(result['valid'])
                self.assertFalse(result['pass'])
                self.assertEqual(result['score'], 0)
                self.assertEqual(result['validity']['gate'], 'bfcl_correctness')
                self.assertEqual(result['validity']['groups'][group]['regressions'], 9)

    def test_bfcl_public_train_count_floor_and_missing_answer(self):
        ids = expected_groups('train')['bfcl/public']
        vanilla = [InstanceObservation(i, 2, True) for i in ids]
        candidate = [InstanceObservation(i, 1, True) for i in ids]
        candidate.pop()
        gate = bfcl_regression_gate(vanilla, candidate, mode='train', rate='public')
        self.assertTrue(gate['pass'])
        self.assertEqual(gate['allowed_regressions'], 1)
        candidate.pop()
        self.assertFalse(bfcl_regression_gate(vanilla, candidate, mode='train', rate='public')['pass'])
        with self.assertRaises(MeasurementError):
            bfcl_regression_gate(vanilla, candidate, mode='train', rate='hidden')

    def test_platform_failure_is_an_execution_error(self):
        self.candidate['swe/public'][0] = replace(self.candidate['swe/public'][0], execution='platform_error', source_status='provider.auth_error')
        with self.assertRaises(MeasurementError):
            evaluate_final(self.vanilla, self.candidate, self.reference)

    def test_duplicate_unexpected_or_missing_reference_data_is_rejected(self):
        variants = []
        duplicate = copy.deepcopy(self.candidate)
        duplicate['swe/public'].append(duplicate['swe/public'][0])
        variants.append((self.vanilla, duplicate, self.reference))
        wrong = copy.deepcopy(self.candidate)
        wrong['swe/public'][0] = replace(wrong['swe/public'][0], instance_id='not-an-original-case')
        variants.append((self.vanilla, wrong, self.reference))
        missing_ref = copy.deepcopy(self.reference)
        missing_ref['bfcl/hidden'].pop()
        variants.append((self.vanilla, self.candidate, missing_ref))
        missing_vanilla = copy.deepcopy(self.vanilla)
        del missing_vanilla['swe/hidden']
        variants.append((missing_vanilla, self.candidate, self.reference))
        for args in variants:
            with self.subTest(), self.assertRaises(MeasurementError):
                evaluate_final(*args)


if __name__ == '__main__':
    unittest.main()
