"""Historical 20261007 legacy/calibration contract, not the active fixed-target policy.

Active v4 fixed-target contract/fault fixtures are in test_fixed_targets.py and
the repository tests/test_vllm_native_contract.py. This transferred archive binds
its original converted_problems/ source and coverage receipt; it is not current
serving calibration or qualification.

Run from this directory with PYTHONDONTWRITEBYTECODE=1 python3 -m unittest
test_beat_spec_contract -v. No pipeline, dataset, network or GPU is used.
"""
import ast
import copy
import hashlib
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

from instance_manifest import expected_groups
from reference_verdict import (
    InstanceObservation, MeasurementError, evaluate_final, evaluate_train,
)


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / 'converted_problems/vllm_llm_serving_optimization'
SPEC = json.loads((TASK / 'judge/beat_spec.json').read_text())
COVERAGE = json.loads((ROOT / 'docs/transfers/vllm_llm_serving_optimization'
                       '/METRIC_COVERAGE_20261007.json').read_text())
REFERENCE_METRICS = {'paired_timing', 'p90', 'p95', 'p99', 'correctness'}


def observations(latency, mode='eval'):
    return {group: [InstanceObservation(ident, latency, True) for ident in ids]
            for group, ids in expected_groups(mode).items()}


def bound_value(result, binding, group):
    value = result
    for key in binding.format(group=group).split('.'):
        value = value[key]
    return value


def bound_passes(result, mode):
    """Resolve documented bindings; no duplicate implementation of predicates."""
    return {
        name: [bound_value(result, metric['pass_binding'], group)
               for group in COVERAGE['phases'][mode]['groups']
               if metric['scope'] == 'all_groups' or group.startswith('bfcl/')]
        for name, metric in COVERAGE['metrics'].items()
    }


class BeatSpecContractTests(unittest.TestCase):
    def test_spec_shape_source_bindings_and_unchanged_source_pins(self):
        metrics = {row['name']: row for row in SPEC['metrics']}
        self.assertEqual(len(metrics), len(SPEC['metrics']))
        self.assertEqual(set(metrics), REFERENCE_METRICS | {'bfcl_regressions'})
        self.assertEqual(set(metrics), set(COVERAGE['metrics']))
        self.assertEqual(SPEC['beats_reference_aggregation'], {'kind': 'conjunction'})
        self.assertEqual(SPEC['solver_feedback'], [])
        self.assertEqual(COVERAGE['provenance']['kind'], 'legacy_task_without_paper_source')
        self.assertIsNone(COVERAGE['provenance']['paper'])
        for name, row in metrics.items():
            with self.subTest(metric=name):
                self.assertEqual(set(row), {'name', 'means', 'source', 'better',
                                            'clears_when', 'tolerance'})
                self.assertTrue(all(isinstance(v, str) and v.strip() for v in row.values()))
                self.assertEqual(row['better'], COVERAGE['metrics'][name]['better'])
                self.assertIn(row['better'], ('higher', 'lower'))
                self.assertTrue(row['tolerance'].startswith('0'))
                self.assertEqual(COVERAGE['metrics'][name]['additional_tolerance'], 0)
                for symbol in COVERAGE['metrics'][name]['source_symbols']:
                    self.assertIn(symbol, row['source'])
        for path, pin in COVERAGE['source_files'].items():
            with self.subTest(source=path):
                source = (ROOT / path).read_bytes()
                self.assertEqual(hashlib.sha256(source).hexdigest(), pin['sha256'])
                names = {node.name for node in ast.walk(ast.parse(source))
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
                self.assertTrue(set(pin['symbols']).issubset(names))

    def test_mapping_resolves_actual_results_with_exact_phase_denominators(self):
        for mode, evaluate in (('train', evaluate_train), ('eval', evaluate_final)):
            with self.subTest(mode=mode):
                phase = COVERAGE['phases'][mode]
                self.assertEqual(phase['groups'],
                                 {g: len(ids) for g, ids in expected_groups(mode).items()})
                result = evaluate(observations(4, mode), observations(1, mode),
                                  observations(2, mode))
                passes = bound_passes(result, mode)
                self.assertEqual({m: len(rows) for m, rows in passes.items()},
                                 COVERAGE['progress_transport']['future_required_counts'][mode])
                self.assertEqual(sum(len(passes[m]) for m in REFERENCE_METRICS),
                                 phase['reference_comparisons'])
                self.assertEqual(len(passes['bfcl_regressions']), phase['bfcl_gate_comparisons'])
                self.assertTrue(all(all(rows) for rows in passes.values()))
                self.assertTrue(result['pass'])
                for group in phase['groups']:
                    self.assertEqual(set(result['reference_verdict']['groups'][group]['comparisons']),
                                     REFERENCE_METRICS)
                    for metric in COVERAGE['metrics'].values():
                        if metric['scope'] == 'bfcl_groups' and not group.startswith('bfcl/'):
                            continue
                        for binding in metric['value_bindings']:
                            self.assertIsNotNone(bound_value(result, binding, group))

    def test_timing_equality_fails_and_correctness_equality_passes_every_group(self):
        vanilla, reference = observations(4), observations(2)
        for group in expected_groups('eval'):
            with self.subTest(group=group):
                candidate = observations(1)
                candidate[group] = copy.deepcopy(reference[group])
                result = evaluate_final(vanilla, candidate, reference)
                passes = bound_passes(result, 'eval')
                for metric in REFERENCE_METRICS - {'correctness'}:
                    self.assertEqual(passes[metric].count(False), 1)
                self.assertTrue(all(passes['correctness']))
                self.assertFalse(result['pass'])

    def test_paired_predicate_uses_log_ratio_not_rounded_display_ratio(self):
        candidate = observations(1)
        for group in candidate:
            candidate[group][0] = replace(candidate[group][0],
                                          latency_seconds=math.nextafter(1.0, 0.0))
        result = evaluate_final(observations(4), candidate, observations(1))
        for group in expected_groups('eval'):
            verdict = result['reference_verdict']['groups'][group]
            self.assertEqual(verdict['paired_speedup_over_reference'], 1.0)
            self.assertTrue(bound_value(result,
                COVERAGE['metrics']['paired_timing']['pass_binding'], group))
        self.assertFalse(result['pass'])  # unchanged strict tail comparisons

    def test_each_tail_independently_blocks_each_group_despite_better_geomean(self):
        for group, ids in expected_groups('eval').items():
            for percentile in (90, 95, 99):
                with self.subTest(group=group, percentile=percentile):
                    vanilla, reference, candidate = observations(400), observations(2), observations(1)
                    reference[group] = [InstanceObservation(i, 10 + j, True)
                                        for j, i in enumerate(ids)]
                    index = (percentile * len(ids) + 99) // 100 - 1
                    candidate[group] = [InstanceObservation(i, 0.1 if j < index else 10 + index, True)
                                        for j, i in enumerate(ids)]
                    result = evaluate_final(vanilla, candidate, reference)
                    verdict = result['reference_verdict']['groups'][group]
                    self.assertEqual([m for m, ok in verdict['comparisons'].items() if not ok],
                                     ['p' + str(percentile)])
                    self.assertFalse(result['pass'])
                    self.assertEqual(result['reference_verdict']['passed_groups'], 3)

    def test_correctness_loss_blocks_each_group_without_cross_group_compensation(self):
        for group in expected_groups('eval'):
            with self.subTest(group=group):
                vanilla, reference, candidate = observations(4), observations(2), observations(1)
                candidate[group][0] = replace(candidate[group][0], correct=False)
                result = evaluate_final(vanilla, candidate, reference)
                verdict = result['reference_verdict']['groups'][group]
                self.assertEqual([m for m, ok in verdict['comparisons'].items() if not ok],
                                 ['correctness'])
                self.assertTrue(result['valid'])
                self.assertFalse(result['pass'])

    def test_bfcl_gate_boundary_is_independent_of_all_reference_comparisons(self):
        for mode, evaluate, allowed in (('train', evaluate_train, 1), ('eval', evaluate_final, 8)):
            for group in expected_groups(mode):
                if not group.startswith('bfcl/'):
                    continue
                for flips in (allowed, allowed + 1):
                    with self.subTest(mode=mode, group=group, flips=flips):
                        vanilla, reference, candidate = [observations(t, mode) for t in (4, 2, 1)]
                        for index in range(flips):
                            candidate[group][index] = replace(candidate[group][index], correct=False)
                            # Equally many new correct answers cannot offset vanilla flips.
                            gain = index + flips
                            vanilla[group][gain] = replace(vanilla[group][gain], correct=False)
                            reference[group][gain] = replace(reference[group][gain], correct=False)
                        result = evaluate(vanilla, candidate, reference)
                        gate = result['validity']['groups'][group]
                        self.assertEqual(gate['allowed_regressions'], allowed)
                        self.assertEqual(gate['regressions'], flips)
                        self.assertTrue(result['reference_verdict']['pass'])
                        self.assertEqual(bound_value(result,
                            COVERAGE['metrics']['bfcl_regressions']['pass_binding'], group), flips <= allowed)
                        self.assertEqual(result['pass'], flips <= allowed)
                        self.assertGreater(result['continuous']['score'], 0)
                        if flips > allowed:
                            self.assertEqual(result['score'], 0)

    def test_missing_data_preserves_comparison_denominators_and_trusted_errors(self):
        for group, ids in expected_groups('eval').items():
            for missing_group in (False, True):
                with self.subTest(group=group, missing_group=missing_group):
                    vanilla, reference, candidate = observations(4), observations(2), observations(1)
                    if missing_group:
                        del candidate[group]
                    else:
                        candidate[group].pop()
                    result = evaluate_final(vanilla, candidate, reference)
                    verdict = result['reference_verdict']['groups'][group]
                    self.assertEqual(verdict['expected_instances'], len(ids))
                    self.assertFalse(verdict['complete'])
                    self.assertFalse(result['pass'])
                    self.assertIsNone(verdict['candidate_tails'])
                    self.assertEqual({m: len(rows) for m, rows in bound_passes(result, 'eval').items()},
                                     COVERAGE['progress_transport']['future_required_counts']['eval'])
                    reference[group].pop()
                    with self.assertRaises(MeasurementError):
                        evaluate_final(vanilla, candidate, reference)


if __name__ == '__main__':
    unittest.main()
