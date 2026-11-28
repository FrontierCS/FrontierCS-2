"""Private fixed-target arithmetic; trust comes from the release-pinned input hash.

The offline builder consumes normalized outputs of the accepted evidence consumer,
not arbitrary candidate data. SHA checks bind provenance; they do not authenticate
scheduler signatures. That upstream verification remains a separate release gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from continuous import RateMeasurement, continuous_score
from instance_manifest import expected_groups
from latency import tail_latencies
from reference_verdict import MeasurementError, _normalize, _timing

SCHEMA = 'frontier.vllm-fixed-targets/v1'
CALIBRATION_SCHEMA = 'frontier.vllm-authenticated-calibration/v1'
POLICY = 'median3-vanilla-time-majority-correct-min-reference-strict-5pct-no-flip/v5'
EXCLUDED_RUN = 'run_01M5559CGK46W7R879S59DZR9Q'
ENVIRONMENT_FIELDS = frozenset({
    'model_artifact_binding_sha256', 'tokenizer_artifact_binding_sha256', 'instance_manifest_sha256',
    'answer_keys_sha256', 'scientific_runtime_sha256', 'schedule_sha256',
    'serving_parameters_sha256', 'resource_contract_sha256',
})
ARTIFACT_BINDINGS = ('model_artifact_binding_sha256', 'tokenizer_artifact_binding_sha256')


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _require(test, message):
    if not test:
        raise MeasurementError(message)


def _positive(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def correctness_allowance(selector_cardinality):
    _require(type(selector_cardinality) is int and selector_cardinality > 0,
             'correctness allowance requires the full positive selector cardinality')
    return max(1, (selector_cardinality + 19) // 20)


def validate_environment(environment):
    _require(isinstance(environment, dict) and set(environment) == ENVIRONMENT_FIELDS,
             'fixed-target environment requires the exact explicit compatibility fields')
    _require(all(_sha(value) for value in environment.values()),
             'environment compatibility values must bind retained content documents')


def _validate_calibration(capture, environment):
    _require(isinstance(capture, dict) and set(capture) == {
        'schema', 'run_id', 'environment', 'environment_provenance', 'phases'},
        'invalid calibration document fields')
    _require(capture['schema'] == CALIBRATION_SCHEMA, 'wrong calibration schema')
    _require(isinstance(capture['run_id'], str) and capture['run_id'].startswith('run_')
             and capture['run_id'] != EXCLUDED_RUN, 'ineligible calibration run')
    _require(capture['environment'] == environment, 'calibration environment is not comparable')
    provenance = capture['environment_provenance']
    _require(isinstance(provenance, dict) and set(provenance) == set(ARTIFACT_BINDINGS),
             'explicit model/tokenizer artifact binding provenance required')
    for key, item in provenance.items():
        _require(isinstance(item, dict) and set(item) == {
            'binding_kind', 'descriptor', 'original_source', 'direct_file_fingerprints'},
            'invalid model/tokenizer binding provenance')
        _require(item['binding_kind'] == 'source/image-declaration'
                 and item['direct_file_fingerprints'] is None,
                 'source/image binding is not a measured file fingerprint')
        descriptor = item['descriptor']
        _require(isinstance(descriptor, dict) and set(descriptor) == {
            'schema', 'artifact_kind', 'pinned_runtime_image_digest',
            'scientific_model_identity', 'tokenizer_identity'}, 'invalid artifact descriptor')
        _require(descriptor['schema'] == 'frontier.vllm-model-artifact-binding/v1'
                 and descriptor['artifact_kind'] == ('model' if key == ARTIFACT_BINDINGS[0] else 'tokenizer')
                 and isinstance(descriptor['pinned_runtime_image_digest'], str)
                 and descriptor['pinned_runtime_image_digest'].startswith('sha256:')
                 and _sha(descriptor['pinned_runtime_image_digest'][7:])
                 and all(isinstance(descriptor[name], str) and descriptor[name]
                         for name in ('scientific_model_identity', 'tokenizer_identity'))
                 and digest(descriptor) == environment[key], 'artifact descriptor identity/hash mismatch')
        source = item['original_source']
        _require(isinstance(source, dict) and set(source) == {'sha256', 'path_or_member'}
                 and _sha(source['sha256']) and isinstance(source['path_or_member'], str)
                 and source['path_or_member'], 'original artifact declaration source required')
    descriptors = [provenance[k]['descriptor'] for k in ARTIFACT_BINDINGS]
    _require(all(descriptors[0][name] == descriptors[1][name] for name in (
        'pinned_runtime_image_digest', 'scientific_model_identity', 'tokenizer_identity')),
        'model and tokenizer descriptors differ in pinned scientific runtime')
    _require(isinstance(capture['phases'], dict) and capture['phases']
             and set(capture['phases']) <= {'train', 'eval'},
             'each calibration needs independently authenticated original phase evidence')
    for mode, phase in capture['phases'].items():
        _require(isinstance(phase, dict) and set(phase) == {
            'evidence_sha256', 'consumer_receipt_sha256', 'groups'}, 'invalid calibration phase')
        _require(_sha(phase['evidence_sha256']) and _sha(phase['consumer_receipt_sha256']),
                 'calibration must bind original evidence and accepted-consumer receipt')
        _require(set(phase['groups']) == set(expected_groups(mode)), 'wrong calibration groups')
        for group, ids in expected_groups(mode).items():
            variants = phase['groups'][group]
            _require(set(variants) == {'vanilla', 'reference'}, 'calibration requires V/R only')
            for records in variants.values():
                _require(isinstance(records, dict) and set(records) == set(ids),
                         'calibration needs exact phase-specific case coverage')
                for row in records.values():
                    _require(isinstance(row, dict) and set(row) == {
                        'latency_seconds', 'correct', 'evidence_member', 'terminal_sha256'},
                        'invalid calibration measurement fields')
                    _require(_positive(row['latency_seconds']) and isinstance(row['correct'], bool),
                             'calibration needs measured positive timing and boolean correctness')
                    member = row['evidence_member']
                    _require(isinstance(member, str) and member and not member.startswith('/')
                             and '..' not in member.split('/') and _sha(row['terminal_sha256']),
                             'calibration measurement lacks original source provenance')


def _source(capture, mode, group, variant, ident):
    phase = capture['phases'][mode]
    row = phase['groups'][group][variant][ident]
    return {'run_id': capture['run_id'], 'phase': mode, 'group': group,
            'variant': variant, 'instance_id': ident,
            'evidence_sha256': phase['evidence_sha256'],
            'consumer_receipt_sha256': phase['consumer_receipt_sha256'], **row}


def build_targets(captures, *, environment):
    """Median vanilla baseline and reference envelope from three verified captures.

    Callers must verify source signatures/custody and approve capture byte hashes
    before calling. This builder cannot turn unverified normalized JSON into trust.
    """
    validate_environment(environment)
    _require(isinstance(captures, (list, tuple)) and 3 <= len(captures) <= 6,
             'fixed targets require three clean calibration captures per phase')
    captures = sorted(copy.deepcopy(captures), key=lambda c: c.get('run_id', ''))
    for capture in captures:
        _validate_calibration(capture, environment)
    _require(len({c['run_id'] for c in captures}) == len(captures),
             'calibration run documents must be distinct')
    phases = {}
    for mode in ('train', 'eval'):
        phase_captures = [c for c in captures if mode in c['phases']]
        _require(len(phase_captures) == 3,
                 'each phase requires exactly three distinct clean calibration captures')
        groups = {}
        for group, ids in expected_groups(mode).items():
            variants = {}
            for variant in ('vanilla', 'reference'):
                times, timing_sources = {}, {}
                all_flags = {}
                for capture in phase_captures:
                    all_flags[capture['run_id']] = {
                        i: capture['phases'][mode]['groups'][group][variant][i]['correct'] for i in ids}
                counts = {run: sum(flags.values()) for run, flags in all_flags.items()}
                selected_count = sorted(counts.values())[1]
                selected_runs = [run for run, count in counts.items() if count == selected_count]
                for ident in ids:
                    sources = [_source(c, mode, group, variant, ident) for c in phase_captures]
                    observed = sorted(s['latency_seconds'] for s in sources)
                    times[ident] = observed[1] if variant == 'vanilla' else observed[0]
                    timing_sources[ident] = [s for s in sources if s['latency_seconds'] == times[ident]]
                majority = variant == 'vanilla'
                flags = ({i: sum(f[i] for f in all_flags.values()) >= 2 for i in ids} if majority
                         else all_flags[selected_runs[0]])
                observed_tails = {c['run_id']: tail_latencies(
                    c['phases'][mode]['groups'][group][variant][i]['latency_seconds']
                    for i in ids) for c in phase_captures}
                tails = {p: min(t[p] for t in observed_tails.values())
                         for p in ('p90', 'p95', 'p99')}
                if majority:
                    tails = tail_latencies(times.values())
                variants[variant] = {'latency_seconds': times,
                    'timing_sources': timing_sources, 'tails_seconds': tails,
                    'tail_sources': {p: (sorted(all_flags) if majority else
                                        [run for run, t in observed_tails.items() if t[p] == tails[p]])
                                     for p in tails},
                    'observed_group_tails_seconds': observed_tails,
                    'per_case_envelope_tails_diagnostic_seconds': tail_latencies(times.values()),
                    'correct': sum(flags.values()), 'correct_labels': flags,
                    'correctness_policy': ('per_case_majority_of_three' if majority else
                        'median_observed_group_count'),
                    'correctness_sources': all_flags, 'observed_correct_counts': counts,
                    'selected_count_source_runs': sorted(all_flags) if majority else selected_runs}
                if majority:
                    variants[variant].update(timing_policy='per_case_median_of_three',
                        tail_policy='percentiles_of_per_case_median_vector')
                if variant == 'reference':
                    variants[variant].update(selector_cardinality=len(ids),
                        median_correct_count=selected_count,
                        correctness_tolerance=correctness_allowance(len(ids)),
                        required_correct_count=selected_count-correctness_allowance(len(ids)))
            groups[group] = {'expected_ids': list(ids), **variants}
        phases[mode] = groups
    return {'schema': SCHEMA, 'policy': POLICY, 'environment': copy.deepcopy(environment),
            'calibrations': captures, 'margin_fraction': 0,
            'timing_comparison': 'strict_less_than', 'phases': phases}


def validate_targets(targets, *, expected_environment):
    _require(isinstance(targets, dict) and set(targets) == {
        'schema', 'policy', 'environment', 'calibrations', 'margin_fraction',
        'timing_comparison', 'phases'}, 'invalid fixed-target manifest fields')
    _require(targets['schema'] == SCHEMA and targets['policy'] == POLICY,
             'unsupported fixed-target manifest/policy')
    validate_environment(expected_environment)
    _require(targets['environment'] == expected_environment,
             'fixed-target environment differs from current scientific contract')
    _require(type(targets['margin_fraction']) in (int, float) and targets['margin_fraction'] == 0
             and targets['timing_comparison'] == 'strict_less_than',
             'fixed timing comparison must be strict with zero margin')
    rebuilt = build_targets(targets['calibrations'], environment=expected_environment)
    _require(canonical_bytes(rebuilt) == canonical_bytes(targets),
             'fixed-target envelope/provenance differs from deterministic source derivation')
    return copy.deepcopy(targets)


def load_targets(path, *, expected_sha256, expected_environment):
    """Load private release-pinned bytes; never obtain the pin from a candidate."""
    _require(_sha(expected_sha256), 'release must pin a fixed-target SHA256')
    try:
        raw = Path(path).read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == expected_sha256,
                 'fixed-target file SHA256 differs from release pin')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                _require(key not in result, 'duplicate JSON key in fixed targets')
                result[key] = value
            return result
        targets = json.loads(raw, object_pairs_hook=unique,
                             parse_constant=lambda _: (_ for _ in ()).throw(
                                 MeasurementError('non-finite fixed-target JSON')))
        _require(raw == canonical_bytes(targets), 'fixed-target file must use canonical JSON bytes')
        return validate_targets(targets, expected_environment=expected_environment)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise MeasurementError('fixed-target file is missing or invalid') from exc


def evaluate_fixed(candidate, targets, *, mode):
    """Use the existing A-D scorer, then apply immutable reference tail targets.

    The trusted runtime loads pinned targets and authenticates candidate records.
    RuntimeGroupFailure retains the original affected-workload no-credit policy.
    """
    from cpu_reference_verdict import evaluate_train, evaluate_final
    from reference_verdict import InstanceObservation

    targets = validate_targets(targets, expected_environment=targets['environment'])
    expected = expected_groups(mode)
    groups = targets['phases'][mode]
    baseline = {variant: {key: [InstanceObservation(
        ident, groups[key][variant]['latency_seconds'][ident],
        groups[key][variant]['correct_labels'][ident]) for ident in ids]
        for key, ids in expected.items()} for variant in ('vanilla', 'reference')}
    result = (evaluate_train if mode == 'train' else evaluate_final)(
        baseline['vanilla'], candidate, baseline['reference'], bfcl_flip_gate=False)
    for key, verdict in result['reference_verdict']['groups'].items():
        reference = groups[key]['reference']
        verdict['comparisons']['correctness'] = (
            verdict['candidate_correct'] >= reference['required_correct_count'])
        if verdict['complete']:
            verdict['comparisons'].update({p: verdict['candidate_tails'][p] < reference['tails_seconds'][p]
                                           for p in ('p90', 'p95', 'p99')})
        verdict.update(reference_tails=reference['tails_seconds'],
                       reference_correct=reference['correct'],
                       reference_correctness_policy=reference['correctness_policy'],
                       reference_observed_correct_counts=reference['observed_correct_counts'],
                       reference_correctness_tolerance=reference['correctness_tolerance'],
                       required_correct_count=reference['required_correct_count'],
                       timing_margin_fraction=0,
                       reference_values_are_fixed_envelope=True,
                       reference_tail_policy='minimum_independently_observed_group_tail',
                       reference_tail_sources=reference['tail_sources'],
                       timing_equality_passes=False)
        verdict['pass'] = verdict['complete'] and all(verdict['comparisons'].values())
    rv = result['reference_verdict']
    rv['passed_groups'] = sum(v['pass'] for v in rv['groups'].values())
    rv['pass'] = all(v['pass'] for v in rv['groups'].values())
    result['pass'] = result['valid'] and rv['pass']
    result.update(fixed_targets_sha256=digest(targets), fixed_targets_policy=POLICY)
    return result
