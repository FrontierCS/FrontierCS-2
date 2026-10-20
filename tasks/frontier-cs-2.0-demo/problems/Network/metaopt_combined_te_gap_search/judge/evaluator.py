"""Native MetaOpt CPU evaluation; returned metrics contain absolute values only."""
from __future__ import annotations
import json
import math
import os
import subprocess
from pathlib import Path
import legacy_engine as engine

ROOT = Path(__file__).resolve().parent

REJECTION_REASONS = {
    'runtime_unavailable': 'dotnet SDK is unavailable',
    'invalid_project': 'submitted project violates the source boundary',
    'build_failed': 'submitted project did not build',
    'build_timeout': 'submitted project exceeded the build time limit',
    'case_timeout': 'candidate exceeded a case time limit',
    'case_execution_failed': 'candidate exited unsuccessfully on a case',
    'output_limit': 'candidate output exceeded the output size limit',
    'invalid_output': 'candidate output did not match the output contract',
    'invalid_matrix': 'candidate matrix failed validation',
    'invalid_candidate': 'candidate validation failed',
}


def retain_rejection(error):
    category = ('build_timeout' if isinstance(error, subprocess.TimeoutExpired)
                else getattr(error, 'category', 'invalid_candidate'))
    if category not in REJECTION_REASONS:
        category = 'invalid_candidate'
    # Case identities and the original exception are trusted evidence only.
    # Solver feedback uses a closed reason list, never arbitrary child output.
    diagnostic = {
        'schema_version': 'frontier.metaopt-rejection/v1',
        'category': category,
        'exception_type': type(error).__name__,
        'cause_exception_type': (type(error.__cause__).__name__
                                 if error.__cause__ is not None else None),
        'reason': ''.join(c for c in str(error) if c.isprintable())[:1024],
        'case_id': getattr(error, 'case_id', None),
        'return_code': getattr(error, 'return_code', None),
        'child_diagnostics': getattr(error, 'diagnostics', None),
    }
    path = ROOT/'rejection-evidence.json'
    with path.open('w') as stream:
        os.chmod(path, 0o600)
        json.dump(diagnostic, stream, allow_nan=False)
    return f'Submission rejected ({category}): {REJECTION_REASONS[category]}.'


def select_instances(role):
    if role not in ('development', 'final'):
        raise ValueError('trusted submission role is required')
    public = engine._suite('agent')
    return public if role == 'development' else public + engine._suite('final')


def score_gaps(candidate_gaps):
    if not candidate_gaps or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in candidate_gaps):
        raise ValueError('gaps must be finite nonnegative numbers')
    mean_gap = sum(candidate_gaps)/len(candidate_gaps)
    return min(100.0, max(0.0, 100.0*(1.0-math.exp(-mean_gap/40.0)))), mean_gap


def evaluate(solution_path):
    (ROOT/'rejection-evidence.json').unlink(missing_ok=True)
    config = json.loads((ROOT/'scoring_config.json').read_text())
    if config.get('science_approved') is not True:
        raise RuntimeError('MetaOpt scientific alignment is pending')
    instances = select_instances(os.environ.get('FRONTIER_SUBMISSION_ROLE'))
    try:
        answers = engine._run_csharp_project(Path(solution_path), instances)
        _, _, candidate_gaps = engine._score_answers_detailed(instances, answers)
    except (ValueError, subprocess.TimeoutExpired) as error:
        return 0.0, 0.0, retain_rejection(error), {}
    # Candidate processes and helpers exit before any reference is computed.
    reference_answers = engine._reference_answers(instances)
    _, _, reference_gaps = engine._score_answers_detailed(instances, reference_answers)
    identities = [row['id'] for row in instances]
    if len(set(identities)) != len(identities) or len(candidate_gaps) != len(identities) or len(reference_gaps) != len(identities):
        raise RuntimeError('all selected cases require fresh candidate and reference replay')
    score, mean_gap = score_gaps(candidate_gaps)
    reference_score, _ = score_gaps(reference_gaps)
    decisions = {name: {'gap': candidate >= reference} for name, candidate, reference in zip(identities, candidate_gaps, reference_gaps)}
    private_path = ROOT/'case-evidence.json'
    with private_path.open('w') as stream:
        os.chmod(private_path, 0o600)
        json.dump({'case_ids': identities, 'candidate_gaps': candidate_gaps,
                   'reference_gaps': reference_gaps, 'reference_score': reference_score,
                   'margin': score-reference_score,
                   'beats_reference': int(all(row['gap'] for row in decisions.values())),
                   'beat_spec_workload_passes': decisions}, stream, allow_nan=False)
    return score, mean_gap, f'score={score:.6f}; mean_gap={mean_gap:.6f}', {'mean_gap': mean_gap, 'instance_count': len(instances)}
