"""Judge of metaopt_combined_te_gap_search for the shared verifier.

evaluate(tree): `tree` is the starter repo (`pristine/`, holding `project/`) with only the
submission's editable files overlaid (`project/MetaOptimize/TrafficEngineering/
BudgetedDemandSearch.cs` and `.../Candidate/`). The C# project is built offline against the
baked NuGet cache and run on each case, each in a fresh confined process (the shared
isolate: no judge access); the candidate's throughput gap on every case is compared
with the fixed reference adversarial search (`gap` passes when candidate >= reference; PASS is
the conjunction over all cases). Score = clip(100 * (1 - exp(-mean_gap / 40)), 0, 100).
check_tree(): the whole-project limits (96 files, 2,000,000 bytes) the old adapter applied.

A candidate failure is a scored zero with every case failed (`beat_spec_workload_passes` all
false), as the old adapter reported it; the closed rejection reason is the message, and the
diagnostic stays in private metrics. Train shows only `mean_gap`, `instance_count` and progress.

Score: raw_score = clip(100 * (1 - exp(-G / 40)), 0, 100), score_unbounded = G, the mean replayed
gap over the selected cases (15 train, 30 final). reference_score is the same formula on the
reference's gaps (0 with no replay for a failed candidate); margin = raw_score - reference_score.
Beat-spec metric (one row per case): gap, higher is better, passes when the candidate's gap is at
least the reference's (ties pass). beats_reference = 1 iff every case passes. The full prior
specification is archived at
docs/task-authoring/metaopt_combined_te_gap_search/provenance/beat_spec.json.
"""
from __future__ import annotations
import json
import math
import os
import subprocess
from pathlib import Path
import legacy_engine as engine

ROOT = Path(__file__).resolve().parent
MAX_PROJECT_FILES = 96
MAX_PROJECT_BYTES = 2_000_000

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
    """The closed public reason and the private diagnostic of a candidate failure."""
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
    return f'Submission rejected ({category}): {REJECTION_REASONS[category]}.', diagnostic


def check_tree(tree, pristine, changes):
    """The resulting project stays within the old limits: 96 files, 2,000,000 bytes."""
    files = [path for path in (Path(tree) / 'project').rglob('*') if path.is_file()]
    if len(files) > MAX_PROJECT_FILES:
        return 'too_many_files', []
    if sum(path.stat().st_size for path in files) > MAX_PROJECT_BYTES:
        return 'too_large', []
    return None


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


def evaluate(tree):
    config = json.loads((ROOT/'scoring_config.json').read_text())
    if config.get('science_approved') is not True:
        raise RuntimeError('MetaOpt scientific alignment is pending')
    instances = select_instances(os.environ.get('FRONTIER_SUBMISSION_ROLE'))
    identities = [row['id'] for row in instances]
    if len(set(identities)) != len(identities):
        raise RuntimeError('case identities must be unique')
    # The Verifier image links /tmp/.dotnet (.NET named mutexes; .NET 8 ignores TMPDIR for them)
    # to this writable directory. .NET makes shm/ and lockfiles/ with mkdtemp in /tmp + rename(),
    # which fails across devices when /frontier is a mounted volume (the Verifier pod's
    # emptyDir), so they are made here, world-writable as .NET would make them.
    try:
        for sub in ('', 'shm', 'lockfiles'):
            path = Path('/frontier/tmp/.dotnet') / sub
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o777)
    except OSError:
        pass
    try:
        answers = engine._run_csharp_project(Path(tree)/'project', instances)
        _, _, candidate_gaps = engine._score_answers_detailed(instances, answers)
    except (ValueError, subprocess.TimeoutExpired) as error:
        message, diagnostic = retain_rejection(error)
        # As before: an invalid candidate fails every case; the reference was not run.
        return 0.0, 0.0, message, {
            'beats_reference': 0,
            'beat_spec_workload_passes': [{'workload': name, 'metrics': {'gap': False}} for name in identities],
            'reference_score': 0.0, 'margin': 0.0, 'reference_replayed': False,
            'rejection_evidence': diagnostic}
    # Candidate processes and helpers exit before any reference is computed.
    reference_answers = engine._reference_answers(instances)
    _, _, reference_gaps = engine._score_answers_detailed(instances, reference_answers)
    if len(candidate_gaps) != len(identities) or len(reference_gaps) != len(identities):
        raise RuntimeError('all selected cases require fresh candidate and reference replay')
    score, mean_gap = score_gaps(candidate_gaps)
    reference_score, _ = score_gaps(reference_gaps)
    rows = [{'workload': name, 'metrics': {'gap': candidate >= reference}}
            for name, candidate, reference in zip(identities, candidate_gaps, reference_gaps)]
    return score, mean_gap, f'score={score:.6f}; mean_gap={mean_gap:.6f}', {
        'mean_gap': mean_gap, 'instance_count': len(instances),
        'beats_reference': int(all(row['metrics']['gap'] for row in rows)),
        'beat_spec_workload_passes': rows,
        'reference_score': reference_score, 'margin': score - reference_score,
        'case_ids': identities, 'candidate_gaps': candidate_gaps, 'reference_gaps': reference_gaps}
