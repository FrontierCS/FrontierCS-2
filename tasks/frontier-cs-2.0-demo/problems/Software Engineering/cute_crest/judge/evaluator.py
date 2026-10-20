#!/usr/bin/env python3
"""Judge of the CREST search-strategy task.

The candidate strategy and the private reference strategy (CREST's
control-flow-directed search) are each run on the same seeds of every graded
program, interleaved in one pool so that both see the same machine load.
gate.py turns the per-seed coverage and search-overhead measurements into the
PASS decision; this file owns the runs, validity and the continuous score.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

TASK = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TASK))
from submission_patch import apply_submission_patch  # noqa: E402
import gate  # noqa: E402

# Built once per verifier image by build_benchmarks.py; read-only afterwards.
# Kept outside the task bundle so a verification does not copy it.
BUILT = Path(os.environ.get('CREST_TASK_BUILT_ROOT', '/opt/crest-task'))
CONFIG = {'submission': {'kind': 'patch', 'path': '/app/solution.patch',
                         'allow_paths': ['src/frontier_solution/'],
                         'max_changed_files': 32, 'max_patch_bytes': 262144}}
SPEC = json.loads((TASK / 'beat_spec.json').read_text())
METRIC_NAMES = [m['name'] for m in SPEC['metrics']]
assert METRIC_NAMES == ['branches_covered', 'search_overhead_seconds']
PROGRAMS = json.loads((TASK / 'programs.json').read_text())
WORKERS = 8


class InvalidCandidate(Exception):
    """The submission cannot be scored.  `code` is safe to show the solver."""

    def __init__(self, code):
        Exception.__init__(self, code)
        self.code = code


class EvaluatorFailure(RuntimeError):
    """The judge itself could not measure.  Never a candidate verdict."""


def suite(final):
    return PROGRAMS['final'] if final else PROGRAMS['development']


def workload_name(index):
    return 'w%d' % index


def empty_metrics(final):
    return {'beats_reference': 0, 'reference_score': 0.0, 'margin': 0.0,
            'beat_spec_workload_passes': [
                {'workload': workload_name(i), 'metrics': {name: False for name in METRIC_NAMES}}
                for i in range(len(suite(final)))]}


def relabel_key(program, seed):
    """Per-run key of the secret id permutation: the same for candidate and
    reference, unknown to the solver, stable across evaluations."""
    text = '%s|%s|%d' % (PROGRAMS['relabel_salt'], program['id'], seed)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') >> 1


def seeds_of(program):
    return [int(program['seed_base']) + i for i in range(gate.K_SEEDS)]


def program_score(program, mean_coverage):
    """Share, in percent, of the branches the bounded-DFS baseline leaves
    uncovered that the strategy covers at budget.  Uses the program's branch
    count and the pinned baseline mean only, so the score of a candidate does
    not depend on any reference measurement."""
    baseline = float(program['baseline_mean_coverage'])
    total = float(program['branch_count'])
    return 100.0 * max(0.0, mean_coverage - baseline) / (total - baseline)


def score_of(programs, coverage_lists):
    """Continuous score: mean over the graded programs of program_score()."""
    parts = [program_score(p, sum(c) / float(len(c))) for p, c in zip(programs, coverage_lists)]
    return sum(parts) / len(parts)


def overhead_ms(result):
    return int(round(1000.0 * (float(result['strategy_cpu_seconds']) + float(result['solver_cpu_seconds']))))


def load_tools():
    tree = BUILT / 'tree'
    if not (tree / 'bin/search_harness').is_file():
        raise EvaluatorFailure('built harness is missing')
    sys.path.insert(0, str(tree))
    try:
        import crest_runner
        import frontier_build
        import solution_policy
    finally:
        sys.path.remove(str(tree))
    return tree, crest_runner, frontier_build, solution_policy


def build_candidate(tools, solution_dir, out_dir, env):
    tree, _, frontier_build, solution_policy = tools
    try:
        return frontier_build.build_strategy(tree, solution_dir, out_dir, env)
    except solution_policy.PolicyViolation as error:
        raise InvalidCandidate(error.code)
    except frontier_build.BuildError:
        raise InvalidCandidate('build_failed')


def measure(tools, programs, hosts, scratch, env):
    """Runs every (program, role, seed).  Returns {(program index, role): [result per seed]}."""
    tree, crest_runner, _, _ = tools
    jobs = []
    for index, program in enumerate(programs):
        spec = crest_runner.ProgramSpec(BUILT / 'bench' / program['directory'], program['argv'],
                                        needs_pty=bool(program['needs_pty']), term=program['term'])
        for n, seed in enumerate(seeds_of(program)):
            # Alternate which side starts, so neither is systematically first.
            for role in (('candidate', 'reference') if n % 2 == 0 else ('reference', 'candidate')):
                jobs.append(((index, role, seed), dict(
                    repo=tree, spec=spec, strategy_host=hosts[role], seed=seed, scratch=scratch,
                    relabel_key=relabel_key(program, seed), run_wall_seconds=int(program['run_wall_seconds']),
                    toolchain_env=env)))
    results = crest_runner.run_many(jobs, WORKERS)
    return {(index, role): [results[(index, role, seed)] for seed in seeds_of(program)]
            for index, program in enumerate(programs) for role in ('candidate', 'reference')}


def grade(tools, programs, measured, final):
    _, crest_runner, _, _ = tools
    rows = []
    for index, program in enumerate(programs):
        candidate, reference = measured[(index, 'candidate')], measured[(index, 'reference')]
        if any(r['status'] != 'budget_exhausted' for r in reference):
            raise EvaluatorFailure('a reference run did not complete its budget')
        # A candidate run may end early (budget caps, or Run() returned): it
        # keeps the coverage it reached.  Anything else is a fault of the
        # strategy process and voids the submission.
        faults = sorted({r['status'] for r in candidate if r['status'] not in crest_runner.COMPLETE_STATUSES})
        if any(s == 'harness_failure' or s.startswith('harness_error') for s in faults):
            raise EvaluatorFailure('a candidate run was lost: ' + ','.join(faults))
        if faults:
            raise InvalidCandidate(faults[0].replace(':', '_'))
        rows.append({'workload': workload_name(index), 'candidate_valid': True,
                     'candidate_coverage': [int(r['covered']) for r in candidate],
                     'reference_coverage': [int(r['covered']) for r in reference],
                     'candidate_overhead_ms': [overhead_ms(r) for r in candidate],
                     'reference_overhead_ms': [overhead_ms(r) for r in reference]})
    beats, decisions = gate.decide(rows)
    score = score_of(programs, [row['candidate_coverage'] for row in rows])
    reference_score = score_of(programs, [row['reference_coverage'] for row in rows])
    all_candidate = [r for index in range(len(programs)) for r in measured[(index, 'candidate')]]
    private = []
    for index, row in enumerate(rows):
        k = float(gate.K_SEEDS)
        private.append({
            'workload': row['workload'],
            'candidate_mean_coverage': sum(row['candidate_coverage']) / k,
            'reference_mean_coverage': sum(row['reference_coverage']) / k,
            'coverage_margin_required': gate.margin(row['candidate_coverage'], row['reference_coverage']),
            'overhead_margin_required_seconds': gate.margin(row['candidate_overhead_ms'], row['reference_overhead_ms']) / 1000.0,
            'candidate_mean_overhead_seconds': sum(row['candidate_overhead_ms']) / k / 1000.0,
            'reference_mean_overhead_seconds': sum(row['reference_overhead_ms']) / k / 1000.0,
            'candidate_coverage': row['candidate_coverage'], 'reference_coverage': row['reference_coverage'],
            'candidate_overhead_ms': row['candidate_overhead_ms'],
            'reference_overhead_ms': row['reference_overhead_ms'],
            'candidate_statuses': [r['status'] for r in measured[(index, 'candidate')]],
            'candidate_solver_attempts': [int(r['solver_attempts']) for r in measured[(index, 'candidate')]],
            'reference_solver_attempts': [int(r['solver_attempts']) for r in measured[(index, 'reference')]],
            'candidate_wall_seconds': [r['wall_seconds'] for r in measured[(index, 'candidate')]],
            'reference_wall_seconds': [r['wall_seconds'] for r in measured[(index, 'reference')]],
        })
    metrics = {'beats_reference': beats, 'beat_spec_workload_passes': decisions,
               'reference_score': reference_score, 'margin': score - reference_score,
               'per_workload_comparisons': private}
    if not final:
        # Development feedback: the candidate's own measurements only.
        n = float(len(all_candidate))
        checkpoints = sorted({point[0] for r in all_candidate for point in r.get('curve', [])})
        curve = []
        for checkpoint in checkpoints:
            values = [point[1] for r in all_candidate for point in r.get('curve', []) if point[0] == checkpoint]
            # Runs that ended before a checkpoint keep their final coverage.
            values += [int(r['covered']) for r in all_candidate
                       if all(point[0] != checkpoint for point in r.get('curve', []))]
            curve.append([checkpoint, round(sum(values) / n, 2)])
        metrics.update({
            'dev_branches_covered_mean': round(sum(int(r['covered']) for r in all_candidate) / n, 3),
            'dev_coverage_curve': curve,
            'dev_search_overhead_seconds_mean': round(sum(overhead_ms(r) for r in all_candidate) / n / 1000.0, 3),
            'dev_solver_attempts_mean': round(sum(int(r['solver_attempts']) for r in all_candidate) / n, 1),
            'dev_runs_ended_early': sum(1 for r in all_candidate if r['status'] != 'budget_exhausted'),
            'dev_killed_executions_mean': round(sum(int(r.get('killed_executions', 0)) for r in all_candidate) / n, 2),
        })
    return score, metrics


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    final = os.environ.get('FRONTIER_SUBMISSION_ROLE') == 'final'
    cleanup = []
    previous_tempdir = tempfile.tempdir
    try:
        solution = Path(solution_path).resolve()
        scratch = Path(os.environ.get('CREST_TASK_SCRATCH') or TASK / '.judge-cache')
        scratch.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(scratch)
        work = Path(tempfile.mkdtemp(prefix='measure-', dir=scratch))
        cleanup.append(work)
        try:
            if not solution.is_file():
                raise InvalidCandidate('missing_submission')
            candidate = apply_submission_patch(solution, TASK, CONFIG)
            if not candidate.ok:
                raise InvalidCandidate('invalid_patch')
            cleanup.append(candidate.cleanup_root)
            tools = load_tools()
            env, _ = tools[2].environment(tools[0], BUILT / 'build', tmp=work / 'tmp')
            hosts = {'candidate': build_candidate(tools, candidate.workdir / 'src/frontier_solution',
                                                  work / 'candidate', env)}
        except InvalidCandidate as error:
            return 0.0, 0.0, 'invalid candidate: ' + error.code, empty_metrics(final)
        reference = apply_submission_patch(TASK / 'reference.patch', TASK, CONFIG)
        if not reference.ok:
            raise EvaluatorFailure('the private reference patch does not apply')
        cleanup.append(reference.cleanup_root)
        try:
            hosts['reference'] = build_candidate(tools, reference.workdir / 'src/frontier_solution',
                                                 work / 'reference', env)
        except InvalidCandidate as error:
            raise EvaluatorFailure('the private reference does not build: ' + error.code)
        programs = suite(final)
        runs = work / 'runs'
        runs.mkdir()
        measured = measure(tools, programs, hosts, runs, env)
        try:
            score, metrics = grade(tools, programs, measured, final)
        except InvalidCandidate as error:
            return 0.0, 0.0, 'invalid candidate: ' + error.code, empty_metrics(final)
        return score, score, 'evaluation complete', metrics
    finally:
        tempfile.tempdir = previous_tempdir
        for path in reversed(cleanup):
            if path:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    result = evaluate(sys.argv[1] if len(sys.argv) > 1 else '')
    print(json.dumps({'score': result[0], 'score_unbounded': result[1], 'message': result[2],
                      'metrics': result[3]}, allow_nan=False))
