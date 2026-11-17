#!/usr/bin/env python3
"""Runs your strategy on the development program and prints what it covered.

usage: run_local.py [--seeds 1-8] [--budget 4000] [--jobs 4]

Run ./frontier-build.sh once first.  Each seed is one run of the search
harness under the same limits the judge applies.  Branch ids are relabelled
per run, here with a key derived from the seed; the judge uses its own keys.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'benchmarks'))
import build_grep  # noqa: E402
import crest_runner  # noqa: E402
import frontier_build  # noqa: E402
import solution_policy  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1-8')
    parser.add_argument('--budget', type=int, default=crest_runner.BUDGET_ITERATIONS)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--json', help='also write the raw per-run results to this file')
    args = parser.parse_args()
    first, last = (args.seeds.split('-') + [args.seeds])[:2]
    seeds = list(range(int(first), int(last) + 1))
    program = HERE / 'benchmarks/grep'
    if not (program / 'grep').is_file():
        raise SystemExit('benchmarks/grep is not built; run ./frontier-build.sh')
    env, _ = frontier_build.environment(HERE, HERE / '.frontier-build')
    try:
        host = frontier_build.build_strategy(HERE, HERE / 'src/frontier_solution',
                                             HERE / '.frontier-build/strategy', env)
    except solution_policy.PolicyViolation as error:
        raise SystemExit('solution rejected: ' + str(error))
    except frontier_build.BuildError as error:
        raise SystemExit(str(error))
    spec = crest_runner.ProgramSpec(program, build_grep.ARGV)
    scratch = Path(tempfile.mkdtemp(prefix='runs-', dir=str(HERE / '.frontier-build')))
    jobs = [(seed, dict(repo=HERE, spec=spec, strategy_host=host, seed=seed, scratch=scratch,
                        budget=args.budget, toolchain_env=env)) for seed in seeds]
    results = crest_runner.run_many(jobs, args.jobs)
    print('seed  covered  iterations  overhead_s  solver_attempts  status')
    for seed in seeds:
        r = results[seed]
        print('%4d  %7d  %10d  %10.2f  %15d  %s' % (
            seed, r['covered'], r['iterations'], crest_runner.search_overhead_seconds(r),
            r.get('solver_attempts', 0), r['status']))
    covered = [results[seed]['covered'] for seed in seeds]
    print('mean covered %.1f over %d seeds' % (sum(covered) / float(len(covered)), len(covered)))
    if args.json:
        Path(args.json).write_text(json.dumps([results[seed] for seed in seeds], indent=1))


if __name__ == '__main__':
    main()
