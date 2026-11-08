#!/usr/bin/env python3
"""The search runs of both sides, on the trusted search machine (task.toml [test_sandboxes.search]).

    run_search.py ROOT

The machine is the Verifier image: the built harness, solver and instrumented programs are its
/opt/crest-task (CREST_TASK_BUILT_ROOT), as on the judge. judge/evaluator.py uploads, as ROOT
(the sandbox's workdir): this file, judge/strategy_bridge.py, each side's compiled strategy
(sides/<role>/strategy) and jobs.json: every (program, role, seed) run in the order to start
them (per seed the side that starts alternates) and the number run at once.

Every job runs through the starter's crest_runner.run_one, as on the judge before: the fixed
search harness drives the program under test and measures coverage and search overhead. The
strategy it starts is sides/<role>/strategy_host, strategy_bridge.py with this interpreter's
line, which runs the side's strategy in a fresh shared-isolate instance on the harness's pipes:
the harness, the solver and the programs under test stay outside every instance, and no
instance sees the other side's strategy. The results go to ROOT/out/results.json, the harness
result of each job in job order; a launcher failure (a record in ROOT/launcher-failures/, never
an exit status) fails this step instead.
"""
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
built = Path(os.environ['CREST_TASK_BUILT_ROOT'])
tree = built / 'tree'
sys.path.insert(0, str(tree))
import crest_runner  # noqa: E402
import frontier_build  # noqa: E402

jobs = json.loads((root / 'jobs.json').read_text())
scratch = root / 'scratch'
env, _ = frontier_build.environment(tree, built / 'build', tmp=scratch / 'tmp')
bridge = (root / 'strategy_bridge.py').read_text()
hosts = {}
for role in ('candidate', 'reference'):
    host = root / 'sides' / role / 'strategy_host'
    # The harness starts it with an empty environment: the interpreter line names this one.
    host.write_text('#!%s -I\n%s' % (sys.executable, bridge))
    host.chmod(0o555)
    hosts[role] = host
work = []
for number, job in enumerate(jobs['runs']):
    spec = crest_runner.ProgramSpec(built / 'bench' / job['directory'], job['argv'],
                                    needs_pty=job['needs_pty'], term=job['term'])
    work.append((number, dict(repo=tree, spec=spec, strategy_host=hosts[job['role']],
                              seed=job['seed'], scratch=scratch, relabel_key=job['relabel_key'],
                              run_wall_seconds=job['run_wall_seconds'], toolchain_env=env)))
failures = root / 'launcher-failures'
failures.mkdir()
results = crest_runner.run_many(work, jobs['workers'])
records = sorted(failures.iterdir())
if records:   # the isolate launcher failed: the judge's fault, never a run's status
    sys.exit('isolate launcher failed in %d runs: %s' % (len(records), records[0].read_text()[:500]))
(root / 'out' / 'results.json').write_text(json.dumps([results[n] for n in range(len(work))]))
