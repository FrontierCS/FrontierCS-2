"""What the search harness starts as a side's strategy_host (judge/run_search.py writes it there
with its interpreter line): the side's compiled strategy in a fresh shared-isolate instance.

The harness forks this process with its two pipes on fds 0 and 1 and measures it as it measured
the strategy before: the exit status (codes 70-72, a killing signal) and the wait4 resource
usage. isolate.exec_on_stdio makes this process the instance's launcher on those pipes, so the
harness sees the strategy's own exit status, whatever it is, and its CPU time is the strategy's
plus a fixed cost per run (this interpreter, the backend's self-test, the instance's setup and
cleanup). A failure of the launcher itself is never an exit status: it writes a record in the
search root's launcher-failures/ (outside every instance), and run_search.py fails the
verification as the judge's own fault.

The instance reads only the system directories and this side's directory (the strategy); the
other side's strategy, the programs under test, the harness's run directories and the judge's
files are not visible. The strategy seals itself further (strategy_host.cc: seccomp, no new
descriptors) before any solution code runs.
"""
import os
import time

import isolate  # the shared launcher, on the Verifier image's import path (frontier.pth)

# The harness ends a run within its wall budget plus HARD_TIMEOUT_SLACK_SECONDS (crest_runner);
# the instance's own limit only backs that up.
LIFETIME_SECONDS = 3600

side = os.path.dirname(os.path.abspath(__file__))
record = os.path.join(os.path.dirname(os.path.dirname(side)), 'launcher-failures',
                      '%d-%d.json' % (os.getpid(), time.time_ns()))
isolate.exec_on_stdio([os.path.join(side, 'strategy')], failure_record=record,
                      timeout_seconds=LIFETIME_SECONDS, read_paths=[side])
