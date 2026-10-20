"""Scrub only task child operations; the evaluator executes serially."""
import os
from contextlib import contextmanager

@contextmanager
def private_child_environment(environment):
    # The canonical patch helper inherits its environment when spawning Git.
    # Scrub it here without modifying the shared helper. Evaluator is single-threaded.
    previous=dict(os.environ)
    environment=dict(environment)
    environment.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null')
    if 'FRONTIER_SUBMISSION_ROLE' in previous:
        environment['FRONTIER_SUBMISSION_ROLE']=previous['FRONTIER_SUBMISSION_ROLE']
    try:
        os.environ.clear();os.environ.update(environment)
        yield
    finally:
        os.environ.clear();os.environ.update(previous)

