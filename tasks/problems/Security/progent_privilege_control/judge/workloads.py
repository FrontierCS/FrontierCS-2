"""Workloads: the four AgentDojo v1.1.2 suites, user tasks split into train and final.

The split is a fixed hash ranking (about 40% of each suite's user tasks are held out
for final). Every injection task of a suite is paired with every selected user task
under the ``important_instructions`` attack; each selected user task also runs once
without an attack (benign utility, reported but not gated).
"""

import hashlib
import math

SUITES = ("banking", "slack", "travel", "workspace")
SPLIT_SALT = "progent-real-agent-split-v1"
FINAL_FRACTION = 0.4


def _rank(suite, task_id):
    return hashlib.sha256(f"{SPLIT_SALT}/{suite}/{task_id}".encode()).hexdigest()


def split(suite, user_task_ids):
    ordered = sorted(user_task_ids, key=lambda t: _rank(suite, t))
    n_final = math.ceil(FINAL_FRACTION * len(ordered))
    final = sorted(ordered[:n_final], key=_natural)
    train = sorted(ordered[n_final:], key=_natural)
    return train, final


def _natural(task_id):
    return int(task_id.rsplit("_", 1)[1])


def jobs(suite, user_task_ids, injection_task_ids, role):
    train, final = split(suite, user_task_ids)
    selected = final if role == "final" else train
    rows = [(u, None) for u in selected]
    rows += [(u, i) for u in selected for i in sorted(injection_task_ids, key=_natural)]
    return rows
