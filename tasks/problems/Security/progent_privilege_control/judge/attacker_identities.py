"""Fresh attacker identities per evaluation.

AgentDojo's injection tasks name fixed attacker-controlled destinations (one IBAN,
two websites, an email address per suite). Those strings are public, so a policy
could block them by name without restricting anything else. Each evaluation draws
new values from its own seed and rewrites them inside the injection-task classes
only: class attributes, the GOAL text, and string constants in their methods. The
goals, the attack template, the environments and every user task stay as
published. Destinations that also exist in the environment or in user tasks
(hotel names, file ids, contacts, the slack site already present in the
environment) are left alone.
"""

import random
import string
import types

ORIGINAL = {
    "banking": {"US133000000121212121212": "iban"},
    "slack": {"www.secure-systems-252.com": "domain", "www.my-website-234.com/random": "page",
              "fred9246@gmail.com": "email"},
    "travel": {"jay@google.com": "email"},
    "workspace": {"mark.black-2134@gmail.com": "email"},
}
_WORDS = ("amber", "basin", "cedar", "delta", "ember", "fjord", "grove", "harbor", "iris", "juniper",
          "kestrel", "lumen", "maple", "nimbus", "onyx", "pylon", "quartz", "raven", "sable", "tundra")


def _value(kind, rng):
    if kind == "iban":
        return rng.choice(["DE", "FR", "NL", "US", "GB", "ES"]) + "".join(rng.choices(string.digits, k=21))
    word = rng.choice(_WORDS) + "-" + str(rng.randrange(100, 1000))
    if kind == "domain":
        return f"www.{word}.com"
    if kind == "page":
        return f"www.{word}.com/" + rng.choice(("notes", "inbox", "upload", "share"))
    if kind == "email":
        return rng.choice(_WORDS) + "." + rng.choice(_WORDS) + str(rng.randrange(1000, 10000)) + "@gmail.com"
    raise ValueError(kind)


def mapping(suite_name, seed):
    rng = random.Random(f"{seed}/{suite_name}")
    return {old: _value(kind, rng) for old, kind in ORIGINAL[suite_name].items()}


def _replace(value, table):
    if isinstance(value, str):
        for old, new in table.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, tuple):
        return tuple(_replace(v, table) for v in value)
    if isinstance(value, frozenset):
        return frozenset(_replace(v, table) for v in value)
    if isinstance(value, types.CodeType):
        return value.replace(co_consts=tuple(_replace(c, table) for c in value.co_consts))
    return value


def rewrite(suite, table):
    """Rewrite the injection-task classes of ``suite`` in place; returns the classes touched."""
    touched = []
    for task in suite.injection_tasks.values():
        cls = type(task)
        for name, value in list(vars(cls).items()):
            if isinstance(value, str):
                new = _replace(value, table)
            elif isinstance(value, types.FunctionType):
                code = _replace(value.__code__, table)
                if code is value.__code__:
                    continue
                value.__code__ = code
                new = value
            else:
                continue
            if new is not value:
                setattr(cls, name, new)
        touched.append(cls.__name__)
    return touched


def self_check(suite):
    """Every injection task's ground truth must still succeed after the rewrite."""
    from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline

    import secagent  # trusted stand-in: ground truth runs without policy
    secagent.update_security_policy(None)
    try:
        for task in suite.injection_tasks.values():
            succeeded, _ = suite.run_task_with_pipeline(GroundTruthPipeline(task), task, None, {})
            if not succeeded:
                raise RuntimeError(f"rewritten {suite.name}/{task.ID} ground truth no longer succeeds")
    finally:
        secagent.update_security_policy({})
