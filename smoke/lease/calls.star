"""The bodies of the lease smoke's judges: each runs `judge.py`, which leases runners through its
target's sessions, and returns what of its result the caller sees."""

PAYLOAD = "/pumpkin/in/payload/raw"
RESULT = "/pumpkin/out/result.json"

# What the train judge's result returns to the agent.
RETURNS = ["score", "lease", "host", "detail"]

def _judged(target, payload, argv):
    """The result that `judge.py`, run as `argv`, writes in `target`."""
    target.put(PAYLOAD, payload)
    target.run(argv)
    return json.decode(target.get(RESULT))

def train_call(target: Target, payload: Payload):
    """The agent's train call: `judge.py` on its session `tlease`, which it leaves open.

    Args:
      target: the train judge's sandbox.
      payload: the agent's submission.

    Returns:
      Each of the result's `RETURNS` that it holds.
    """
    result = _judged(target, payload, ["python3", "-u", "/opt/smoke/judge.py", "tlease", "keep"])
    return {name: result[name] for name in RETURNS if name in result}

def final_call(target: Target, payload: Payload):
    """The final call: `judge.py` on its session `slease`, which it closes, and on `untrusted`.

    Args:
      target: the final judge's sandbox.
      payload: the agent's selected submission.

    Returns:
      The result's `score`, if it holds one.
    """
    result = _judged(target, payload, ["python3", "-u", "/opt/smoke/judge.py", "slease", "close", "untrusted"])
    return {"score": result["score"]} if "score" in result else {}
