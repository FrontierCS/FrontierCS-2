"""Lease smoke judge calls. The train judge leaves its lease open for teardown. The final
body uses a scoped runner for the measurement and keeps the isolation probe on a separate runner."""

JUDGE = ["python3", "-u", "/opt/smoke/judge.py"]
PAYLOAD = "/pumpkin/in/payload/raw"
RESULT = "/pumpkin/out/result.json"

# What the train judge's result returns to the agent.
RETURNS = ["score", "lease", "host", "detail"]

def train_call(target: Target, payload: Payload):
    """The agent's train call: `judge.py` on its session `tlease`, which it leaves open.

    Args:
      target: the train judge's sandbox.
      payload: the agent's submission.

    Returns:
      Each of the result's `RETURNS` that it holds.
    """
    target.put(PAYLOAD, payload)
    target.run(JUDGE + ["train"])
    result = json.decode(target.get(RESULT))
    return {name: result[name] for name in RETURNS if name in result}

def _measure(runner: Runner):
    """Measure the payload, then return the count file and the original modes tree to the judge."""
    runner.put("/pumpkin/in/payload", "/workspace/sub")
    measured = runner.run(JUDGE + ["measure"])
    runner.get("/workspace/res/bytes", "/pumpkin/out/measurement")
    runner.put("/pumpkin/home/modes", "/workspace/modes")
    modes = runner.run(JUDGE + ["show-modes"])
    runner.get("/workspace/modes", "/pumpkin/home/modes-back")
    return [measured["stdout_tail"], modes["stdout_tail"]]

def final_call(target: Target, submissions):
    """The final judge checks its scoped score runner and its separate untrusted lease.

    Args:
      target: the final judge's sandbox.
      submissions: the last saved payload by agent role name.

    Returns:
      The result's `score`, if it holds one; a `score` of zero for an agent that saved nothing.
    """
    payload = submissions["agent"]
    if payload == None:
        return {"score": 0.0}
    target.put(PAYLOAD, payload)
    lease = target.capture(JUDGE + ["prepare-final"])
    host, modes = target.with_runner("slease", _measure)
    target.run(JUDGE + ["finish-final", lease, host, modes])
    result = json.decode(target.get(RESULT))
    return {"score": result["score"]} if "score" in result else {}
