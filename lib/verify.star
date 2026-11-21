"""The bodies that run the shared verifier: the train function's and the final call's. Each runs
the verifier once, in a fresh sandbox of its target, on the payload, and returns what of its
result the judge's caller sees."""

load("//lib/submit.star", "submission")

# Where the verifier reads the payload, which its target's environment names
# `FRONTIER_PAYLOAD_PATH`.
PAYLOAD = "/pumpkin/in/payload/raw"

# The verifier, which reads its mode, its selector and its paths from its target's environment.
VERIFIER = ["/opt/frontier/verifier"]

# Where the verifier writes its result, which its target's environment names
# `FRONTIER_RESULT_PATH`, `/frontier/output/result.json`, through the link `/frontier/output`.
RESULT = "/pumpkin/out/result.json"

def _verified(target, payload):
    """The result that the verifier writes for `payload` in `target`, whatever its exit status."""
    target.put(PAYLOAD, payload)
    target.run(VERIFIER)
    return json.decode(target.get(RESULT))

def train_call(target: Target, payload: Payload, *, metrics: list[str]):
    """A train call: the score, the metrics and the failure that the agent sees.

    Args:
      target: the train judge's sandbox.
      payload: the agent's train submission.
      metrics: the names of the result's `metrics` that the agent sees.

    Returns:
      The result's `raw_score`, each of `metrics` that its `metrics` holds, and its `failure`,
      each that it holds, by its name.
    """
    result = _verified(target, payload)
    found = result.get("metrics", {})
    raw = {"raw_score": result["raw_score"]} if "raw_score" in result else {}
    shown = {name: found[name] for name in metrics if name in found}
    failure = {"failure": result["failure"]} if "failure" in result else {}
    return raw | shown | failure

def final_call(target: Target, submissions):
    """The final call: the run's score, status and verdict against the reference, and the rest.

    No agent sees the final's result, so it is returned whole: its `private_metrics` hold the
    verdict against the reference, which its `metrics` leave out.

    Args:
      target: the final judge's sandbox.
      submissions: the last saved payload by agent role name.

    Returns:
      The result's `raw_score` as `score`, its `status`, the `beats_reference` of its
      `private_metrics`, then its `summary`, `metrics`, `private_metrics` and `failure`, each that
      it holds, by its name.
    """
    result = _verified(target, submission(submissions))
    private = result.get("private_metrics", {})
    shown = {"score": result["raw_score"]} if "raw_score" in result else {}
    status = {"status": result["status"]} if "status" in result else {}
    beats = {"beats_reference": private["beats_reference"]} if "beats_reference" in private else {}
    names = ["summary", "metrics", "private_metrics", "failure"]
    rest = {name: result[name] for name in names if name in result}
    return shown | status | beats | rest
