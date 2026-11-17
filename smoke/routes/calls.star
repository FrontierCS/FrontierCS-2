"""The bodies of the routes smoke's judges: each runs `routes.py judge`, which asks the model for a
digit through its target's route and writes it as its result's score."""

PAYLOAD = "/pumpkin/in/payload/raw"
RESULT = "/pumpkin/out/result.json"

def _judged(target, payload, chat_model):
    """The result that `routes.py judge` writes in `target`, asking `chat_model`."""
    target.put(PAYLOAD, payload)
    target.run(["python3", "-u", "/opt/routes/routes.py", "judge", chat_model])
    return json.decode(target.get(RESULT))

def train_call(target: Target, payload: Payload, *, chat_model: str):
    """The agent's train call.

    Args:
      target: the train judge's sandbox.
      payload: the agent's submission.
      chat_model: the model the judge asks, which its route allows.

    Returns:
      The result's `score` and `answer`, each that it holds.
    """
    result = _judged(target, payload, chat_model)
    return {name: result[name] for name in ["score", "answer"] if name in result}

def final_call(target: Target, payload: Payload, *, chat_model: str):
    """The final call.

    Args:
      target: the final judge's sandbox.
      payload: the agent's selected submission.
      chat_model: the model the judge asks, which its route allows.

    Returns:
      The result's `score`, if it holds one.
    """
    result = _judged(target, payload, chat_model)
    return {"score": result["score"]} if "score" in result else {}
