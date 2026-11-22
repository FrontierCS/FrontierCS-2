"""The body of the agent's now function `final`, by which it submits."""

FINAL_HELP = "Save or replace your submission. Set finalize=true to lock it and end your work. Returns stored or accepted."

def submit(run: Run, payload: Payload, finalize: bool = False) -> dict[str, str]:
    """Selects `payload` for the final call, replacing the last selection.

    Args:
      run: the run's state.
      payload: the agent's submission.
      finalize: whether the call also ends the agent's work.

    Returns:
      The selection's state: `stored`, or `accepted` once the work has ended.
    """
    run.set_submission(payload)
    if finalize:
        run.end_work()
        return {"state": "accepted"}
    return {"state": "stored"}
