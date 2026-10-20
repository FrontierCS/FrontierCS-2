"""Solver-visible diagnostic of a rejected dynamic_tanh submission: fixed categories only.

The evaluator builds the diagnostic from these tables and the verifier admits
it to train feedback only after `public_diagnostic` has re-derived it from the
same tables. No field can carry a number, a threshold, a reference value, a
workload identity, a repeat index or text from the submission.
"""
SCHEMA = "dyt.candidate-diagnostic/v1"
STAGES = {
    "patch_policy": "the submission patch was not accepted: allowed paths, size, file count or application to the pristine tree",
    "semantic_validation": "the conversion checks or the single-precision numerical guards on a freshly converted model failed, or the submission raised an error there",
    "model_workload": "training through the unchanged entrypoint, loading the trained state or running a timed block failed or raised an error",
    "timed_path": "a result produced under the timed conditions failed its comparison",
}
BLOCKS = {"training": "training block", "inference": "inference block"}
QUANTITIES = {
    "output": "model output", "loss": "loss",
    "gradient_joint": "gradient of all parameters jointly",
    "gradient_other": "gradient of the parameters outside the replacement sites",
    "gradient_alpha": "gradient of the replacement sites' alpha",
    "gradient_scale": "gradient of the replacement sites' weight (scale)",
    "gradient_shift": "gradient of the replacement sites' bias (shift)",
    "shape": "missing or wrong-shape result",
    "nonfinite": "nonfinite values that do not match",
}


def diagnostic(stage, block=None, quantity=None):
    """The one admissible diagnostic for a stage, or for a timed-path block and quantity."""
    if type(stage) is not str or stage not in STAGES:
        raise ValueError("unknown rejection stage")
    if stage == "timed_path":
        if type(block) is not str or block not in BLOCKS or type(quantity) is not str or quantity not in QUANTITIES:
            raise ValueError("unknown timed-path category")
        message = f"Rejected: {STAGES[stage]} ({BLOCKS[block]}, {QUANTITIES[quantity]})."
    else:
        if block is not None or quantity is not None:
            raise ValueError("only a timed-path rejection names a block and a quantity")
        message = f"Rejected: {STAGES[stage]}."
    return {"schema": SCHEMA, "stage": stage, "block": block, "quantity": quantity, "message": message}


def public_diagnostic(value):
    """Validate an evaluator-supplied diagnostic; return a fresh copy built from the tables."""
    if type(value) is not dict or set(value) != {"schema", "stage", "block", "quantity", "message"}:
        raise ValueError("candidate diagnostic has invalid fields")
    if value["schema"] != SCHEMA:
        raise ValueError("candidate diagnostic has an invalid schema")
    expected = diagnostic(value["stage"], value["block"], value["quantity"])
    if type(value["message"]) is not str or value["message"] != expected["message"]:
        raise ValueError("candidate diagnostic message is not approved")
    return expected
