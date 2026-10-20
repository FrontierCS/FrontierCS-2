"""Solver-visible diagnostic of a rejected lejepa_world_model submission: fixed categories only.

The evaluator builds the diagnostic from this table and the verifier admits it
to train feedback only after `public_diagnostic` has re-derived it from the
same table. No field can carry a number measured from the submission, a path,
a symbol name, a reference value, a workload identity or text from the
submission.
"""
SCHEMA = "lejepa.candidate-diagnostic/v1"
CATEGORIES = {
    "stateful_loss": "a scored loss (alignment_loss, SIGReg or whitening_loss) is not a pure function of its arguments: on the same input and the same global-generator seed, its value or its gradient with respect to the input differs between a freshly imported loss and one that has already been called",
}


def diagnostic(category):
    """The one admissible diagnostic for a rejection category."""
    if type(category) is not str or category not in CATEGORIES:
        raise ValueError("unknown rejection category")
    return {"schema": SCHEMA, "category": category, "message": f"Rejected: {CATEGORIES[category]}."}


def public_diagnostic(value):
    """Validate an evaluator-supplied diagnostic; return a fresh copy built from the table."""
    if type(value) is not dict or set(value) != {"schema", "category", "message"}:
        raise ValueError("candidate diagnostic has invalid fields")
    if value["schema"] != SCHEMA:
        raise ValueError("candidate diagnostic has an invalid schema")
    expected = diagnostic(value["category"])
    if type(value["message"]) is not str or value["message"] != expected["message"]:
        raise ValueError("candidate diagnostic message is not approved")
    return expected
