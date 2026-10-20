"""Solver-visible diagnostic of a rejected edlib submission: fixed categories only.

The evaluator builds the diagnostic from this table and the verifier admits it
to train feedback only after `public_diagnostic` has re-derived it from the
same table. No field can carry a number, a path, a symbol name, a reference
value, a workload identity or text from the submission.
"""
SCHEMA = "edlib.candidate-diagnostic/v1"
OUTSIDE = "patch modifies source outside the permitted function body"
CATEGORIES = {
    "patch_policy": "the submission patch was not accepted: allowed file, size, file count or application to the pristine tree",
    "outside_before_body": OUTSIDE + " (before the body)",
    "outside_after_body": OUTSIDE + " (after the body)",
    "body_scope": "the function body does not stay inside its function: it closes the function, or uses a line continuation, a trigraph or a raw string literal",
    "body_forbidden_construct": "the function body contains a preprocessor directive or a call that accesses the environment, processes, files or timing",
    "launch_surface": "the built program's start-up or runtime symbols differ from the reference build",
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
