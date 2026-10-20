"""Fixed public diagnostic for a typed candidate source-policy rejection."""

SCHEMA = "diffusion-policy.candidate-diagnostic/v1"
MESSAGE = (
    "Rejected: the submission violates the patch or source policy; only the "
    "conditional_sample body may change, and signature whitespace is protected."
)


def diagnostic(category):
    if type(category) is not str or category != "source_policy":
        raise ValueError("unknown rejection category")
    return {"schema": SCHEMA, "category": category, "message": MESSAGE}


def public_diagnostic(value):
    if type(value) is not dict or set(value) != {"schema", "category", "message"}:
        raise ValueError("invalid diagnostic fields")
    expected = diagnostic(value["category"])
    if value != expected:
        raise ValueError("invalid diagnostic")
    return expected
