"""Solver-visible diagnostic of a rejected RUNLTS submission: fixed categories only.

The evaluator builds the diagnostic from this table and the verifier admits it
to train feedback only after `public_diagnostic` has re-derived it from the
same table. No field can carry a number measured from the submission, a path,
a symbol name, a reference value, a trace identity or text from the submission.
The only numbers in the sentences are the public constants of the statement.
"""
SCHEMA = "runlts.candidate-diagnostic/v1"
CATEGORIES = {
    "patch_policy": "the submission patch was not accepted: it is unreadable, empty, malformed, over the size or file-count limit, touches a path other than my_cond_branch_predictor.cc and solver_extensions/, or does not apply to the pristine tree",
    "outside_editable_regions": "the patch changes bytes of my_cond_branch_predictor.cc outside the nine editable regions, or changes another retained file; every byte between the regions is protected, and a change to whitespace, a blank line or a line ending there counts",
    "lexical_form": "an editable region or a solver_extensions file contains a backslash, two consecutive question marks, a %: token or a raw string literal (comments included), or a comment, string or character literal that is not closed inside that region or file",
    "preprocessor_directive": "added code contains a preprocessor directive other than a quoted include of an existing file under solver_extensions/",
    "forbidden_identifier": "added code contains a forbidden word outside comments: an identifier beginning with two underscores, template, or the name of an allocation, input/output, process, cast or dynamic-container facility (for example new, delete, free, read, write, open, exit, signal, raise, function, vector); the whole word is matched, so your own names must not use these words",
    "forbidden_construct": "added code uses a construct outside the permitted C++ subset: a template or lambda, a raw pointer or raw array, raw-array indexing, address-of or dereference, a C-style, const, reinterpret or pointer/reference cast, dynamic allocation, a mutable reference alias, a function returning a pointer or reference, a compiler builtin, or a call that is unresolved, indirect or outside the closed external call set",
    "protected_state_access": "added code refers to a retained predictor name outside the listed available names, or writes to retained state other than pv.LSUM and extra_weight",
    "state_declaration": "persistent state is declared with a type whose logical width cannot be audited; use fixed-width scalars, bit-fields, structs of those and std::array with a constant extent, without unions, inheritance or empty types",
    "semantic_audit": "the patched source has C++ errors in the audit's C++17 parse",
    "storage_cap": "declared logical predictor storage exceeds the cap of 1,572,864 bits (192 KiB)",
    "source_audit_incomplete": "the source audit of the patched predictor exceeded its time or memory limit",
    "build_failed": "the patched predictor did not compile and link with the evaluator's build command, or the build exceeded its time limit",
    "simulator_run_failed": "a simulator run of the patched predictor exited with an error, was stopped by a resource limit or exceeded its time limit",
    "output_format": "the simulator output of a run did not contain exactly one well-formed copy of each original measurement section with consistent counters and rates",
    "instruction_count_mismatch": "the instruction counts reported by a run of the patched predictor differ from the expected counts of that trace, which are those of the reference run",
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
