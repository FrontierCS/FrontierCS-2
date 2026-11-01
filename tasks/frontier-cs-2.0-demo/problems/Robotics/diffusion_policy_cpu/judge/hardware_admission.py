"""Hardware admission for the sealed naive timing constants (issue #639).

Ln and Tn are wall-clock measurements, so an admitted host must have the sealed
architecture and exact CPU model strings. The kernel patch release and the CPU
IDs are not part of the class: any kernel of the sealed series and distribution
build is admitted (6.12.*-*.amzn2023.x86_64 for 6.12.110-135.201.amzn2023.x86_64),
with the same number of allowed CPUs. A kernel release outside that layout is
admitted only by exact equality.
"""

import re

IDENTITY_FIELDS = {"architecture", "kernel", "cpu_models", "affinity"}
KERNEL_RELEASE = re.compile(r"(\d+)\.(\d+)\.\d+-\d+(?:\.\d+)*\.([a-z][a-z0-9_]*\.[a-z0-9_]+)")


def hardware_class(identity):
    """Return the admission class of a measured identity, or None if it is malformed."""
    if not (isinstance(identity, dict) and set(identity) == IDENTITY_FIELDS):
        return None
    architecture, kernel = identity["architecture"], identity["kernel"]
    models, cpus = identity["cpu_models"], identity["affinity"]
    if not (isinstance(architecture, str) and architecture and isinstance(kernel, str) and kernel
            and isinstance(models, list) and models
            and all(isinstance(model, str) and model for model in models)
            and isinstance(cpus, list) and cpus
            and all(type(cpu) is int and cpu >= 0 for cpu in cpus) and len(set(cpus)) == len(cpus)):
        return None
    match = KERNEL_RELEASE.fullmatch(kernel)
    series = {"kernel": kernel} if match is None else {"kernel_series": "{}.{}.*-*.{}".format(*match.groups())}
    return {"architecture": architecture, "cpu_models": sorted(set(models)), "cpus": len(cpus), **series}


def admits(sealed, actual):
    """Exact equality, as before, or membership in the sealed identity's class."""
    if sealed == actual:
        return True
    accepted = hardware_class(sealed)
    return accepted is not None and accepted == hardware_class(actual)
