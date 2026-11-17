"""Identity of the interpreter actually used for measurement; no ML imports."""

import hashlib
import importlib.metadata
import json
import sys


def identity():
    digest = hashlib.sha256()
    with open(sys.executable, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"python": sys.version, "executable_sha256": digest.hexdigest(),
            "base_prefix": sys.base_prefix,
            "distributions": sorted((d.metadata["Name"], d.version)
                                    for d in importlib.metadata.distributions())}


if __name__ == "__main__":
    print(json.dumps(identity(), sort_keys=True, allow_nan=False))
