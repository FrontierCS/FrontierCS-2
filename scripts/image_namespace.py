"""The repository's one declared image namespace: config/image-namespace.json.

Every tool that names an image this repository builds or publishes, and every check that
compares such an image's repository, derives it from namespace(), read on first use (never
at import). Pinned references (digest-bound, in config/ and tasks/) keep their own full
repository.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "config" / "image-namespace.json"
SCHEMA = "frontier.image-namespace/v1"
# <registry host[:port]>/<path>[/<path>...]: always fully qualified.
NAMESPACE_RE = re.compile(
    r"[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
)


@functools.cache
def namespace() -> str:
    """<registry>/<namespace>, e.g. docker.io/joyemang33; a missing or malformed file fails."""
    document = json.loads(PATH.read_text(encoding="utf-8"))
    if set(document) != {"description", "namespace", "schema_version"} or (
        document["schema_version"] != SCHEMA
    ):
        raise ValueError(f"{PATH}: expected {SCHEMA} with description and namespace")
    value = document["namespace"]
    if not isinstance(value, str) or NAMESPACE_RE.fullmatch(value) is None:
        raise ValueError(f"{PATH}: namespace must be <registry>/<namespace>")
    return value


def repository(name: str) -> str:
    """The repository of an image this repository publishes, e.g. fcs2-codex-overlay."""
    return f"{namespace()}/{name}"
