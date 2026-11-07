"""Bounded trusted report capture before disposable workload directories close."""

import json
import os
from pathlib import Path
import time

MAX_BYTES = 192 * 1024
MAX_REPORT_BYTES = 16 * 1024
MAX_EVENTS = 32


class Journal:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.document = {"schema_version": 1, "events": []}

    def record(self, event, **fields):
        if self.path is None:
            return
        row = {"event": event, "time_unix": time.time(), **fields}
        if len(self.document["events"]) >= MAX_EVENTS:
            raise ValueError("private workload evidence event limit")
        document = {**self.document, "events": [*self.document["events"], row]}
        body = json.dumps(document, allow_nan=False, ensure_ascii=True).encode("ascii")
        if len(body) > MAX_BYTES:
            raise ValueError("private workload evidence byte limit")
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
        temporary.replace(self.path)
        self.document = document


def report_bytes(path):
    with Path(path).open("rb") as stream:
        body = stream.read(MAX_REPORT_BYTES + 1)
    if len(body) > MAX_REPORT_BYTES:
        raise ValueError("trusted phase report byte limit")
    return body


def read(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("invalid private workload evidence file")
    with path.open("rb") as stream:
        body = stream.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("private workload evidence byte limit")
    document = json.loads(body)
    if (set(document) != {"schema_version", "events"} or document["schema_version"] != 1
            or not isinstance(document["events"], list) or len(document["events"]) > MAX_EVENTS):
        raise ValueError("invalid private workload evidence")
    return document
