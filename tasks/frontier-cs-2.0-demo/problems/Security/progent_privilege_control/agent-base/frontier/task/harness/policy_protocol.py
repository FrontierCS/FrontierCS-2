"""Framed transport between the trusted episode runner and the sandboxed policy worker.

Parent -> worker: one JSON line per message, ``{"op": ..., "payload": ...}``.
Worker -> parent: one JSON line per message, either a model request
``{"kind": "llm", "system": str, "user": str, "temperature": float}`` (answered by
the parent with ``{"op": "llm_result", ...}``) or the final reply to the pending
operation ``{"kind": "done", "op": str, "decision"?: "allow"|"deny", "error"?: str}``.

Everything crossing the boundary is plain JSON; the worker never receives Python
objects from the parent and the parent never deserializes anything but bounded JSON.
"""

import json

MAX_REQUEST = 8 * 1024 * 1024
MAX_RESPONSE = 1024 * 1024
MAX_ERROR = 2000  # a denial message is tool feedback the agent sees
OPERATIONS = ("configure", "base", "generate", "update", "check")


def encode(message):
    frame = json.dumps(message, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    return frame


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def parse_worker_message(raw, pending_op):
    """Validate one worker line; returns the dict or raises ValueError."""
    if len(raw) > MAX_RESPONSE or not raw.endswith(b"\n"):
        raise ValueError("invalid worker frame")
    message = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(message, dict):
        raise ValueError("worker frame is not an object")
    kind = message.get("kind")
    if kind == "llm":
        if set(message) != {"kind", "system", "user", "temperature"}:
            raise ValueError("unexpected model request fields")
        if not isinstance(message["system"], str) or not isinstance(message["user"], str):
            raise ValueError("model request text must be strings")
        temperature = message["temperature"]
        if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
            raise ValueError("model request temperature out of range")
        return message
    if kind == "done":
        allowed = {"kind", "op", "decision", "error"}
        if not set(message) <= allowed or message.get("op") != pending_op:
            raise ValueError("unexpected completion")
        if pending_op == "check":
            if message.get("decision") not in ("allow", "deny"):
                raise ValueError("check completion needs a decision")
        elif "decision" in message:
            raise ValueError("decision on a non-check operation")
        if "error" in message and not isinstance(message["error"], str):
            raise ValueError("error must be a string")
        return message
    raise ValueError("unknown worker message kind")


def parse_parent_message(raw):
    if len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
        raise ValueError("invalid parent frame")
    message = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(message, dict) or "op" not in message:
        raise ValueError("invalid parent message")
    return message
