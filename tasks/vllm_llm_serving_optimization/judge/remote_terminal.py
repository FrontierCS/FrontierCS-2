"""Normalize untrusted remote terminal bytes at the surviving CPU boundary."""
import os
from pathlib import Path

from containment_policy import Boundary, child_failure_metadata
from instance_supervisor import Capture
from terminal_contract import canonical, terminal, validate_record


def _retain_prefix(path, prefix):
    path = Path(path)
    pending = path.with_suffix('.pending')
    with pending.open('xb') as stream:
        stream.write(prefix)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def observe_remote(ledger, slot, raw, *, verify, boundary, elapsed, cap,
                   maximum_bytes, prefix_bytes, evidence):
    """Verifier authenticates remote bytes; only this CPU commits the terminal.

    Remote parse/signature/shape failures are candidate-domain observations.
    Local ledger, evidence persistence and CPU signer failures are outside the
    remote-validation catch and remain explicit trusted-controller defects.
    """
    if (not isinstance(boundary, Boundary) or boundary.kind == 'platform'
            or type(maximum_bytes) is not int or maximum_bytes <= 0
            or type(prefix_bytes) is not int or not 0 < prefix_bytes <= maximum_bytes
            or not isinstance(raw, bytes) or not callable(verify)):
        raise ValueError('typed CPU boundary and explicit remote byte bounds required')
    existing = ledger.read(slot)
    if existing is not None:
        return existing
    # The original raw prefix is evidence even when parsing or signature fails.
    # No remote identity selects its own evidence destination.
    _retain_prefix(evidence, raw[:prefix_bytes])
    rejected = False
    try:
        if len(raw) > maximum_bytes:
            raise ValueError('remote record exceeds publisher allocation')
        record = verify(raw)
        validate_record(record)
        if canonical(record['slot']) != slot.key().encode():
            raise ValueError('remote terminal belongs to another expected slot')
    except Exception:
        rejected = True
    dispatched = ledger.dispatched(slot)
    if rejected:
        capture = Capture(prefix_bytes)
        capture.add(raw[:prefix_bytes + 1])
        capture.complete = len(raw) <= prefix_bytes
        record = terminal(slot, status='runtime_error', cause='protocol_error',
            elapsed=elapsed, cap=cap, streams={'stdout': capture.summary()})
    else:
        # A remote terminal can report measurements, but cannot rewrite CPU
        # dispatch observations or claim an independent platform exception.
        record = dict(record)
        record.pop('failure_context', None)
    record['candidate_input_dispatched'] = dispatched
    if record['status'] != 'complete':
        record['failure_context'] = child_failure_metadata(slot, boundary, dispatched)
    return ledger.commit_once(slot, record)[1]
