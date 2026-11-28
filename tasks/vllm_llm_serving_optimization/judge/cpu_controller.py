"""Surviving CPU authority for a frozen remote operation's exact expected slots."""
from pathlib import Path
import math
import time

from containment_policy import Boundary, child_failure_metadata
from controller_envelope import ControllerSigner
from controller_identity import load_context
from remote_terminal import _retain_prefix, observe_remote
from terminal_contract import canonical, terminal
from terminal_ledger import Ledger


class CPUController:
    def __init__(self, slots, *, environment, content_contract, source_root,
                 evidence, maximum_contract_bytes, maximum_source_bytes,
                 maximum_terminal_bytes, maximum_manifest_bytes, deadline,
                 started):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        # Slots are frozen before any remote dispatch; missing trusted identity,
        # key creation, ledger and manifest errors deliberately propagate.
        self.slots = tuple(slots)
        context = load_context(environment, content_contract, source_root, self.slots,
            maximum_contract_bytes=maximum_contract_bytes, maximum_source_bytes=maximum_source_bytes)
        self.signer = ControllerSigner(context, self.slots, Ed25519PrivateKey.generate(),
            maximum_terminal_bytes=maximum_terminal_bytes,
            maximum_manifest_bytes=maximum_manifest_bytes)
        if (any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n)
                for n in (deadline, started)) or deadline <= started):
            raise ValueError('controller requires actual positive owner work interval')
        self.deadline, self.started = deadline, started
        self.maximum_terminal_bytes = maximum_terminal_bytes
        self.evidence = Path(evidence)
        self.evidence.mkdir(parents=True, exist_ok=False)
        self.ledger = Ledger(self.evidence / 'ledger.sqlite')
        self.by_key = {slot.key(): slot for slot in self.slots}
        for slot in self.slots:
            self.ledger.register(slot, '/'.join((slot.operation_id, slot.group,
                                               slot.variant, slot.instance_id)))
        _retain_prefix(self.evidence / 'manifest.json', canonical(self.signer.manifest()))

    def _owned(self, slot):
        if slot.key() not in self.by_key:
            raise ValueError('slot is outside immutable CPU controller ownership')

    def mark_dispatched(self, slot):
        self._owned(slot)
        return self.ledger.mark_dispatched(slot)

    def _sign(self, slot, record):
        import hashlib
        path = self.evidence / (hashlib.sha256(slot.key().encode()).hexdigest() + '.terminal.json')
        # Repeated observation must not overwrite or re-sign first-winner bytes.
        encoded = canonical(self.signer.sign(record))
        if path.exists():
            with path.open('rb') as stream:
                if stream.read(self.maximum_terminal_bytes + 1) != encoded:
                    raise ValueError('persisted CPU envelope differs from first terminal')
        else:
            _retain_prefix(path, encoded)
        return record

    def observe(self, slot, raw, *, verify, boundary, prefix_bytes):
        import hashlib
        self._owned(slot)
        elapsed = time.monotonic() - self.started
        record = observe_remote(self.ledger, slot, raw, verify=verify, boundary=boundary,
            elapsed=elapsed, cap=self.deadline - self.started,
            maximum_bytes=self.maximum_terminal_bytes, prefix_bytes=prefix_bytes,
            evidence=self.evidence / (hashlib.sha256(slot.key().encode()).hexdigest() + '.raw-prefix'))
        return self._sign(slot, record)

    def fail_unresolved(self, *, boundary, cause='collector_failure'):
        if not isinstance(boundary, Boundary) or boundary.kind == 'platform':
            raise ValueError('CPU closure requires an explicit C1 or C2 boundary')
        closed = []
        for slot in self.slots:
            record = self.ledger.read(slot)
            if record is None:
                record = terminal(slot, status='timeout' if cause == 'deadline' else 'runtime_error',
                    cause=cause, elapsed=time.monotonic() - self.started,
                    cap=self.deadline - self.started)
                record['candidate_input_dispatched'] = self.ledger.dispatched(slot)
                record['failure_context'] = child_failure_metadata(slot, boundary,
                    record['candidate_input_dispatched'])
                record = self.ledger.commit_once(slot, record)[1]
            closed.append(self._sign(slot, record))
        return closed
