"""Absolute watchdog around the entire request to a model-used SWE service."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
import time

from containment_policy import Boundary
from instance_supervisor import Limits, run
from terminal_contract import canonical
from terminal_ledger import Ledger


class RequestFailure(RuntimeError):
    def __init__(self, terminal):
        super().__init__('signed case request did not complete within its owned boundary')
        self.terminal = terminal


def request_signed(url, method, headers, body, *, slot, deadline, cleanup_seconds,
                   maximum_frame_bytes, maximum_stderr_bytes):
    """Headers authorize exactly one request; the worker receives no signer.

    The process supervisor bounds wall time even if urllib or a sidecar dribbles
    bytes forever. Its record is a local observation, not an infra credential.
    """
    remaining = deadline - time.monotonic()
    request_digest = 'sha256:' + hashlib.sha256(canonical({
        'url': url, 'method': method, 'body': body})).hexdigest()
    request_slot = replace(slot, request_digest=request_digest)
    if remaining <= 0:
        raise TimeoutError('signed request owner deadline already elapsed')
    limits = Limits(remaining, cleanup_seconds, maximum_frame_bytes, maximum_stderr_bytes, 128)
    with tempfile.TemporaryDirectory(prefix='vllm-signed-request-') as temporary:
        root = Path(temporary)
        request_path = root / 'request.json'
        # This temporary capability is never copied into scientific evidence.
        request_path.write_bytes(canonical({'url': url, 'method': method, 'headers': headers,
            'body': body, 'timeout_seconds': remaining, 'maximum_body_bytes': maximum_frame_bytes}))
        request_path.chmod(0o600)
        ledger = Ledger(root / 'ledger.sqlite')
        ledger.register(request_slot, slot.key())
        record = run(ledger, request_slot,
            [sys.executable, '-I', str(Path(__file__).with_name('http_request_worker.py')),
             '--request-file', str(request_path), '--maximum-frame-bytes', str(maximum_frame_bytes)],
            limits, boundary=Boundary('swe_case', 'tool'))
        if record['status'] != 'complete':
            raise RequestFailure(record)
        return record['result']
