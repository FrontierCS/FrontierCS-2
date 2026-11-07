"""Signed correctness service of one reused SWE check worker.

One container per original instance runs the canonical correctness request of
every serving variant in the fixed order vanilla, reference, candidate. The
first run uses the image as prepared; later runs follow a verified reset. The
trusted scheduler is the only caller: each request is signed with the
operation's attestation key, and every receipt returns to the CPU Verifier
through the scheduler's signed summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

from instance_manifest import expected_groups
from . import sidecar_env
from .accuracy import (CorrectnessRuntimeError, OVERSIZE_REQUEST_SCHEMA, SWE_TEST_TIMEOUT_SECONDS, build_test_request,
                       needs_startup_control)
from .swe_tool_service import IDENTITY_FIELDS, bootstrap_identity, make_server
from .swe_case_failure import CaseDeadlineExceeded, failure, remaining
from .swe_test_worker import PreparedSWEWorker
from .testbed_reset import ResetError, TestbedReset, kill_other_processes
from .tool_auth import ToolAuthVerifier

TESTBED = Path('/testbed')
VARIANT_ORDER = ('vanilla', 'reference', 'candidate')


class CorrectnessService:
    routes = frozenset({('GET', '/ready'), ('POST', '/correctness')})

    def __init__(self, identity, *, max_request_bytes, max_output_bytes, auth=None,
                 testbed=None, reset=None, worker_factory=None, process_cleanup=None,
                 variants=VARIANT_ORDER):
        if variants not in (VARIANT_ORDER, ('candidate',)):
            raise ValueError('correctness order must match an immutable evaluation policy')
        self.variants = variants
        if (not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS
                or any(not isinstance(value, str) or not value for value in identity.values())):
            raise ValueError('expected exactly the six nonempty prepared identity fields')
        if identity['instance_id'] not in expected_groups('eval')['swe/public']:
            raise ValueError('instance is not in the frozen SWE inventory')
        for value in (max_request_bytes, max_output_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('publisher byte limits must be positive integers')
        self.identity = dict(identity)
        self.testbed = TESTBED if testbed is None else Path(testbed)
        self.auth = auth if auth is not None else ToolAuthVerifier.from_environment()
        context = self.auth.context
        if (context['operation_id'] != self.identity['operation_id']
                or context['sidecar_id'] != self.identity['child_id']):
            raise ValueError('tool authentication context differs from sidecar identity')
        self.max_request_bytes = max_request_bytes
        self.max_output_bytes = max_output_bytes
        self.reset = TestbedReset(self.testbed, output_limit=max_output_bytes) if reset is None else reset
        self.worker_factory = PreparedSWEWorker if worker_factory is None else worker_factory
        self.process_cleanup = kill_other_processes if process_cleanup is None else process_cleanup
        self.snapshot = None
        self.completed = []
        self.command_lock = threading.Lock()

    def prepare(self):
        """Take the image inventory before the first run; fail closed if unclean."""
        self.snapshot = self.reset.snapshot()
        return self.snapshot

    def handle(self, method, path, body):
        if (method, path) == ('GET', '/ready'):
            return 200, {'ready': self.snapshot is not None and (self.testbed / '.git').exists(),
                         'completed_variants': list(self.completed)}
        return self.correctness(body)

    def _startup_control(self, request, receipt, variant, started):
        if not needs_startup_control(request, receipt):
            return
        control = {'schema': 'frontier.vllm-fixed-base-startup-control/v1',
                   'request_sha256': request['request_sha256'], 'variant': variant,
                   'state': 'unavailable'}
        receipt['startup_control'] = control
        deadline = started + request['timeout_seconds']
        executions = receipt.get('patch_attempts', []) + [receipt.get(name, {}) for name in
                      ('syntax_probe_execution', 'eval_execution')]
        consumed = sum(row.get('output_bytes_retained', 0) + row.get('stage_bytes_retained', 0)
                       for row in executions)
        remaining_bytes = self.max_output_bytes - consumed
        try:
            if time.monotonic() >= deadline or remaining_bytes <= 0:
                raise CorrectnessRuntimeError('startup control has no original request budget remaining')
            self.process_cleanup()
            control['reset'] = self.reset.reset(deadline=deadline)
            baseline_request = build_test_request(request['instance_id'], '')
            worker = self.worker_factory(instance_id=self.identity['instance_id'], testbed=self.testbed,
                                         max_output_bytes=remaining_bytes)
            control['receipt'] = worker.run_fixed_base_control(baseline_request, deadline=deadline)
            control['elapsed_seconds'] = time.monotonic() - started
            if time.monotonic() > deadline:
                raise CorrectnessRuntimeError('startup control exhausted the original request deadline')
            # A control shares the existing response allocation, including its metadata.
            if len(json.dumps(receipt).encode()) > self.max_output_bytes:
                raise CorrectnessRuntimeError('startup control exceeded the original response allocation')
            control['elapsed_seconds'] = time.monotonic() - started
            if time.monotonic() > deadline:
                raise CorrectnessRuntimeError('startup control exhausted the original request deadline')
            control['state'] = 'complete'
        except (CorrectnessRuntimeError, ResetError, subprocess.TimeoutExpired) as error:
            control.pop('receipt', None)
            control['error_type'] = type(error).__name__

    def correctness(self, body):
        if not isinstance(body, dict) or set(body) != {'identity', 'variant', 'request'}:
            return 400, {'error': 'invalid request fields'}
        if body['identity'] != self.identity:
            return 409, {'error': 'prepared identity mismatch'}
        request, variant = body['request'], body['variant']
        if not isinstance(request, dict) or request.get('instance_id') != self.identity['instance_id']:
            return 400, {'error': 'request selects another fixed instance'}
        if (request.get('schema') == OVERSIZE_REQUEST_SCHEMA
                and (request.get('correctness_identity') != self.identity
                     or request.get('variant') != variant)):
            return 409, {'error': 'oversize disposition selects another worker or variant'}
        timeout = request.get('timeout_seconds')
        if type(timeout) is not int or timeout != SWE_TEST_TIMEOUT_SECONDS:
            return 400, {'error': 'a finite canonical correctness timeout is required'}
        if self.snapshot is None:
            return 503, {'error': 'testbed inventory is not prepared'}
        if not self.command_lock.acquire(blocking=False):
            return 409, {'error': 'correctness already running'}
        try:
            expected = self.variants[len(self.completed)] if len(self.completed) < len(self.variants) else None
            if variant != expected:
                return 409, {'error': 'variant differs from the fixed correctness order',
                             'expected_variant': expected}
            started = time.monotonic()
            deadline = started + timeout
            reset_receipt = None
            response = None
            try:
                if self.completed or variant != 'vanilla':
                    remaining(deadline)
                    self.process_cleanup()
                    remaining(deadline)
                    reset_receipt = self.reset.reset(deadline=deadline)
                remaining(deadline)
                worker = self.worker_factory(instance_id=self.identity['instance_id'], testbed=self.testbed,
                                             max_output_bytes=self.max_output_bytes)
                worker._request_deadline = deadline
                receipt = worker.run_correctness(request)
                remaining(deadline)
                self._startup_control(request, receipt, variant, started)
                remaining(deadline)
                # The fixed response allocation includes metadata and escaping.
                if len(json.dumps(receipt, allow_nan=False).encode()) > self.max_output_bytes:
                    raise CorrectnessRuntimeError('correctness receipt exceeds the published output allowance')
                remaining(deadline)
                # Retain the fixed variant ordinal; candidate-only does not claim
                # that the two preceding legacy variants were executed.
                response = (200, {'variant': variant, 'run_index': VARIANT_ORDER.index(variant),
                                  'receipt': receipt, 'reset': reset_receipt})
            except ResetError as error:
                response = (503, {'error': 'testbed reset failed', 'error_type': type(error).__name__,
                                  'case_failure': failure('correctness', error)})
            except CorrectnessRuntimeError as error:
                response = (422, {'error': 'correctness request rejected', 'error_type': type(error).__name__,
                                  'case_failure': failure('correctness', error)})
            except BaseException as error:
                response = (422, {'error': 'SWE case execution failed',
                                  'case_failure': failure('correctness', error,
                                      reason='deadline' if isinstance(error, CaseDeadlineExceeded)
                                      else 'boundary_error')})
            finally:
                try:
                    self.process_cleanup()
                    remaining(deadline)
                except BaseException as error:
                    cleanup = failure('cleanup', error,
                                      reason='deadline' if isinstance(error, CaseDeadlineExceeded)
                                      else 'boundary_error')
                    if response is not None and response[0] != 200:
                        response[1]['cleanup_failure'] = cleanup
                    else:
                        response = (422, {'error': 'SWE case cleanup failed', 'case_failure': cleanup})
            if response[0] == 200:
                self.completed.append(variant)
            return response
        finally:
            self.command_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver-identity-file', type=Path, required=True)
    parser.add_argument('--instance-id', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--max-request-bytes', required=True, type=int)
    parser.add_argument('--max-output-bytes', required=True, type=int)
    parser.add_argument('--variants', choices=('legacy', 'candidate'), default='legacy')
    args = parser.parse_args()
    driver_identity = sidecar_env.install('verifier')
    identity = bootstrap_identity(args.instance_id, args.driver_identity_file, driver_identity)
    service = CorrectnessService(identity, max_request_bytes=args.max_request_bytes,
                                 max_output_bytes=args.max_output_bytes,
                                 auth=ToolAuthVerifier.from_environment(),
                                 variants=('candidate',) if args.variants == 'candidate' else VARIANT_ORDER)
    sidecar_env.scrub()
    service.prepare()
    with make_server(service, listen_profile='bridge', host=args.host, port=args.port) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
