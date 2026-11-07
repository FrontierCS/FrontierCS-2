"""CPU ownership precedes candidate server startup, including before first input."""
import hashlib
import json
import time

from containment_policy import Boundary
from driver_client import AuthenticatedProvisioningError, RemoteExecutionError, SidecarExecutionError
from terminal_contract import canonical, terminal


class CPULifecycleEvidenceError(SidecarExecutionError):
    """Local lifecycle evidence IO failed outside remote response parsing."""


def owned_lifecycle(factory, action, variant, document, output, *, deadline, invoke):
    """Contain remote lifecycle failures; CPU allocation and evidence defects propagate."""
    if action not in {'start', 'stop'}:
        raise ValueError('unknown CPU-owned lifecycle operation')
    started = time.monotonic()
    kind = 'startup' if action == 'start' else 'shutdown'
    controller = factory(kind, variant, document,
        'sha256:' + hashlib.sha256(canonical(document)).hexdigest(), output,
        started=started, deadline=deadline)
    boundary = Boundary('candidate_service', 'startup' if action == 'start' else 'execution')
    try:
        result = invoke()
        if (not isinstance(result, dict) or
                result.get('state') != ('ready' if action == 'start' else 'terminated') or
                (action == 'start' and result.get('model_endpoint') != 'http://main:8000/v1') or
                (action == 'start' and any(type(result.get(key)) is not int or result[key] <= 0
                                          for key in ('pid', 'process_start_ticks'))) or
                (action == 'stop' and result.get('remote_artifacts_removed') is not True)):
            raise RemoteExecutionError('parent lifecycle did not reach its required state',
                                       cause='protocol_error')
        try:
            raw = json.dumps(result, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode()
        except (UnicodeError, ValueError, RecursionError) as error:
            raise RemoteExecutionError('parent lifecycle receipt is not canonical finite JSON',
                                       cause='protocol_error') from error
    except RemoteExecutionError as error:
        controller.fail_unresolved(boundary=boundary, cause=error.cause)
        raise
    except (AuthenticatedProvisioningError, CPULifecycleEvidenceError):
        raise
    except Exception as error:
        controller.fail_unresolved(boundary=boundary, cause='unknown_exception')
        raise RemoteExecutionError('unknown parent lifecycle failure', cause='protocol_error') from error
    # Only the CPU emits a lifecycle completion; this is not a model measurement.
    for slot in controller.slots:
        def verified(_):
            return terminal(slot, status='complete', cause='normal_completion',
                elapsed=time.monotonic() - started, cap=deadline - started,
                result={'lifecycle': kind, 'state': result['state']},
                result_limit_bytes=controller.maximum_terminal_bytes, result_limit_depth=8)
        controller.observe(slot, raw, verify=verified, boundary=boundary,
            prefix_bytes=min(2048, controller.maximum_terminal_bytes))
    return result
