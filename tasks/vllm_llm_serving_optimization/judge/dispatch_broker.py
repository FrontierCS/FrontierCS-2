"""Trusted scheduler broker supervises fresh episode execs and owns tool auth."""
from dataclasses import asdict
from pathlib import Path
import sys
import time

from bounded_http import RequestFailure
from containment_policy import Boundary
from instance_supervisor import CaseBoundaryFailure, Limits, TrustedControllerError, run


def run_episode(ledger, slot, instance, *, workload, endpoint, settings, testbed,
                deadline, cleanup_seconds, maximum_frame_bytes, maximum_stderr_bytes):
    from batch_coordinator import BackendError
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('episode reached its existing owner deadline before dispatch')
    if testbed is not None:
        testbed.transport = dict(slot=slot, deadline=deadline, cleanup_seconds=cleanup_seconds,
            maximum_frame_bytes=maximum_frame_bytes, maximum_stderr_bytes=maximum_stderr_bytes)
    def rpc(method, arguments, owner_deadline):
        if testbed is None or testbed.tainted:
            raise CaseBoundaryFailure('case testbed is absent or already tainted')
        if method == 'run':
            if (set(arguments) != {'command', 'timeout'} or not isinstance(arguments['command'], str)
                    or type(arguments['timeout']) is not int or arguments['timeout'] != 60):
                raise CaseBoundaryFailure('worker changed original SWE command interface')
        elif method == 'read_patch':
            if arguments:
                raise CaseBoundaryFailure('worker changed original SWE patch interface')
        else:
            raise CaseBoundaryFailure('worker requested an unknown tool method')
        try:
            return (list(testbed.run(arguments['command'], timeout=arguments['timeout']))
                    if method == 'run' else testbed.read_patch())
        except TrustedControllerError:
            raise
        except (RequestFailure, TimeoutError) as error:
            testbed.tainted = True
            raise CaseBoundaryFailure('model-used SWE request did not complete') from error
        except Exception as error:
            # Testbed protocol/identity/case_failure is a C2 observation. Local
            # filesystem/key defects retain explicit trusted-error boundaries.
            if isinstance(error, BackendError):
                testbed.tainted = True
                raise CaseBoundaryFailure('model-used SWE service rejected the operation') from error
            raise TrustedControllerError('trusted broker request failed') from error
    # Isolated Python ignores environment flags, including bytecode suppression.
    return run(ledger, slot,
        [sys.executable, '-I', '-B', str(Path(__file__).with_name('episode_worker.py')),
         '--maximum-frame-bytes', str(maximum_frame_bytes)],
        Limits(remaining, cleanup_seconds, maximum_frame_bytes, maximum_stderr_bytes, 128),
        boundary=Boundary('swe_case' if workload == 'swe' else 'candidate_service', 'execution'),
        input_document={'workload': workload, 'instance': instance, 'endpoint': endpoint,
            'request_timeout_seconds': 900 if workload == 'swe' else 300, 'settings': asdict(settings)},
        rpc=rpc)
