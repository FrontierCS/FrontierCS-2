"""Complete operator-only phase on one parent with its declared trusted scheduler.

The submitted vLLM process stays in the GPU parent. Poisson timing, BFCL state,
signed SWE calls and batch output stay in the scheduler sidecar the Driver
started beside the parent; everything it reports crosses the parent signed.
"""
import hashlib
import json
from pathlib import Path

from batch_coordinator import FINAL_HIDDEN_JPS, PreparedGroup
from cpu_batch_measurements import project_complete_batch, validate_batch_plan, expected_input_digest
from batch_transport import prepared_testbeds, scheduler_document
from candidate_projection import runtime_group_failure, timed_case_failures, timed_case_bounds
from correctness_phase import run_swe_correctness
from driver_client import RemoteExecutionError, SidecarExecutionError
from instance_manifest import expected_groups
from cpu_reference_verdict import TrustedMeasurementError, MeasurementError, evaluate_final, evaluate_train
from relay_client import SchedulerRelay, TrustedRelayEvidenceError
from phase_controller import PhaseControllerFactory
from phase_lifecycle import CPULifecycleEvidenceError, owned_lifecycle
from runtime_config import validate_projection, execution_variants
from scheduler_stream import decode
from sidecar_catalog import VARIANTS


SOURCE_COMMIT = 'b8b302cde434df8c9289a2b465406b47ebab1c2d'
MODEL_ENDPOINT = 'http://main:8000/v1'


def source_digest(patch):
    return 'sha256:' + hashlib.sha256(
        (SOURCE_COMMIT + '\0' + hashlib.sha256(patch).hexdigest()).encode()).hexdigest()


def _parent_action(session, action, document, *, remote, output, startup_seconds,
                   stop_seconds, max_stdout_bytes, max_stderr_bytes):
    raw = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    result_path = session.run([
        '/opt/frontier/task-runtime', '/opt/frontier/runtime-python',
        '/opt/frontier/judge/variant_batch.py', action,
        '--root', remote, '--max-input-bytes', str(len(raw)),
        '--max-output-bytes', str(max_stdout_bytes),
        '--max-diagnostic-bytes', str(max_stderr_bytes),
        '--startup-seconds', str(startup_seconds), '--stop-seconds', str(stop_seconds),
    ], stdin=raw, output=output, max_stdout_bytes=max_stdout_bytes,
       max_stderr_bytes=max_stderr_bytes)
    try:
        (Path(output) / 'request.json').write_bytes(raw)
        response = result_path.read_bytes()
    except OSError as error:
        raise CPULifecycleEvidenceError('CPU lifecycle evidence IO failed') from error
    try:
        result = decode(response)
    except (UnicodeError, ValueError, RecursionError):
        raise RemoteExecutionError('parent lifecycle did not return one JSON receipt',
                                   cause='protocol_error') from None
    if (not isinstance(result, dict)
            or result.get('schema') != 'frontier.vllm-parent-server/v1'
            or result.get('variant') != document['variant']
            or result.get('model_source_digest') != document['model_source_digest']):
        raise RemoteExecutionError('parent lifecycle returned a different variant identity',
                                   cause='protocol_error')
    return result


def _owned_parent_action(factory, session, action, document, **arguments):
    if factory is None:
        return _parent_action(session, action, document, **arguments)
    output = Path(arguments['output'])
    return owned_lifecycle(factory, action, document['variant'], document,
        output.with_name(output.name + '-ownership'), deadline=session.client.deadline,
        invoke=lambda: _parent_action(session, action, document, **arguments))


def _validate_shutdown(payload, group, *, model_case_failures=frozenset()):
    shutdown = payload.get('testbed_shutdown')
    expected = {bed.child_id for bed in group.testbeds}
    ids = {bed.instance_id for bed in group.testbeds}
    if not isinstance(model_case_failures, frozenset) or not model_case_failures.issubset(ids):
        raise ValueError('CPU-owned case failure set required for tainted bed shutdown')
    tainted = {bed.child_id for bed in group.testbeds if bed.instance_id in model_case_failures}
    if (not isinstance(shutdown, dict) or set(shutdown) != expected
            or any(value != 'confirmed' for name, value in shutdown.items() if name not in tainted)):
        raise SidecarExecutionError('all interactive testbeds of the variant must confirm shutdown')


def _validate_owned_payload(kind, variant, payload, *, groups, factory, calibration, input_digests=None):
    if kind != 'batch':
        return
    group = groups[variant]
    try:
        validate_batch_plan(payload.get('batch'), group, calibration=calibration,
                            input_digest=None if input_digests is None else input_digests[variant])
    except MeasurementError as error:
        raise RemoteExecutionError('remote batch differs from the frozen CPU plan',
                                   cause='protocol_error') from error
    if group.workload == 'swe':
        try:
            failures = (timed_case_failures(factory, group, variant=variant)
                if factory is not None and
                   runtime_group_failure(factory, group, variant=variant) is None else frozenset())
        except Exception as error:
            raise TrustedRelayEvidenceError('CPU timed ownership evidence failed') from error
        try:
            _validate_shutdown(payload, group, model_case_failures=failures)
        except SidecarExecutionError as error:
            raise RemoteExecutionError(str(error), cause='protocol_error') from error


def run_phase(session, config, plan, *, candidate_patch, evidence, startup_seconds, stop_seconds,
              max_stdout_bytes, max_stderr_bytes, calibration=False, hidden_jps=None, selected_group=None,
              controller_environment=None, controller_audit=None):
    """Caller validates the selected archive/patch before allocating this parent."""
    validate_projection(config)
    variants = execution_variants(config)
    mode = session.mode
    expected = expected_groups(mode)
    if selected_group is not None and selected_group not in expected:
        raise SidecarExecutionError('selected group differs from the phase workload')
    if mode == 'eval' and hidden_jps != FINAL_HIDDEN_JPS:
        raise SidecarExecutionError('final requires the configured hidden arrival rates')
    if mode == 'train' and hidden_jps is not None:
        raise SidecarExecutionError('train cannot select hidden rates')
    if any(type(n) is not int or n <= 0 for n in (startup_seconds, stop_seconds)):
        raise SidecarExecutionError('explicit publisher startup and shutdown bounds required')
    if not isinstance(plan, dict) or not isinstance(plan.get('relay'), dict) or not isinstance(plan.get('correctness'), dict):
        raise SidecarExecutionError('phase requires the publisher relay and correctness bounds')
    if not isinstance(getattr(session, 'sidecars', None), dict):
        raise SidecarExecutionError('session has no validated declared sidecars')
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    patches = {'candidate': candidate_patch}
    if variants == VARIANTS:
        patches.update(vanilla=b'', reference=(Path(__file__).parents[1]/'judge/reference.patch').read_bytes())
    if not isinstance(candidate_patch, bytes) or len(candidate_patch) > 1_500_000:
        raise SidecarExecutionError('candidate patch must already satisfy the original payload bound')
    relay_limits, correctness_limits = plan['relay'], plan['correctness']
    observations = {variant: {} for variant in variants}
    for group_index, group_name in enumerate(expected):
        if selected_group is not None and group_name != selected_group:
            continue
        workload, rate = group_name.split('/')
        group_path = evidence / f'group-{group_index:02d}'
        group_path.mkdir()
        groups = {}
        for variant in variants:
            beds = prepared_testbeds(mode, variant, session.sidecars['names']) if workload == 'swe' else ()
            groups[variant] = PreparedGroup(
                session.runner_id, session.runner_id, session.runner_id,
                mode, workload, rate, MODEL_ENDPOINT, source_digest(patches[variant]), beds)
        all_groups = [groups[variant] for variant in variants]
        ownership, factory = {}, None
        if controller_environment is not None:
            factory = PhaseControllerFactory(session, groups, source_root=Path(__file__).parents[1],
                maximum_terminal_bytes=max_stdout_bytes, maximum_manifest_bytes=max_stdout_bytes,
                environment=controller_environment, audit_plan=controller_audit)
            ownership = {'stream': True, 'controller_factory': factory,
                         'cleanup_seconds': plan['phases'][mode]['cleanup_seconds']}
        # Read trusted fixed inputs before the remote payload-validation catch.
        input_digests = {variant: expected_input_digest(groups[variant]) for variant in variants}
        relay = SchedulerRelay(session, public_key=session.sidecars['public_key'], relay_limits=relay_limits,
            max_stderr_bytes=max_stderr_bytes, evidence=group_path / 'relay',
            payload_validator=lambda kind, variant, payload: _validate_owned_payload(
                kind, variant, payload, groups=groups, factory=factory, calibration=calibration,
                input_digests=input_digests),
            **ownership)
        batches, variant_failures = {}, {}
        for variant in variants:
            candidate_failure = None
            remote = f'/workspace/vllm-phase/{group_index:02d}-{variant}'
            start_request = {'variant': variant,
                'patch': patches[variant].decode('utf-8'),
                'model_source_digest': groups[variant].model_source_digest}
            try:
                start_receipt = _owned_parent_action(
                    factory, session, 'start', start_request, remote=remote,
                    output=group_path / variant / 'parent-start',
                    startup_seconds=startup_seconds, stop_seconds=stop_seconds,
                    max_stdout_bytes=max_stdout_bytes,
                    max_stderr_bytes=max_stderr_bytes)
            except RemoteExecutionError:
                # An uncertain server must never be reused for another group.
                # Live parallel phases own one group and release this parent.
                candidate_failure = (runtime_group_failure(factory, groups[variant], variant=variant)
                    if factory is not None and selected_group is not None
                    else None)
                if candidate_failure is None:
                    raise
                variant_failures[variant] = candidate_failure
                continue
            if start_receipt.get('state') != 'ready' or start_receipt.get('model_endpoint') != MODEL_ENDPOINT:
                raise SidecarExecutionError('parent vLLM server did not become ready')
            stop_request = {key: start_receipt[key] for key in
                ('variant', 'model_source_digest', 'pid', 'process_start_ticks')}
            try:
                payload = relay.job('batch', variant, scheduler_document(
                    groups[variant], all_groups,
                    jps=None if rate == 'public' else hidden_jps[workload],
                    calibration=calibration, variants=variants), timeout=relay_limits['batch_timeout_seconds'][mode],
                    name=f'{variant}-batch')
                batch = payload.get('batch')
                try:
                    validate_batch_plan(batch, groups[variant], calibration=calibration,
                                        input_digest=input_digests[variant])
                except MeasurementError as error:
                    if variant in {'vanilla', 'reference'}:
                        raise TrustedMeasurementError(variant, group_name,
                            'invalid_or_incomplete_measurement') from error
                    raise
                if workload == 'swe':
                    # No correctness run starts while a timed testbed may still run.
                    _validate_shutdown(payload, groups[variant],
                        model_case_failures=(timed_case_failures(factory, groups[variant], variant=variant)
                            if factory is not None and
                               runtime_group_failure(factory, groups[variant], variant=variant) is None else frozenset()))
                batches[variant] = batch
                if factory is not None:
                    candidate_failure = runtime_group_failure(factory, groups[variant], variant=variant)
            except RemoteExecutionError:
                candidate_failure = (runtime_group_failure(factory, groups[variant], variant=variant)
                    if factory is not None else None)
                if candidate_failure is None:
                    raise
            finally:
                try:
                    stopped = _owned_parent_action(
                        factory, session, 'stop', stop_request, remote=remote,
                        output=group_path / variant / 'parent-stop',
                        startup_seconds=startup_seconds, stop_seconds=stop_seconds,
                        max_stdout_bytes=max_stdout_bytes,
                        max_stderr_bytes=max_stderr_bytes)
                    if stopped.get('state') != 'terminated' or stopped.get('remote_artifacts_removed') is not True:
                        raise SidecarExecutionError('parent vLLM cleanup remains uncertain')
                except RemoteExecutionError:
                    candidate_failure = (runtime_group_failure(factory, groups[variant], variant=variant)
                        if factory is not None and selected_group is not None
                        else None)
                    if candidate_failure is None:
                        raise
            if candidate_failure is not None:
                variant_failures[variant] = candidate_failure
        for variant in variants:
            candidate_failure = variant_failures.get(variant)
            if candidate_failure is not None:
                observations[variant][group_name] = candidate_failure
                continue
            if workload == 'swe':
                rows = run_swe_correctness(
                    batches[variant], groups[variant], relay=relay, variant=variant,
                    evidence=group_path / variant / 'correctness', calibration=calibration,
                    request_bytes=correctness_limits['request_bytes'],
                    timeout=relay_limits['correctness_timeout_seconds'][mode],
                    model_case_failures=(timed_case_failures(factory, groups[variant], variant=variant)
                        if factory is not None else frozenset()),
                    model_case_bounds=(timed_case_bounds(factory, groups[variant], variant=variant)
                        if factory is not None else {}))
            else:
                rows = project_complete_batch(
                    batches[variant], groups[variant], calibration=calibration,
                    variant=variant,
                    model_case_failures=(timed_case_failures(factory, groups[variant], variant=variant)
                        if factory is not None else frozenset()))
            observations[variant][group_name] = rows
    if selected_group is not None:
        if any(set(groups) != {selected_group} for groups in observations.values()):
            raise SidecarExecutionError('selected group did not return all three variants')
        (evidence / 'group-result.json').write_text(json.dumps({'group': selected_group, 'variants': list(variants),
            'score_ready': False, 'scope': 'operator-only group; complete join required'}) + '\n')
        return observations
    if variants != VARIANTS:
        raise SidecarExecutionError('candidate-only groups require the fixed-target parallel join')
    result = (evaluate_train if mode == 'train' else evaluate_final)(
        observations['vanilla'], observations['candidate'], observations['reference'])
    (evidence / 'operator-result.json').write_text(json.dumps({
        'mode': mode, 'calibration_only': calibration,
        'selected_patch_sha256': 'sha256:' + hashlib.sha256(candidate_patch).hexdigest(),
        'result': result,
        'scope': 'operator-only complete phase; native final acceptance remains separate',
    }, indent=2) + '\n')
    return result
