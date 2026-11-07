"""Closed scheduler terminal-failure payload, authenticated by its outer envelope."""
from sidecar_catalog import VARIANTS

SCHEMA = 'frontier.vllm-scheduler-terminal-failure/v1'
READINESS_CAUSES = frozenset({'connection', 'timeout', 'http_5xx'})


def validate(payload):
    if (not isinstance(payload, dict) or set(payload) != {'schema', 'variant', 'failure'}
            or payload['schema'] != SCHEMA or payload['variant'] not in VARIANTS):
        raise ValueError('invalid scheduler terminal failure envelope')
    failure = payload['failure']
    if (not isinstance(failure, dict) or set(failure) !=
            {'stage', 'cause', 'sidecar_id', 'candidate_input_forwarded'}
            or failure['stage'] not in {'preinput_sidecar_ready', 'scheduler_control'}
            or failure['cause'] not in READINESS_CAUSES | {'unclassified'}
            or (failure['stage'] == 'scheduler_control' and
                (failure['cause'] != 'unclassified' or failure['sidecar_id'] is not None
                 or failure['candidate_input_forwarded'] is not None))):
        raise ValueError('invalid closed scheduler terminal failure')
    if failure['stage'] == 'preinput_sidecar_ready':
        if failure['candidate_input_forwarded'] is not False or failure['cause'] not in READINESS_CAUSES:
            raise ValueError('readiness failure is only authoritative before input')
        from sidecar_catalog import SWE_SIDECAR_INDICES, testbed_name
        declared = {testbed_name(index, payload['variant']) for index in SWE_SIDECAR_INDICES}
        if failure['sidecar_id'] not in declared:
            raise ValueError('unknown declared testbed selector')
    return failure
