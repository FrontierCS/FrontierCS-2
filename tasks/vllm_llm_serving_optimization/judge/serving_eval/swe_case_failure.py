"""Closed case-local failure intent; caller must authenticate request identity."""
import math
import time

SCHEMA = 'frontier.vllm-swe-case-failure/v1'
STAGES = frozenset({'execute', 'patch', 'correctness', 'serialize', 'cleanup'})
REASONS = frozenset({'boundary_error', 'deadline', 'output_limit'})
FIELDS = frozenset({'schema', 'disposition', 'stage', 'reason', 'error_type'})


class CaseDeadlineExceeded(RuntimeError):
    pass


def validate_failure(value):
    if (not isinstance(value, dict) or set(value) != FIELDS
            or value['schema'] != SCHEMA or value['disposition'] != 'incorrect'
            or not isinstance(value['stage'], str) or value['stage'] not in STAGES
            or not isinstance(value['reason'], str) or value['reason'] not in REASONS
            or not isinstance(value['error_type'], str)
            or not 0 < len(value['error_type']) <= 128
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
                   for c in value['error_type'])):
        raise ValueError('invalid closed SWE case failure')
    return value


def failure(stage, error, *, reason='boundary_error'):
    name = type(error).__name__
    name = ''.join(c if c.isascii() and (c.isalnum() or c == '_') else '_'
                   for c in name[:128]) or 'UnknownError'
    return validate_failure({'schema': SCHEMA, 'disposition': 'incorrect',
                             'stage': stage, 'reason': reason, 'error_type': name})


def remaining(deadline):
    if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)):
        raise ValueError('finite SWE owner deadline required')
    value = deadline - time.monotonic()
    if value <= 0:
        raise CaseDeadlineExceeded('SWE request reached its original deadline')
    return value
