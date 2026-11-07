"""Validate scheduler-owned failed-case timing, never child-declared authority."""
from dataclasses import asdict
import math

from terminal_contract import validate_record


def measured_case_failure_seconds(row, *, slot):
    """Caller verifies scheduler attestation and supplies its CPU-owned exact slot."""
    record = row.get('supervised_terminal')
    validate_record(record)
    decision = record.get('failure_context', {}).get('decision', {})
    boundary = record.get('failure_context', {}).get('boundary', {})
    if (record['slot'] != asdict(slot) or row.get('instance_id') != slot.instance_id
            or not slot.group.startswith('swe/') or record['status'] == 'complete'
            or record.get('candidate_input_dispatched') is not True
            or boundary.get('kind') != 'swe_case'
            or decision.get('attribution') != 'candidate'
            or decision.get('disposition') != 'case_fail'
            or decision.get('affected_scope') != 'case'
            or decision.get('retry_eligible') is not False):
        raise ValueError('supervised case timing does not match the owned failed SWE case')
    elapsed = record['elapsed_seconds']
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('supervised failed case lacks positive actual elapsed time')
    return elapsed
