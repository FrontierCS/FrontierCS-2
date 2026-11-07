"""Project only CPU-owned C1 candidate failures through the existing scorer type."""
from instance_manifest import expected_groups
from reference_verdict import InstanceObservation


def timed_case_failures(factory, group, *, variant='candidate'):
    """Read C2 outcomes from the CPU's completed timed-batch ownership only."""
    if variant not in {'vanilla', 'reference', 'candidate'}:
        raise ValueError('known serving variant required')
    name = group.workload + '/' + group.rate
    ids = expected_groups(group.mode)[name]
    failures = set()
    for controller in factory.controllers:
        slots = controller.slots
        if (len(slots) != len(ids) or {slot.instance_id for slot in slots} != set(ids)
                or any(slot.variant != variant or slot.phase != group.mode
                       or slot.operation_id != group.operation_id
                       or slot.source_digest != group.model_source_digest or slot.group != name
                       for slot in slots)):
            continue
        for slot in slots:
            record = controller.ledger.read(slot)
            if record is None:
                raise ValueError('CPU timed batch has an unresolved original case')
            if record['status'] == 'complete':
                continue
            context = record.get('failure_context', {})
            decision = context.get('decision', {})
            if (context.get('boundary', {}).get('kind') != 'swe_case' or
                    decision.get('disposition') != 'case_fail'):
                raise ValueError('timed case projection requires C2-only CPU outcomes')
            failures.add(slot.instance_id)
    return frozenset(failures)


def timed_case_bounds(factory, group, *, variant):
    """Only C2 terminals may consume the CPU's pre-input watchdog allocation."""
    failed = timed_case_failures(factory, group, variant=variant)
    output = {}
    for controller in factory.controllers:
        for slot in controller.slots:
            if (slot.instance_id not in failed or slot.variant != variant
                    or slot.phase != group.mode or slot.operation_id != group.operation_id
                    or slot.source_digest != group.model_source_digest
                    or slot.group != group.workload + '/' + group.rate):
                continue
            record = controller.ledger.read(slot)
            if record is None or record['status'] == 'complete':
                continue
            bound = controller.timing_allocation.for_slot(slot)
            if record['cap_seconds'] != bound.bound_seconds:
                raise ValueError('C2 timing terminal differs from the pre-input CPU bound')
            output[slot.instance_id] = bound
    if set(output) != set(failed):
        raise ValueError('each failed timed model case requires a frozen CPU bound')
    return output


def runtime_group_failure(factory, group, *, variant='candidate'):
    if variant not in {'vanilla', 'reference', 'candidate'}:
        raise ValueError('known serving variant required')
    """A C1 terminal denies credit to the complete original workload group.

    The factory is the surviving CPU authority, never a decoded remote object.
    Baseline/reference convention is deliberately absent from this projection.
    """
    name = group.workload + '/' + group.rate
    ids = expected_groups(group.mode)[name]
    for controller in factory.controllers:
        slots = controller.slots
        if (len(slots) != len(ids) or {slot.instance_id for slot in slots} != set(ids)
                or any(slot.variant != variant or slot.phase != group.mode
                       or slot.operation_id != group.operation_id
                       or slot.source_digest != group.model_source_digest
                       or slot.group not in {name, name + '/startup', name + '/shutdown', name + '/group_control'}
                       for slot in slots)):
            continue
        for slot in slots:
            record = controller.ledger.read(slot)
            if record is None or record['status'] == 'complete':
                continue
            context = record.get('failure_context', {})
            decision = context.get('decision', {})
            if (context.get('boundary', {}).get('kind') == 'candidate_service'
                    and decision.get('attribution') == 'candidate'
                    and decision.get('disposition') == 'runtime_error'
                    and decision.get('affected_scope') == 'workload_group'
                    and decision.get('retry_eligible') is False):
                from cpu_reference_verdict import RuntimeGroupFailure
                return RuntimeGroupFailure(variant, name, tuple(ids))
    return None
