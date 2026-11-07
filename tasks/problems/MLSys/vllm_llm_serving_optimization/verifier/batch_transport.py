"""Descriptors of the declared testbeds and the one batch document the scheduler accepts.

Parent lifetime is owned by GPUSession; the Driver started every testbed with
the parent. Nothing here creates, polls or terminates a container.
"""
from dataclasses import asdict

from batch_coordinator import PreparedTestbed
from batch_run import SCHEMA
from driver_client import SidecarExecutionError
from instance_manifest import expected_groups
from sidecar_catalog import TESTBED_PORT, instance_index, testbed_name


def prepared_testbeds(mode, variant, sidecar_names):
    """One descriptor per original SWE instance of the phase, by declared name."""
    names = set(sidecar_names)
    beds = []
    for ident in expected_groups(mode)['swe/public']:
        name = testbed_name(instance_index(ident), variant)
        if name not in names:
            raise SidecarExecutionError('Driver view lacks a declared interactive testbed')
        beds.append(PreparedTestbed(ident, f'http://{name}:{TESTBED_PORT}', name, mode))
    return tuple(beds)


def scheduler_document(group, all_groups, *, jps=None, calibration=False, variants=None):
    """Build the only batch body accepted by the trusted scheduler."""
    from sidecar_catalog import VARIANTS, validate_variants
    variants = validate_variants(VARIANTS if variants is None else variants)
    if (group.identity() != {key: group.operation_id for key in
            ('operation_id', 'group_id', 'placement_id')}
            or not isinstance(all_groups, (list, tuple)) or len(all_groups) != len(variants)
            or group not in all_groups):
        raise SidecarExecutionError('scheduler batch does not bind one complete owned variant barrier')
    return {'prepared': {'schema': SCHEMA, 'group': asdict(group)},
            'all_prepared': [{'schema': SCHEMA, 'group': asdict(item)}
                             for item in all_groups],
            'jps': jps, 'calibration': calibration}
