"""Fixed names and addresses of the containers of a parent's pod; shared by publisher and runtime.

Pumpkin starts every companion that a judge's runner declares beside the GPU
parent, in this catalog's order, and each container reaches the others by
name. Nothing here chooses a budget: the publisher renders this catalog
into the runtime projection, the CPU Verifier starts exactly its services, and
the trusted scheduler derives its peers from it.
"""
from __future__ import annotations

import re

from instance_manifest import expected_groups

# Retain the original frozen sidecar IDs after removing two final-only cases.
SWE_SIDECAR_INDICES = tuple(index for index in range(50) if index not in (18, 39))
VARIANTS = ('vanilla', 'reference', 'candidate')
CANDIDATE_ONLY = ('candidate',)
SCHEDULER = 'scheduler'
SCHEDULER_PORT = 8002
TESTBED_PORT = 8001
CHECK_PORT = 8003
NAME = re.compile(r'^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$')
MAX_SIDECARS_PER_PARENT = 250  # Modal's concurrent sidecar cap per Sandbox.
# What a pod's process writes outside its testbed, it writes below here: a Pumpkin sandbox's /tmp
# may be read-only.
SCRATCH = '/var/tmp'
# Where a pod's containers reach the parent: Pumpkin names its own container
# `main`, and each companion by its name.
MAIN_ADDRESS = 'main'
# Where a service listens: every address, as each caller arrives over the pod's
# network, from the same address whoever it is, so a service authenticates the
# scheduler by its token (`serving_eval/sidecar_env.py`).
SERVICE_HOST = '0.0.0.0'


def instance_index(instance_id):
    """Position of an original SWE instance in the frozen final order."""
    ids = expected_groups('eval')['swe/public']
    if instance_id not in ids:
        raise ValueError('instance is not in the frozen SWE inventory')
    return SWE_SIDECAR_INDICES[ids.index(instance_id)]


def testbed_name(index, variant):
    if variant not in VARIANTS:
        raise ValueError('unknown serving variant')
    return f'swe-{index:03d}-{variant}'


def check_name(index):
    return f'swe-{index:03d}-check'


def image_role(index):
    return f'swe-{index:03d}'


def validate_variants(variants):
    if variants not in (VARIANTS, CANDIDATE_ONLY):
        raise ValueError('variant set differs from a fixed published evaluation policy')
    return variants


def expected_sidecars(mode, *, variants=VARIANTS):
    """Every sidecar name the Driver must list on a parent of this phase."""
    validate_variants(variants)
    ids = expected_groups(mode)['swe/public']
    names = [SCHEDULER]
    for ident in ids:
        index = instance_index(ident)
        names.extend(testbed_name(index, variant) for variant in variants)
    names.extend(check_name(instance_index(ident)) for ident in ids)
    if len(names) != len(set(names)) or len(names) > MAX_SIDECARS_PER_PARENT:
        raise ValueError('sidecar catalog exceeds the provider cap or repeats a name')
    return tuple(names)


def address(name, mode, *, variants=VARIANTS):
    """Where the pod's containers reach companion ``name`` of a phase's SWE parent."""
    if name not in expected_sidecars(mode, variants=variants):
        raise ValueError('companion is not in the fixed catalog of the phase')
    return name


def phases_for_instance(instance_id):
    """Public instances serve both phases; the rest exist only in the final."""
    public = expected_groups('train')['swe/public']
    return ['train', 'eval'] if instance_id in public else ['eval']


WORKER_PYTHON = '/opt/frontier/worker-python/bin/python'
SCHEDULER_PYTHON = '/opt/frontier/python/bin/python'
IDENTITY_FILE = SCRATCH + '/frontier-identity.json'
LIMIT_KEYS = frozenset({'scheduler_request_bytes', 'scheduler_response_bytes', 'tool_request_bytes',
                        'tool_output_bytes', 'worker_request_bytes', 'worker_output_bytes'})


def render_profiles(limits, *, variants=VARIANTS):
    """The complete declared catalog as policy profiles (`image` names the role).

    Every command is fixed here: the publisher only supplies byte bounds. The
    runtime projection re-validates the rendered arguments.
    """
    validate_variants(variants)
    if not isinstance(limits, dict) or set(limits) != LIMIT_KEYS or any(
            type(value) is not int or value <= 0 for value in limits.values()):
        raise ValueError('every sidecar byte bound must be an explicit positive integer')
    # The scheduler listens on its own loopback alone: only the relay, a command in its own
    # container, reaches it.
    profiles = {SCHEDULER: {
        'image': SCHEDULER,
        'args': [SCHEDULER_PYTHON, '/opt/frontier/judge/scheduler_service.py', '--host', '127.0.0.1',
                 '--port', str(SCHEDULER_PORT), '--max-request-bytes', str(limits['scheduler_request_bytes']),
                 '--max-response-bytes', str(limits['scheduler_response_bytes'])],
        'workdir': SCRATCH, 'phases': ['train', 'eval']}}
    if variants == CANDIDATE_ONLY:
        profiles[SCHEDULER]['args'].extend(['--variants', 'candidate'])
    for ident in expected_groups('eval')['swe/public']:
        # Isolate trusted worker imports; child test interpreters keep their original flags.
        index = instance_index(ident)
        phases = phases_for_instance(ident)
        for variant in variants:
            profiles[testbed_name(index, variant)] = {
                'image': image_role(index),
                'args': [WORKER_PYTHON, '-P', '-m', 'serving_eval.swe_tool_service', '--driver-identity-file',
                         IDENTITY_FILE, '--instance-id', ident, '--listen-profile', 'bridge',
                         '--host', SERVICE_HOST, '--port', str(TESTBED_PORT),
                         '--max-request-bytes', str(limits['tool_request_bytes']),
                         '--max-output-bytes', str(limits['tool_output_bytes'])],
                'workdir': '/testbed', 'phases': list(phases)}
        profiles[check_name(index)] = {
            'image': image_role(index),
            'args': [WORKER_PYTHON, '-P', '-m', 'serving_eval.swe_correctness_service', '--driver-identity-file',
                     IDENTITY_FILE, '--instance-id', ident, '--host', SERVICE_HOST, '--port', str(CHECK_PORT),
                     '--max-request-bytes', str(limits['worker_request_bytes']),
                     '--max-output-bytes', str(limits['worker_output_bytes'])],
            'workdir': '/testbed', 'phases': list(phases)}
        if variants == CANDIDATE_ONLY:
            profiles[check_name(index)]['args'].extend(['--variants', 'candidate'])
    if len(profiles) != 1 + (len(variants)+1) * len(expected_groups('eval')['swe/public']):
        raise ValueError('rendered catalog is incomplete')
    return profiles
