"""Answer-free SWE problem inputs for the trusted timed scheduler."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from instance_manifest import expected_groups

DATA = Path(__file__).with_name('swe_data') / 'runtime_inputs.json'
SOURCE_INSTANCES_SHA256 = '97f551b7340630436b8a0f573172fa745e0e01ace174f45bae7ac6ef880dcd9f'
RUNTIME_INPUTS_SHA256 = '4fdf8604dcf8a22c357c47e796ad1b1410e9daeba38e2bd4a830f4557f66287d'


class SWEInputError(ValueError):
    pass


def load_swe_inputs(mode):
    raw = DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RUNTIME_INPUTS_SHA256:
        raise SWEInputError('fixed SWE runtime input bytes changed')
    value = json.loads(raw)
    expected = expected_groups('eval')['swe/public']
    if (not isinstance(value, dict)
            or set(value) != {'schema', 'source_instances_sha256', 'instances'}
            or value['schema'] != 'frontier.vllm-swe-runtime-inputs/v1'
            or value['source_instances_sha256'] != SOURCE_INSTANCES_SHA256
            or not isinstance(value['instances'], list)
            or tuple(row.get('instance_id') for row in value['instances']) != expected
            or any(not isinstance(row, dict) or set(row) != {'instance_id', 'problem_statement'}
                   or not isinstance(row['problem_statement'], str) for row in value['instances'])):
        raise SWEInputError('fixed SWE runtime inputs differ from the frozen answer-free projection')
    selected = set(expected_groups(mode)['swe/public'])
    return [row for row in value['instances'] if row['instance_id'] in selected]


def verify_projection(source_path):
    raw = Path(source_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_INSTANCES_SHA256:
        raise SWEInputError('full SWE source data digest changed')
    source = json.loads(raw)
    projected = [{'instance_id': row['instance_id'], 'problem_statement': row['problem_statement']}
                 for row in source['instances']]
    if projected != load_swe_inputs('eval'):
        raise SWEInputError('answer-free SWE runtime projection changed')
    return True
