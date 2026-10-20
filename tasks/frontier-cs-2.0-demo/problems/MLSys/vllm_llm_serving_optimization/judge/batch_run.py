"""Run one Driver-prepared local group and retain operator evidence on disk.

This command provisions nothing and emits no score. Its caller owns the fixed
images, network boundary, process lifetime and authenticity of collected files.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

from batch_coordinator import BackendError, PreparedGroup, PreparedTestbed, _endpoint, run_group


SCHEMA = 'frontier.vllm-prepared-batch/v1'


def prepared_group(document):
    if not isinstance(document, dict) or set(document) != {'schema', 'group'} or document['schema'] != SCHEMA:
        raise BackendError('invalid prepared batch document')
    raw = document['group']
    required = {field.name for field in fields(PreparedGroup)}
    if not isinstance(raw, dict) or set(raw) != required:
        raise BackendError('prepared group fields differ from the fixed interface')
    if any(not isinstance(raw[key], str) or not raw[key] for key in required - {'testbeds'}):
        raise BackendError('prepared group identities must be nonempty strings')
    if raw['mode'] not in {'train', 'eval'} or raw['workload'] not in {'swe', 'bfcl'} or raw['rate'] not in {'public', 'hidden'}:
        raise BackendError('invalid prepared group phase or workload')
    _endpoint(raw['model_endpoint'])
    if not isinstance(raw['testbeds'], list):
        raise BackendError('prepared testbeds must be an ordered list')
    beds = []
    for row in raw['testbeds']:
        if (not isinstance(row, dict) or set(row) != {field.name for field in fields(PreparedTestbed)}
                or any(not isinstance(value, str) or not value for value in row.values())):
            raise BackendError('invalid prepared testbed descriptor')
        _endpoint(row['endpoint'])
        beds.append(PreparedTestbed(**row))
    return PreparedGroup(**{**raw, 'testbeds': tuple(beds)})


def execute(prepared, output, *, jps=None, calibration=False):
    raw = Path(prepared).read_bytes()
    group = prepared_group(json.loads(raw))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'prepared.json').write_bytes(raw)
    manifest = {
        'schema': 'frontier.vllm-batch-invocation/v1',
        'prepared_sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
        'identity': group.identity(), 'calibration_only': calibration,
        'score_ready': False,
    }
    (output / 'invocation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        result = run_group(group, jps=jps, calibration=calibration)
        encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
        pending = output / 'batch.pending.json'
        pending.write_text(encoded)
        pending.replace(output / 'batch.json')
    except Exception as exc:
        # Retain an execution failure without promoting it to a candidate zero.
        (output / 'error.json').write_text(json.dumps({
            'identity': group.identity(), 'error_type': type(exc).__name__,
            'origin': 'batch_setup_or_runtime_unclassified', 'score_ready': False,
        }, indent=2) + '\n')
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New operator evidence directory')
    parser.add_argument('--jps', type=float)
    parser.add_argument('--calibration', action='store_true')
    args = parser.parse_args(argv)
    execute(args.prepared, args.output, jps=args.jps, calibration=args.calibration)


if __name__ == '__main__':
    main()
