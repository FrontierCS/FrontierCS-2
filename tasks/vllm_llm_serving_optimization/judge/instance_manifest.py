"""Frozen original instance identities, independent of measurement-supplied IDs."""
from __future__ import annotations

import json
from pathlib import Path


def expected_groups(mode: str) -> dict[str, tuple[str, ...]]:
    if mode not in {'train', 'eval'}:
        raise ValueError('unknown evaluation mode')
    manifest = json.loads(Path(__file__).with_suffix('.json').read_text())
    if manifest.get('schema') != 'frontier.vllm-instance-manifest/v1':
        raise ValueError('invalid instance manifest schema')
    sizes = {'train': {'swe': 5, 'bfcl': 8}, 'eval': {'swe': 48, 'bfcl': 155}}
    for phase, workloads in sizes.items():
        for workload, count in workloads.items():
            ids = manifest[phase][workload]
            if len(ids) != count or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != count:
                raise ValueError('instance manifest differs from the original fixed case counts')
    for workload in ('swe', 'bfcl'):
        public = manifest['train'][workload]
        if manifest['eval'][workload][:len(public)] != public:
            raise ValueError('final must preserve the original public prefix')
    rates = ('public',) if mode == 'train' else ('public', 'hidden')
    return {f'{workload}/{rate}': tuple(manifest[mode][workload]) for workload in ('swe', 'bfcl') for rate in rates}
