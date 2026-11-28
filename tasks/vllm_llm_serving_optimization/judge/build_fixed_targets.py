"""Offline materializer from a root-reviewed plan; never reads the network.

Input approval and original evidence authentication are external gates. This tool
checks their exact-byte bindings, not the authority of a locally supplied plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fixed_targets import build_targets, canonical_bytes, digest, _require, _sha


def read_pinned(path, sha256):
    _require(_sha(sha256), 'input requires SHA256')
    raw = Path(path).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == sha256, 'offline input SHA256 mismatch')
    return json.loads(raw)


def materialize(plan, root):
    _require(isinstance(plan, dict) and set(plan) == {
        'schema', 'captures', 'environment'},
        'invalid fixed-target build plan')
    _require(plan['schema'] == 'frontier.vllm-fixed-target-build-plan/v2', 'wrong build-plan schema')
    _require(isinstance(plan['captures'], list) and 3 <= len(plan['captures']) <= 6,
             'reviewed build plan requires three captures per phase')
    captures = []
    for item in plan['captures']:
        _require(isinstance(item, dict) and set(item) == {
            'path', 'sha256', 'receipt_paths'}, 'invalid capture plan')
        capture = read_pinned(root/item['path'], item['sha256'])
        _require(isinstance(item['receipt_paths'], dict)
                 and set(item['receipt_paths']) == set(capture['phases']),
                 'receipt paths must match actual authenticated capture phases')
        for mode in capture['phases']:
            receipt = read_pinned(root/item['receipt_paths'][mode],
                capture['phases'][mode]['consumer_receipt_sha256'])
            _require(receipt.get('schema') == 'frontier.vllm-fixed-calibration-consumer-receipt/v1'
                     and receipt.get('status') == 'PASS'
                     and receipt.get('run_id') == capture['run_id']
                     and receipt.get('mode') == mode
                     and receipt.get('evidence_sha256') == capture['phases'][mode]['evidence_sha256']
                     and receipt.get('environment_sha256') == digest(capture['environment'])
                     and receipt.get('environment_provenance_sha256') == digest(capture['environment_provenance'])
                     and receipt.get('normalized_groups_sha256') == hashlib.sha256(
                         canonical_bytes(capture['phases'][mode]['groups'])).hexdigest(),
                     'accepted-consumer receipt does not bind the normalized phase groups')
        captures.append(capture)
    return build_targets(captures, environment=plan['environment'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True,
                        help='exact plan hash reviewed by root; never derive from candidate data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read_pinned(args.plan, args.plan_sha256)
    targets = materialize(plan, args.plan.resolve().parent)
    raw = canonical_bytes(targets)
    # Exclusive creation keeps a previous release target immutable.
    with args.output.open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'output': str(args.output), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw), 'status': 'REQUIRES_ROOT_RELEASE_REVIEW'}))


if __name__ == '__main__':
    main()
