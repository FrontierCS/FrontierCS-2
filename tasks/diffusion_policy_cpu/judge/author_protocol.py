"""Reproducible author-only invocation marker; never a solver scoring payload."""
import argparse
import hashlib
import json
from pathlib import Path

from native_data import strict_json
from measurement_contract import validate_suite, validate_final_extension


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(body):
    return hashlib.sha256(body).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def invocation(dataset):
    require(0 < len(dataset) <= 65536, 'Author dataset exceeds bound')
    spec = strict_json(dataset)
    require(isinstance(spec, dict) and set(spec) == {
        'schema', 'purpose', 'mode', 'suites', 'maximum_seconds'}, 'Invalid author specification')
    require(spec['schema'] == 'frontier.diffusion-author-calibration/v1'
            and spec['purpose'] == 'fresh_baseline_and_raw_reference'
            and spec['mode'] == 'eval', 'Wrong author purpose')
    require(type(spec['maximum_seconds']) is int and 300 <= spec['maximum_seconds'] <= 3600,
            'Invalid finite author deadline')
    require(isinstance(spec['suites'], dict) and set(spec['suites']) == {'dev', 'final'},
            'Both private suites are required')
    for mode, suite in spec['suites'].items():
        validate_suite(suite, mode)
    validate_final_extension(spec['suites']['dev'], spec['suites']['final'])
    return {'schema': 'frontier.diffusion-author-invocation/v1',
            'purpose': 'fresh_baseline_and_raw_reference', 'mode': 'eval',
            'dataset_sha256': sha(dataset),
            'suite_sha256': {mode: sha(canonical(suite)) for mode, suite in spec['suites'].items()},
            'resources': {'cpus': 2, 'memory_bytes': 4 * 1024**3,
                          'maximum_seconds': spec['maximum_seconds']},
            'measurement': {'baseline_steps': 20, 'reference_steps': 100,
                            'baseline_trajectories': 18, 'reference_trajectories': 18}}


def validate_payload(payload, dataset):
    require(0 < len(payload) <= 16384, 'Author payload exceeds bound')
    expected = invocation(dataset)
    require(strict_json(payload) == expected and payload == canonical(expected),
            'Author payload differs from canonical dataset-bound invocation')
    return strict_json(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    payload = canonical(invocation(raw))
    validate_payload(payload, raw)
    with args.output.open('xb') as stream:
        stream.write(payload)
    args.output.chmod(0o400)


if __name__ == '__main__':
    main()
