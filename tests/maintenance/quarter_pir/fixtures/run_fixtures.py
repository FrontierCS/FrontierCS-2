#!/usr/bin/env python3
"""Run every fixture through the real evaluator and compare with expected.json.

Usage (inside the verifier image, FRONTIER_PRIVATE_STATE set to a writable
directory outside the task): run_fixtures.py REPORT.json [name-substring ...]
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import evaluator  # noqa: E402


def outcome(metrics):
    rejection = metrics.get('private_rejection')
    if rejection:
        return rejection['category']
    if metrics['beats_reference']:
        return 'valid_beating'
    rows = metrics['beat_spec_workload_passes']
    passes = lambda name: all(row['metrics'][name] for row in rows if name in row['metrics'])
    fewer, within = passes('online_communication_bytes'), passes('client_storage_bytes')
    if fewer:
        return 'valid_fewer_bytes_within_storage' if within else 'valid_fewer_bytes_over_storage'
    if not within:
        return 'valid_over_storage'
    if all(e['transcript_identical'] for e in metrics['private_experiment_measurements']):
        return 'valid_identical_transcript'
    return 'valid_not_beating'


def accepts(expect, got):
    if expect == 'valid_not_beating':
        return got in ('valid_not_beating', 'valid_identical_transcript')
    if expect == 'valid_fewer_bytes_within_storage':
        return got in ('valid_fewer_bytes_within_storage', 'valid_beating')
    return expect == got


def main():
    report_path, filters = Path(sys.argv[1]), sys.argv[2:]
    expected = json.loads((HERE / 'expected.json').read_text())['fixtures']
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name in sorted(expected, key=lambda item: (item.startswith('positive/'), item)):
        if filters and not any(part in name for part in filters):
            continue
        started = time.monotonic()
        score, _, message, metrics = evaluator.evaluate(str(HERE / name))
        got = outcome(metrics)
        gate = {}
        for row in metrics['beat_spec_workload_passes']:
            for key, value in row['metrics'].items():
                gate.setdefault(key, [0, 0])
                gate[key][0] += bool(value)
                gate[key][1] += 1
        report[name] = {
            'expected': expected[name]['expect'], 'observed': got, 'ok': accepts(expected[name]['expect'], got),
            'message': message, 'score': score, 'beats_reference': metrics['beats_reference'],
            'validity_code': metrics.get('candidate_public_validity_code'), 'gate': gate,
            'wall_seconds': round(time.monotonic() - started, 1),
            'experiments': [{
                'size': e['size'], 'transcript_identical': e['transcript_identical'],
                **{f'{side}_{key}': e[side][key] for side in ('candidate', 'reference')
                   for key in ('offline_elapsed_ns', 'online_elapsed_ns', 'client_storage_bytes',
                               'upload_bytes_per_query', 'download_bytes_per_query')}}
                for e in metrics.get('private_experiment_measurements', [])],
        }
        report_path.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(name, got, 'OK' if report[name]['ok'] else 'MISMATCH', report[name]['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
