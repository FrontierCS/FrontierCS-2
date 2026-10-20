"""Trusted parent lifecycle for one patched vLLM server.

This module prepares and starts only the submitted serving process. Whole-batch
scheduling and result capture run in a separate fixed trusted sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

SOURCE_COMMIT = 'b8b302cde434df8c9289a2b465406b47ebab1c2d'
MODEL = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'


class VariantStartError(RuntimeError):
    """Carry bounded operator diagnostics across mandatory remote cleanup."""

    def __init__(self, record):
        super().__init__('variant server lifecycle failed')
        self.record = record


def source_digest(patch):
    return 'sha256:' + hashlib.sha256(
        (SOURCE_COMMIT + '\0' + hashlib.sha256(patch).hexdigest()).encode()).hexdigest()


def _process_start_ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _group_exists(pgid):
    for stat_path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = stat_path.read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) == pgid:
                return True
        except (FileNotFoundError, IndexError, ValueError, PermissionError):
            continue
    return False


def stop_group_id(pgid, start_ticks, stop_seconds):
    """Terminate only the exact process generation returned by trusted start."""
    observed = _process_start_ticks(pgid)
    if observed is None:
        # A process-group identifier cannot be reused while members of that
        # group remain. The leader may exit after spawning workers, so a
        # missing /proc/<pgid> is not itself a cleanup receipt.
        if not _group_exists(pgid):
            return {'already_exited': True, 'term_sent': False, 'kill_sent': False}
    elif observed != start_ticks:
        raise RuntimeError('server process identity changed before cleanup')
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        if _group_exists(pgid):
            raise RuntimeError('server process group changed during cleanup')
        return {'already_exited': True, 'term_sent': False, 'kill_sent': False}
    deadline = time.monotonic() + stop_seconds
    while _group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    killed = _group_exists(pgid)
    if killed:
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + stop_seconds
        while _group_exists(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _group_exists(pgid):
        raise RuntimeError('server process group survived bounded cleanup')
    return {'already_exited': False, 'term_sent': True, 'kill_sent': killed}


def stop_group(process, stop_seconds):
    """Stop a child and every member of its inherited process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=stop_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader can exit while workers in its process group remain.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError('variant startup deadline expired')
    return remaining


def _validate_start(document):
    if (not isinstance(document, dict)
            or set(document) != {'variant', 'patch', 'model_source_digest'}
            or document.get('variant') not in {'vanilla', 'reference', 'candidate'}
            or not isinstance(document.get('patch'), str)
            or len(document['patch'].encode()) > 1_500_000
            or not isinstance(document.get('model_source_digest'), str)):
        raise ValueError('invalid closed variant start request')
    patch = document['patch'].encode()
    if document['variant'] == 'vanilla' and patch:
        raise ValueError('vanilla may not include a patch')
    if source_digest(patch) != document['model_source_digest']:
        raise ValueError('variant source digest differs from its patch')
    return patch


def start(document, *, root, startup_seconds, stop_seconds, diagnostic_bytes):
    patch = _validate_start(document)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    process = None
    build = None
    started = time.monotonic()
    deadline = started + startup_seconds
    stage = 'prepare'
    try:
        patch_path = root / 'solution.patch'
        patch_path.write_bytes(patch)
        runtime = importlib.machinery.SourceFileLoader(
            'vllm_runtime', '/opt/frontier/vllm-runtime').load_module()
        lock = json.loads(runtime.ASSETS.read_text())
        with (root / 'setup.log').open('wb') as log:
            command = ['/opt/frontier/runtime-python', '/opt/frontier/vllm-runtime', 'prepare',
                       '--output', str(root / 'variant'), '--patch', str(patch_path)]
            build = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                if build.wait(timeout=_remaining(deadline)) != 0:
                    raise RuntimeError('offline variant preparation failed')
            finally:
                stop_group(build, stop_seconds)
        prepared = json.loads((root / 'variant/prepared.json').read_text())
        stage = 'engine_readiness'
        command = runtime.server_command(prepared['python'], runtime.MODEL_DIRECTORY, 8000)
        environment = runtime.offline_environment(
            runtime.WHEEL_DIRECTORY / lock['wheel']['filename'], lock['vllm_version'])
        log = (root / 'server.log').open('wb')
        try:
            process = subprocess.Popen(command, cwd=root / 'variant', env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            if process.poll() is not None:
                raise RuntimeError('variant engine exited before readiness')
            remaining = _remaining(deadline)
            try:
                with opener.open('http://127.0.0.1:8000/v1/models',
                                 timeout=min(1, remaining)) as response:
                    models = json.load(response)
                if not any(row.get('id') == MODEL for row in models.get('data', [])):
                    raise RuntimeError('wrong served model')
                break
            except Exception:
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        ticks = _process_start_ticks(process.pid)
        if ticks is None:
            raise RuntimeError('variant engine identity disappeared after readiness')
        receipt = {'schema': 'frontier.vllm-parent-server/v1', 'state': 'ready',
                   'variant': document['variant'],
                   'model_source_digest': document['model_source_digest'],
                   'pid': process.pid, 'process_start_ticks': ticks,
                   'setup_seconds': time.monotonic() - started,
                   'model_endpoint': 'http://main:8000/v1', 'score_ready': False}
        (root / 'start.json').write_text(json.dumps(receipt, separators=(',', ':')) + '\n')
        return receipt
    except BaseException as error:
        try:
            if process is not None:
                stop_group(process, stop_seconds)
            record = failure_diagnostics(root, stage, error, diagnostic_bytes,
                                         time.monotonic() - started)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        raise VariantStartError(record) from None


def diagnostics(root, byte_limit):
    records = {}
    remaining = byte_limit
    for name in ('server.log', 'setup.log'):
        path = Path(root) / name
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        size = path.stat().st_size
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(65536), b''):
                digest.update(chunk)
            stream.seek(max(0, size - remaining))
            tail = stream.read(remaining)
        remaining -= len(tail)
        records[name] = {'bytes': size, 'sha256': 'sha256:' + digest.hexdigest(),
                         'tail': tail.decode('utf-8', errors='replace'),
                         'tail_bytes': len(tail), 'truncated': len(tail) < size}
    return records


def failure_diagnostics(root, stage, error, byte_limit, elapsed_seconds):
    """Build a bounded failure record before the remote workspace is removed."""
    tail_limit = max(0, byte_limit // 4)
    error_message = str(error)
    while True:
        record = {'schema': 'frontier.vllm-parent-start-failure/v1',
                  'state': 'failed', 'stage': stage,
                  'error_type': type(error).__name__[:128],
                  'error_message': error_message,
                  'elapsed_seconds': elapsed_seconds,
                  'diagnostics': diagnostics(root, tail_limit), 'score_ready': False}
        encoded = json.dumps(record, allow_nan=False, separators=(',', ':')).encode()
        if len(encoded) <= byte_limit:
            return record
        if tail_limit:
            tail_limit //= 2
        elif error_message:
            error_message = error_message[:len(error_message) // 2]
        else:
            fallback = {'schema': 'frontier.vllm-parent-start-failure/v1',
                        'state': 'failed', 'stage': stage, 'score_ready': False}
            if len(json.dumps(fallback, separators=(',', ':')).encode()) > byte_limit:
                raise ValueError('diagnostic byte limit cannot hold failure record')
            return fallback


def _validate_stop(document):
    keys = {'variant', 'model_source_digest', 'pid', 'process_start_ticks'}
    if (not isinstance(document, dict) or set(document) != keys
            or document['variant'] not in {'vanilla', 'reference', 'candidate'}
            or not isinstance(document['model_source_digest'], str)
            or any(type(document[key]) is not int or document[key] <= 1
                   for key in ('pid', 'process_start_ticks'))):
        raise ValueError('invalid closed variant stop request')


def stop(document, *, root, stop_seconds, diagnostic_bytes):
    _validate_stop(document)
    root = Path(root)
    start_receipt = json.loads((root / 'start.json').read_text())
    for key in ('variant', 'model_source_digest', 'pid', 'process_start_ticks'):
        if start_receipt.get(key) != document[key]:
            raise RuntimeError('variant stop identity differs from trusted start')
    cleanup = stop_group_id(document['pid'], document['process_start_ticks'], stop_seconds)
    result = {'schema': 'frontier.vllm-parent-server/v1', 'state': 'terminated',
              'variant': document['variant'],
              'model_source_digest': document['model_source_digest'],
              'cleanup': cleanup, 'diagnostics': diagnostics(root, diagnostic_bytes),
              'score_ready': False}
    shutil.rmtree(root)
    result['remote_artifacts_removed'] = not root.exists()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--max-input-bytes', required=True, type=int)
    parser.add_argument('--max-output-bytes', required=True, type=int)
    parser.add_argument('--startup-seconds', required=True, type=int)
    parser.add_argument('--stop-seconds', required=True, type=int)
    parser.add_argument('--max-diagnostic-bytes', required=True, type=int)
    args = parser.parse_args()
    if min(args.max_input_bytes, args.max_output_bytes, args.startup_seconds,
           args.stop_seconds, args.max_diagnostic_bytes) <= 0:
        raise ValueError('explicit publisher limits required')
    raw = sys.stdin.buffer.read(args.max_input_bytes + 1)
    if len(raw) > args.max_input_bytes:
        raise ValueError('variant lifecycle request exceeds stream bound')
    document = json.loads(raw)
    result = (start(document, root=args.root, startup_seconds=args.startup_seconds,
                    stop_seconds=args.stop_seconds, diagnostic_bytes=args.max_diagnostic_bytes)
              if args.action == 'start' else
              stop(document, root=args.root, stop_seconds=args.stop_seconds,
                   diagnostic_bytes=args.max_diagnostic_bytes))
    encoded = json.dumps(result, allow_nan=False, separators=(',', ':')).encode()
    if len(encoded) > args.max_output_bytes:
        raise ValueError('variant lifecycle result exceeds stream bound')
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    try:
        main()
    except VariantStartError as error:
        sys.stderr.buffer.write(json.dumps(
            error.record, allow_nan=False, separators=(',', ':')).encode())
        raise SystemExit(1)
    except Exception:
        print('variant server lifecycle failed; no scientific result', file=sys.stderr)
        raise SystemExit(1)
