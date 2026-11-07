"""Local correctness backend for one already prepared fixed SWE testbed.

This module creates no container, server, namespace, or provider resource. The
trusted invoker supplies a fresh testbed and output bound; transport and process
isolation are integration requirements, not established by a local receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time

from .accuracy import (CorrectnessRuntimeError, OVERSIZE_REQUEST_SCHEMA, _case, _harness,
                       build_test_request, validate_oversize_request)


@dataclass
class Execution:
    exit_code: int
    output: bytes
    stages: bytes
    elapsed_seconds: float
    stop_reason: str | None = None
    script_sha256: str | None = None


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(argv, *, cwd, timeout, output_limit, stdin=b'', script=None, command_prefix=(), temporary_directory=None):
    """Bound both output streams and terminate the command's process group."""
    if timeout <= 0 or output_limit <= 0:
        raise CorrectnessRuntimeError('worker command has no funded time/output allowance')
    started = time.monotonic()
    output, stages = bytearray(), bytearray()
    stage_read, stage_write = os.pipe()
    process = None
    script_sha256 = None
    try:
        with tempfile.TemporaryFile() as input_file, tempfile.TemporaryDirectory(prefix='swe-script-', dir=temporary_directory) as temporary:
            input_file.write(stdin)
            input_file.seek(0)
            if script is not None:
                path = Path(temporary) / 'eval.sh'
                instrumented = script(stage_write)
                script_sha256 = _digest(instrumented.encode())
                path.write_text(instrumented)
                if command_prefix:
                    Path(temporary).chmod(0o755)
                    path.chmod(0o444)
                argv = ['/bin/bash', str(path)]
            process = subprocess.Popen([*command_prefix, *argv], cwd=cwd, stdin=input_file, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(stage_write,))
            os.close(stage_write)
            stage_write = -1
            stop_reason = None
            stopped_at = None
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, output)
                selector.register(stage_read, selectors.EVENT_READ, stages)
                while selector.get_map() or process.poll() is None:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0 and stop_reason is None:
                        stop_reason = 'timeout'
                        stopped_at = time.monotonic()
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    for key, _ in selector.select(max(0, min(remaining, 0.1)) if stop_reason is None else 0.1):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        available = output_limit - len(output) - len(stages)
                        key.data.extend(chunk[:max(0, available)])
                        if len(chunk) > available and stop_reason is None:
                            stop_reason = 'output_limit'
                            stopped_at = time.monotonic()
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    # A descendant may close over a pipe after the main process
                    # dies. The caller still owns full testbed teardown.
                    if stop_reason is not None and time.monotonic() - stopped_at > 1:
                        break
            exit_code = process.wait(timeout=1)
            process.stdout.close()
            return Execution(exit_code, bytes(output), bytes(stages), time.monotonic() - started, stop_reason, script_sha256)
    finally:
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=1)
            if process.stdout is not None:
                process.stdout.close()
        os.close(stage_read)
        if stage_write != -1:
            os.close(stage_write)


def _tree(raw):
    result = {}
    try:
        for entry in raw.split(b'\0'):
            if not entry:
                continue
            metadata, path = entry.split(b'\t', 1)
            mode, kind, blob = metadata.split()
            if not ((kind == b'blob' and mode in (b'100644', b'100755', b'120000'))
                    or (kind == b'commit' and mode == b'160000')):
                raise CorrectnessRuntimeError('unsupported object in prepared SWE source')
            name = os.fsdecode(path)
            if (name in result or not name or Path(name).is_absolute()
                    or any(part in ('.', '..', '.git') for part in name.split('/'))
                    or len(blob) != 40 or any(c not in b'0123456789abcdef' for c in blob)):
                raise CorrectnessRuntimeError('invalid prepared SWE source path or object')
            result[name] = (mode.decode(), blob.decode())
    except (ValueError, UnicodeError) as exc:
        raise CorrectnessRuntimeError('invalid prepared SWE source tree') from exc
    return result


def _same_mode(left, right):
    return left == right or {left, right} <= {'100644', '100755'}


def inspect_testbed(testbed: Path, base_commit: str, output_limit: int, *, execute=None):
    execute = _run if execute is None else execute
    testbed = Path(testbed).resolve()
    metadata_root = testbed / '.git'
    if not metadata_root.is_dir() or metadata_root.is_symlink():
        raise CorrectnessRuntimeError('prepared SWE Git metadata is not a local directory')
    verified, submodules, transformations = [], [], []
    visited = set()
    # Commands never inherit Git overrides, global attributes, replace refs or
    # fsmonitor hooks. Attribute/status commands use private metadata below.
    environment = ['env', '-i', 'PATH=/usr/bin:/bin', 'GIT_CONFIG_NOSYSTEM=1',
                   'GIT_CONFIG_GLOBAL=/dev/null', 'GIT_CONFIG_SYSTEM=/dev/null',
                   'GIT_ATTR_NOSYSTEM=1', 'GIT_NO_REPLACE_OBJECTS=1',
                   'GIT_OPTIONAL_LOCKS=0']
    options = ['-c', 'core.fileMode=false', '-c', 'core.fsmonitor=false',
               '-c', 'core.untrackedCache=false', '-c', 'core.attributesFile=/dev/null',
               '-c', 'core.excludesFile=/dev/null']

    def run(repo, arguments, *, stdin=b'', objects=None):
        prefix = environment + ([] if objects is None else ['GIT_OBJECT_DIRECTORY=' + str(objects)])
        result = execute(prefix + ['git', '-c', 'safe.directory=' + str(repo)] + options + arguments,
                         cwd=repo, timeout=1200, output_limit=output_limit, stdin=stdin)
        if result.exit_code or result.stop_reason:
            raise CorrectnessRuntimeError('prepared SWE Git identity could not be read')
        return result.output

    def same_tree(left, right):
        return left.keys() == right.keys() and all(left[p][1] == right[p][1]
                and _same_mode(left[p][0], right[p][0]) for p in left)

    def actual_blob(repo, name, mode, *, materialize=False):
        path = repo / name
        parent = path.parent
        while parent != repo:
            if parent.is_symlink():
                raise CorrectnessRuntimeError('prepared SWE file ancestor is a symlink')
            parent = parent.parent
        if not path.parent.resolve().is_relative_to(repo):
            raise CorrectnessRuntimeError('prepared SWE file location differs')
        if mode == '120000':
            if not path.is_symlink():
                raise CorrectnessRuntimeError('prepared SWE symlink type differs')
            data = os.fsencode(os.readlink(path))
            return data if materialize else blob_id(data)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repo):
            raise CorrectnessRuntimeError('prepared SWE file type or location differs')
        if materialize:
            return path.read_bytes()
        digest = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def blob_id(data):
        return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()

    def inspect(repo, ref, depth=0):
        if depth > 16 or repo in visited or not repo.is_dir() or repo.is_symlink():
            raise CorrectnessRuntimeError('invalid prepared SWE submodule location or depth')
        visited.add(repo)
        top = Path(os.fsdecode(run(repo, ['rev-parse', '--show-toplevel']).strip())).resolve()
        gitdir = Path(os.fsdecode(run(repo, ['rev-parse', '--absolute-git-dir']).strip())).resolve()
        if top != repo or not gitdir.is_relative_to(metadata_root):
            raise CorrectnessRuntimeError('prepared SWE repository location differs')
        objects = Path(os.fsdecode(run(repo, ['rev-parse', '--git-path', 'objects']).strip()))
        objects = (repo / objects).resolve() if not objects.is_absolute() else objects.resolve()
        if not objects.is_dir() or not objects.is_relative_to(metadata_root):
            raise CorrectnessRuntimeError('prepared SWE object store location differs')
        head = run(repo, ['rev-parse', 'HEAD']).decode().strip()
        if depth and head != ref:
            raise CorrectnessRuntimeError('prepared SWE submodule HEAD differs from fixed commit')
        base = _tree(run(repo, ['ls-tree', '-r', '-z', ref]))
        current = _tree(run(repo, ['ls-tree', '-r', '-z', 'HEAD']))
        if not same_tree(base, current):
            raise CorrectnessRuntimeError('prepared SWE HEAD differs from the fixed base content')
        index = {}
        try:
            for entry in run(repo, ['ls-files', '--stage', '-z']).split(b'\0'):
                if entry:
                    meta, path = entry.split(b'\t', 1)
                    mode, obj, stage = meta.split()
                    if stage != b'0':
                        raise ValueError('unmerged index')
                    name = os.fsdecode(path)
                    if name in index:
                        raise ValueError('duplicate index entry')
                    index[name] = (mode.decode(), obj.decode())
        except (ValueError, UnicodeError) as exc:
            raise CorrectnessRuntimeError('prepared SWE index is not fresh') from exc
        if not same_tree(base, index):
            raise CorrectnessRuntimeError('prepared SWE index differs from fixed source')
        # Attribute files are verified before any rule is interpreted. Live
        # .git/info/attributes and repository/global config are never consulted.
        for name, (mode, expected) in base.items():
            if Path(name).name == '.gitattributes':
                if mode not in ('100644', '100755') or actual_blob(repo, name, mode) != expected:
                    raise CorrectnessRuntimeError('prepared SWE attributes differ from fixed source')
        with tempfile.TemporaryDirectory(prefix='swe-fixed-attributes-') as temporary:
            private = Path(temporary) / 'git'
            run(repo, ['init', '--bare', '--quiet', '--template=', str(private)])
            isolated = ['--git-dir=' + str(private), '--work-tree=' + str(repo)]
            run(repo, isolated + ['read-tree', ref], objects=objects)
            run(repo, isolated + ['update-ref', 'HEAD', ref], objects=objects)
            for name, (mode, expected) in base.items():
                if mode == '160000':
                    child = repo / name
                    if child.is_symlink() or not child.resolve().is_relative_to(repo):
                        raise CorrectnessRuntimeError('prepared SWE submodule path escapes source')
                    parent = child.parent
                    while parent != repo:
                        if parent.is_symlink():
                            raise CorrectnessRuntimeError('prepared SWE submodule ancestor is a symlink')
                        parent = parent.parent
                    inspect(child, expected, depth + 1)
                    continue
                if actual_blob(repo, name, mode) == expected:
                    continue
                attributes = run(repo, isolated + ['check-attr', '--cached', '-z', '--stdin',
                                 'eol', 'text', 'filter', 'working-tree-encoding', 'ident'],
                                 stdin=os.fsencode(name) + b'\0', objects=objects).split(b'\0')
                attrs = {}
                try:
                    if len(attributes) != 16 or attributes[-1] != b'':
                        raise ValueError('attribute response length mismatch')
                    for offset in range(0, len(attributes) - 1, 3):
                        path, key, value = attributes[offset:offset + 3]
                        if os.fsdecode(path) != name:
                            raise ValueError('attribute path mismatch')
                        attrs[key.decode()] = value.decode()
                    if set(attrs) != {'eol', 'text', 'filter', 'working-tree-encoding', 'ident'}:
                        raise ValueError('attribute response keys mismatch')
                except (ValueError, UnicodeError) as exc:
                    raise CorrectnessRuntimeError('invalid fixed SWE attributes') from exc
                if (mode == '120000' or attrs.get('eol') != 'crlf'
                        or attrs.get('text') == 'unset'
                        or any(attrs.get(key) not in ('unspecified', 'unset')
                               for key in ('filter', 'working-tree-encoding', 'ident'))):
                    raise CorrectnessRuntimeError('prepared SWE file content differs from fixed base')
                original = run(repo, ['cat-file', 'blob', expected])
                expected_checkout = original.replace(b'\n', b'\r\n')
                if (b'\r' in original or b'\0' in original
                        or (repo / name).stat().st_size != len(expected_checkout)):
                    raise CorrectnessRuntimeError('prepared SWE file content differs from fixed base')
                data = actual_blob(repo, name, mode, materialize=True)
                if data != expected_checkout:
                    raise CorrectnessRuntimeError('prepared SWE file content differs from fixed base')
                transformations.append({'path': str((repo / name).relative_to(testbed)),
                                        'blob_id': expected, 'representation': 'declared-eol-crlf',
                                        'checkout_sha256': _digest(data), 'checkout_bytes': len(data)})
            # Ignore submodules only here: each child was independently checked
            # above. This also prevents Git spawning a child's mutable config.
            if run(repo, isolated + ['status', '--porcelain', '--untracked-files=all',
                                    '--ignore-submodules=all'], objects=objects):
                raise CorrectnessRuntimeError('prepared SWE worktree is not fresh')
        record = {'path': str(repo.relative_to(testbed)), 'base_commit': ref, 'observed_head': head,
                  'tracked_files_verified': len(base), 'regular_executable_mode_differences':
                  sum(base[name][0] != current[name][0] for name in base)}
        verified.append(record)
        if depth:
            submodules.append(record)
        return record

    root = inspect(testbed, base_commit)
    return {'base_commit': base_commit, 'observed_head': root['observed_head'],
            'tracked_files_verified': root['tracked_files_verified'],
            'regular_executable_mode_differences': root['regular_executable_mode_differences'],
            'content_matches_fixed_base': True, 'fresh_worktree': True,
            'submodules_verified': submodules, 'declared_checkout_representations': transformations}


def instrument_script(request, row):
    """Keep original commands, observe stages and stop on failed setup."""
    _, specs, make_script = _harness()
    commands = make_script(row, specs[row['repo']][row['version']], 'testbed', '/testbed',
                           row['base_commit'], row['test_patch'])
    canonical = '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + commands) + '\n'
    if canonical != request['eval_script']:
        raise CorrectnessRuntimeError('canonical harness script changed')
    from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
    if len(commands) < 6 or commands[-4] != f": '{START_TEST_OUTPUT}'" or commands[-2] != f": '{END_TEST_OUTPUT}'" or commands[-6] != commands[-1]:
        raise CorrectnessRuntimeError('unsupported original harness stage layout')
    if not commands[-5].startswith("git apply -v - <<"):
        raise CorrectnessRuntimeError('missing original gold-test patch command')

    def wrapper(descriptor):
        observed = []
        for index, command in enumerate(commands):
            observed.append(command)
            if index < len(commands) - 6:
                observed.append(f"frontier_setup_status=$?; printf 'setup_{index}=%s\\n' \"$frontier_setup_status\" >&{descriptor}; if [ \"$frontier_setup_status\" -ne 0 ]; then exit \"$frontier_setup_status\"; fi")
            label = {len(commands) - 6: 'test_reset', len(commands) - 5: 'test_patch',
                     len(commands) - 3: 'tests', len(commands) - 1: 'test_cleanup'}.get(index)
            if label:
                observed.append(f"printf '{label}=%s\\n' \"$?\" >&{descriptor}")
        return '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + observed) + '\n'
    return wrapper


def _execution_receipt(result):
    return {'exit_code': result.exit_code, 'elapsed_seconds': result.elapsed_seconds,
            'stop_reason': result.stop_reason, 'output_bytes_retained': len(result.output),
            'output_sha256': _digest(result.output), 'stage_bytes_retained': len(result.stages),
            'stage_sha256': _digest(result.stages), 'executed_script_sha256': result.script_sha256}


class PreparedSWEWorker:
    """One-shot local backend. The invoker owns freshness, isolation and teardown."""
    def __init__(self, *, instance_id, testbed=Path('/testbed'), max_output_bytes, execute=None, inspect_execute=None,
                 testbed_python='/opt/miniconda3/envs/testbed/bin/python'):
        _case(instance_id)
        if not isinstance(max_output_bytes, int) or isinstance(max_output_bytes, bool) or max_output_bytes <= 0:
            raise ValueError('a positive externally funded output bound is required')
        self.instance_id, self.testbed = instance_id, Path(testbed).resolve()
        self.max_output_bytes, self.used = max_output_bytes, False
        self.execute = _run if execute is None else execute
        self.inspect_execute = _run if inspect_execute is None else inspect_execute
        self.testbed_python = testbed_python

    def _bounded(self, function, *args, **kwargs):
        deadlines = [getattr(self, name, None) for name in ('_control_deadline', '_request_deadline')]
        deadline = min((value for value in deadlines if value is not None), default=None)
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CorrectnessRuntimeError('startup control exhausted the original request deadline')
            kwargs['timeout'] = min(kwargs['timeout'], remaining)
        result = function(*args, **kwargs)
        if deadline is not None and time.monotonic() > deadline:
            raise CorrectnessRuntimeError('startup control exhausted the original request deadline')
        return result

    def run_fixed_base_control(self, request, *, deadline):
        """Trusted local entrypoint; the HTTP caller cannot select this mode."""
        if request.get('candidate_patch') != '' or self.used:
            raise CorrectnessRuntimeError('fixed-base control requires an unused no-prediction testbed')
        self._control_deadline = deadline
        self._fixed_base_control = True
        return self.run_correctness(request)

    def run_correctness(self, request):
        if not isinstance(request, dict) or request.get('instance_id') != self.instance_id:
            raise CorrectnessRuntimeError('request does not match the prepared instance')
        oversize = request.get('schema') == OVERSIZE_REQUEST_SCHEMA
        if oversize:
            oversize_evidence = validate_oversize_request(request)
        elif request != build_test_request(self.instance_id, request.get('candidate_patch')):
            raise CorrectnessRuntimeError('request differs from fixed canonical correctness instructions')
        if self.used:
            raise CorrectnessRuntimeError('correctness testbed has already been consumed')
        self.used = True
        # All stages share the canonical request allowance; activity cannot
        # restart it for another patch attempt, syntax probe or test command.
        cutoff = time.monotonic() + request['timeout_seconds']
        self._request_deadline = min(getattr(self, '_request_deadline', cutoff), cutoff)
        row = _case(self.instance_id)
        identity = inspect_testbed(self.testbed, row['base_commit'], self.max_output_bytes,
                                   execute=lambda *args, **kwargs: self._bounded(self.inspect_execute, *args, **kwargs))
        receipt = {'schema': 'frontier.vllm-swe-worker-receipt/v1', 'instance_id': self.instance_id,
                   'request_sha256': request['request_sha256'], 'testbed': identity,
                   'harness_version': request.get('harness_version'), 'timeout_seconds': request['timeout_seconds'],
                   'candidate_patch_applied': False, 'test_patch_applied': False,
                   'tests_started': False, 'tests_completed': False, 'patch_attempts': [],
                   'execution': 'incomplete', 'scope': 'Local execution evidence; not an authenticated platform verdict.'}
        if oversize:
            receipt.update(execution='candidate_patch_oversize', oversize_evidence=oversize_evidence)
            return receipt
        # Preserve 4.1.0's fallback order, including its partial-apply behavior.
        commands = [('git', 'apply', '--verbose', '-'), ('git', 'apply', '--verbose', '--reject', '-'),
                    ('patch', '--batch', '--fuzz=5', '-p1', '-i', '-')]
        logs = []
        remaining = self.max_output_bytes
        if getattr(self, '_fixed_base_control', False):
            receipt['fixed_base_control'] = True
            receipt['candidate_patch_applied'] = True
        for command in (() if getattr(self, '_fixed_base_control', False) else commands):
            result = self._bounded(self.execute, list(command), cwd=self.testbed, timeout=request['timeout_seconds'],
                          output_limit=remaining, stdin=request['candidate_patch'].encode())
            logs.append(result.output)
            remaining -= len(result.output) + len(result.stages)
            receipt['patch_attempts'].append({'command': list(command), **_execution_receipt(result)})
            if result.stop_reason:
                receipt['execution'], receipt['stopped_stage'] = result.stop_reason, 'candidate_patch'
                break
            if result.exit_code == 0:
                receipt['candidate_patch_applied'] = True
                break
        receipt['patch_apply_log'] = b'\n'.join(logs).decode('utf-8', errors='replace')
        if not receipt['candidate_patch_applied']:
            if receipt['execution'] == 'incomplete':
                receipt['execution'] = 'candidate_patch_rejected'
            return receipt
        if remaining <= 0:
            receipt['execution'], receipt['stopped_stage'] = 'output_limit', 'before_tests'
            return receipt
        probe = self._bounded(self.execute, [self.testbed_python, '-I', str(Path(__file__).with_name('swe_syntax_probe.py')),
                              row['base_commit'], str(self.testbed)], cwd=self.testbed,
                             timeout=request['timeout_seconds'], output_limit=remaining)
        remaining -= len(probe.output) + len(probe.stages)
        receipt['syntax_probe_execution'] = _execution_receipt(probe)
        if probe.stop_reason or probe.exit_code or remaining <= 0:
            receipt['execution'], receipt['stopped_stage'] = probe.stop_reason or 'syntax_probe_failed', 'syntax_probe'
            receipt['syntax_probe_log'] = probe.output.decode('utf-8', errors='replace')
            return receipt
        try:
            syntax = json.loads(probe.output)
            if not isinstance(syntax, dict) or any(not isinstance(syntax.get(field), list)
                                                 for field in ('checks', 'errors', 'skipped_paths')):
                raise ValueError('invalid syntax evidence fields')
            receipt['candidate_syntax_evidence'] = syntax['errors']
            receipt['candidate_syntax_checks'] = syntax['checks']
            receipt['syntax_probe_skipped_paths'] = syntax['skipped_paths']
        except (ValueError, UnicodeError):
            receipt['execution'], receipt['stopped_stage'] = 'syntax_probe_failed', 'syntax_probe'
            receipt['syntax_probe_log'] = probe.output.decode('utf-8', errors='replace')
            return receipt
        wrapper = instrument_script(request, row)
        result = self._bounded(self.execute, None, cwd=self.testbed, timeout=request['timeout_seconds'],
                      output_limit=remaining, script=wrapper)
        receipt['eval_script_sha256'] = request['eval_script_sha256']
        # Descriptor-independent template identifies instrumentation separately.
        receipt['instrumented_script_template_sha256'] = _digest(wrapper(3).encode())
        receipt['eval_execution'] = _execution_receipt(result)
        receipt['log'] = result.output.decode('utf-8', errors='replace')
        trace = result.stages.decode('ascii', errors='replace').splitlines()
        _, specs, make_script = _harness()
        commands = make_script(row, specs[row['repo']][row['version']], 'testbed', '/testbed', row['base_commit'], row['test_patch'])
        setup_labels = ['setup_' + str(index) for index in range(len(commands) - 6)]
        labels = setup_labels + ['test_reset', 'test_patch', 'tests', 'test_cleanup']
        exits = {}
        for expected, line in zip(labels, trace):
            try:
                key, value = line.split('=', 1)
                if key != expected:
                    raise ValueError
                exits[key] = int(value)
            except ValueError:
                break
        receipt['stage_exit_codes'] = exits
        receipt['stage_trace'] = trace
        receipt['test_patch_applied'] = exits.get('test_reset') == 0 and exits.get('test_patch') == 0
        receipt['tests_started'] = 'test_patch' in exits
        receipt['tests_completed'] = 'tests' in exits and result.stop_reason is None
        if result.stop_reason:
            receipt['execution'], receipt['stopped_stage'] = result.stop_reason, 'eval_script'
        elif any(exits.get(label, 0) != 0 for label in setup_labels):
            receipt['execution'] = 'setup_failed'
            receipt['stopped_stage'] = next(label for label in setup_labels if exits.get(label, 0) != 0)
        elif len(trace) != len(labels) or list(exits) != labels or not receipt['test_patch_applied']:
            receipt['execution'] = 'incomplete_test_stages'
        else:
            # A nonzero test exit is evidence for the existing original parser,
            # not an infrastructure verdict or a new worker-level pass rule.
            receipt['execution'] = 'complete'
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance-id', required=True)
    parser.add_argument('--request', required=True, help='Request file, or - for bounded stdin')
    parser.add_argument('--receipt', required=True, help='New receipt file, or - for stdout')
    parser.add_argument('--max-request-bytes', type=int, required=True)
    parser.add_argument('--max-output-bytes', type=int, required=True)
    args = parser.parse_args(argv)
    if args.max_request_bytes <= 0:
        parser.error('request requires a positive externally supplied byte bound')
    if args.request == '-':
        raw = sys.stdin.buffer.read(args.max_request_bytes + 1)
    else:
        with Path(args.request).open('rb') as handle:
            raw = handle.read(args.max_request_bytes + 1)
    if len(raw) > args.max_request_bytes:
        parser.error('request exceeds the externally supplied byte bound')
    request = json.loads(raw)
    backend = PreparedSWEWorker(instance_id=args.instance_id, max_output_bytes=args.max_output_bytes)
    receipt = backend.run_correctness(request)
    encoded = json.dumps(receipt, indent=2, allow_nan=False) + '\n'
    if args.receipt == '-':
        sys.stdout.write(encoded)
        sys.stdout.flush()
    else:
        with Path(args.receipt).open('x') as handle:
            handle.write(encoded)


if __name__ == '__main__':
    main()
