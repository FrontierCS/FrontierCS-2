"""Glenside access-pattern rewriting judge.

The shared verifier applies the patch (task.toml [submission]: code/src/language/rewrites.rs
and .rs files under code/src/language/solver_extensions/), runs check_tree (the source rules:
fixed mappers and shape constraint exact and unshadowed, safe Rust that leaves the probe's
stdout alone, no accelerator construction) and then evaluate(tree). The build and every probe
run under the shared isolate launcher. A compile or analysis-contract failure is a bare scored zero.

Score: per workload w, D = valid im2col discovery (0/1, three numerically verified
witnesses), t = three-run median rewrite seconds, g = max e-graph node growth;
r_t = (10 / t) ** 0.125, r_g = (100001 / (g + 1)) ** 0.125,
score_w = D * (60 + 20 r_t / (1 + r_t) + 20 r_g / (1 + r_g)); raw_score = score_unbounded =
mean over the 14 workloads of the suite (development or final).

Metrics, one row per workload in beat_spec_workload_passes, against the reference measured
interleaved in the same evaluation:
  im2col_discovery      higher is better; passes when candidate >= reference.
  rewrite_time_seconds  lower is better; passes when candidate <= reference.
  egraph_node_growth    lower is better; passes when candidate <= reference.
beats_reference = 1 iff every workload passes all three. Train also returns
development_im2col_discovered_count and the two development medians.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from statistics import median

import pytest
import yaml

TASK = Path(__file__).resolve().parent
if str(TASK) not in sys.path:
    sys.path.insert(0, str(TASK))

from cargo_target import CargoLibraryTarget
import isolate  # the shared launcher, beside edit_scope in the Verifier base

# The starter repo, where the shared verifier overlays the submission; the trusted tree is
# this plus reference.patch.
APP = TASK / 'pristine'
PROJECT = Path('code')
ENTRYPOINT = 'code/src/lib.rs'
REWRITES = PROJECT / 'src/language/rewrites.rs'
REFERENCE_PATCH = TASK / 'reference.patch'
BASELINE_PATCH = TASK / 'baseline_naive.patch'
WORK_ROOT = Path(os.environ.get('GLENSIDE_WORK_ROOT', str(TASK / '.evaluator-work'))).resolve()
CONFIG = yaml.safe_load((TASK / 'config.yaml').read_text())
METRIC_NAMES = ('im2col_discovery', 'rewrite_time_seconds', 'egraph_node_growth')  # the beat-spec metrics
_DEADLINE = None


def _remaining(limit):
    remaining = limit if _DEADLINE is None else min(limit, _DEADLINE - time.monotonic())
    if remaining <= 0:
        raise RuntimeError('evaluation deadline exceeded')
    return remaining


def generate_suite(seed: int, label: str) -> list[dict]:
    generator = random.Random(seed)
    workloads = []
    for index in range(12):
        layout = ('NCHW', 'NHWC')[(index + index // 4) % 2]
        batch = (1, 2, 3)[(index + index // 4) % 3]
        channels = [2, 3, 4, 5][index % 4]
        outputs = [2, 3, 5, 6][(index + index // 3) % 4]
        height, width = generator.randint(6, 10), generator.randint(6, 10)
        kernel_height, kernel_width = [(1, 1), (2, 3), (3, 2), (2, 2)][index % 4]
        strides = [[1, 1], [2, 1], [1, 2], [2, 2]][index % 4]
        padding = [0, 0, 0, 0] if index % 4 == 0 else [generator.randint(0, 2) for _ in range(4)]
        tensor_seed = generator.getrandbits(48)
        name_seed = generator.getrandbits(48)
        workloads.append({
            'id': f'{label}-{index:02d}',
            'data_shape': [batch, channels, height, width] if layout == 'NCHW' else [batch, height, width, channels],
            'weight_shape': [outputs, channels, kernel_height, kernel_width] if layout == 'NCHW' else [kernel_height, kernel_width, channels, outputs],
            'strides': strides, 'padding': padding, 'layout': layout,
            'wrapper': (index // 3) % 4, 'seed': tensor_seed,
            'data_name': f'tensor-{name_seed:x}', 'weight_name': f'filter-{name_seed:x}',
        })
    return workloads


OUTER_CONCAT_GENERATION_SEEDS = {
    'development': 2026092901,
    'graded': 2026092902,
}
OUTER_CONCAT_NUMERICAL_SEEDS = {
    'development': (0x6A1D4C8E22B7, 0x19E3F570CA41),
    'graded': (0x74B2A90D31EF, 0x2C8F615BE407),
}


def _outer_tensor(name: str, outer: list[int], item: list[int]) -> dict:
    return {'name': name, 'shape': outer + item, 'access_axis': 2}


def generate_outer_concat_suite(label: str, generation_seed: int, numerical_seeds: tuple[int, int]) -> list[dict]:
    generator = random.Random(generation_seed)
    records = []
    for index, (kind, concat_axis, numerical_seed) in enumerate(
        [('left', 0, numerical_seeds[0]), ('right', 3, numerical_seeds[1])]
    ):
        item_extent = [generator.choice((2, 3)), generator.choice((2, 3))]
        common_left = generator.choice((2, 3))
        common_right = generator.choice((2, 3))
        if kind == 'left':
            varying_outer = generator.sample((2, 3), 2)
            common_outer = generator.choice((2, 3))
            left_outer = ([varying_outer[0], common_left], [varying_outer[1], common_left])
            right_outer = ([common_right, common_outer], [common_right, common_outer])
        else:
            common_outer = generator.choice((2, 3))
            varying_outer = generator.sample((2, 3), 2)
            left_outer = ([common_outer, common_left], [common_outer, common_left])
            right_outer = ([common_right, varying_outer[0]], [common_right, varying_outer[1]])
        names = [
            f'{label}-outer-{kind}-{index}-{slot}-{generator.getrandbits(48):012x}'
            for slot in ('a0', 'a1', 'b0', 'b1')
        ]
        left = [_outer_tensor(names[0], left_outer[0], item_extent),
                _outer_tensor(names[1], left_outer[1], item_extent)]
        right = [_outer_tensor(names[2], right_outer[0], item_extent),
                 _outer_tensor(names[3], right_outer[1], item_extent)]
        records.append({
            'id': f'{label}-outer-{kind}',
            'data_shape': left[0]['shape'],
            'weight_shape': right[0]['shape'],
            'strides': [], 'padding': [], 'layout': 'EXPLICIT_OUTER_CONCAT',
            'wrapper': 0, 'seed': numerical_seed,
            'data_name': names[0], 'weight_name': names[2],
            'structure': {'concat_axis': concat_axis, 'left': left, 'right': right},
        })
    return records


DEV_WORKLOADS = generate_suite(932801, 'development') + generate_outer_concat_suite(
    'development', OUTER_CONCAT_GENERATION_SEEDS['development'], OUTER_CONCAT_NUMERICAL_SEEDS['development'])
FINAL_WORKLOADS = generate_suite(6701543921, 'graded') + generate_outer_concat_suite(
    'graded', OUTER_CONCAT_GENERATION_SEEDS['graded'], OUTER_CONCAT_NUMERICAL_SEEDS['graded'])


def _suite() -> tuple[str, list[dict]]:
    return ('final', FINAL_WORKLOADS) if os.environ.get('FRONTIER_SUBMISSION_ROLE') == 'final' else ('development', DEV_WORKLOADS)


def _function(text: str, name: str) -> str:
    start = text.index('pub fn ' + name + '(')
    opening = text.index('{', start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (text[cursor] == '{') - (text[cursor] == '}')
        cursor += 1
    return text[start:cursor]


def _without_rust_comments(source):
    output = []
    cursor = 0
    while cursor < len(source):
        if source.startswith('//', cursor):
            end = source.find('\n', cursor)
            end = len(source) if end < 0 else end
            output.append(' ' * (end - cursor))
            cursor = end
        elif source.startswith('/*', cursor):
            start, depth = cursor, 1
            cursor += 2
            while cursor < len(source) and depth:
                if source.startswith('/*', cursor):
                    depth += 1
                    cursor += 2
                elif source.startswith('*/', cursor):
                    depth -= 1
                    cursor += 2
                else:
                    cursor += 1
            if depth:
                raise ValueError('unterminated Rust comment')
            output.append(''.join('\n' if c == '\n' else ' '
                                  for c in source[start:cursor]))
        else:
            raw = re.match(r'(?:br|cr|r)(\#*)"', source[cursor:])
            if raw and (cursor == 0 or not (source[cursor-1].isalnum()
                                           or source[cursor-1] == '_')):
                terminator = '"' + raw.group(1)
                end = source.find(terminator, cursor + raw.end())
                if end < 0:
                    raise ValueError('unterminated Rust raw string')
                end += len(terminator)
            elif source[cursor] == '"':
                end = cursor + 1
                while end < len(source):
                    if source[end] == '\\':
                        end += 2
                    elif source[end] == '"':
                        end += 1
                        break
                    else:
                        end += 1
                else:
                    raise ValueError('unterminated Rust string')
            elif source[cursor] == "'":
                char = re.match(r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^\\'\n])'",
                                source[cursor:])
                end = cursor + len(char.group()) if char else cursor + 1
            else:
                end = cursor + 1
            output.append(source[cursor:end])
            cursor = end
    return ''.join(output)


class CandidateRejection(ValueError):
    """A deterministic rejection of submitted source, not a trusted runtime fault.

    `category` is the shared rejection category (edit_scope.CATEGORIES) when the source check
    found it; a compile or analysis-contract failure has none and is a scored zero."""

    def __init__(self, message, category=None):
        super().__init__(message)
        self.category = category


def _validate_candidate_tree(tree: 'Path', trusted: bool = False) -> None:
    # Which files may change (rewrites.rs, Rust files under solver_extensions/), symlinks and
    # file types are task.toml [submission], enforced by the shared edit scope before this runs.
    text = (tree / REWRITES).read_text()
    starter = (APP / REWRITES).read_text()
    fixed = _function(starter, 'systolic_array')
    if _function(text, 'systolic_array') != fixed:
        raise CandidateRejection('the fixed downstream mapping function must remain unchanged', 'out_of_scope')
    blocked = _function(starter, 'systolic_array_with_blocking')
    if _function(text, 'systolic_array_with_blocking') != blocked:
        raise CandidateRejection('the fixed blocked mapping function must remain unchanged', 'out_of_scope')
    constraint = starter[starter.index('fn constrain_access('):starter.index('pub fn flatten_unflatten_any_access')].strip()
    if constraint not in text:
        raise CandidateRejection('the fixed downstream shape constraint must remain unchanged', 'out_of_scope')
    if trusted:
        return
    if len(re.findall(r'pub\s+fn\s+systolic_array\s*\(', text)) != 1:
        raise CandidateRejection('the fixed downstream mapping function cannot be shadowed', 'out_of_scope')
    if len(re.findall(r'pub\s+fn\s+systolic_array_with_blocking\s*\(', text)) != 1:
        raise CandidateRejection('the fixed blocked mapping function cannot be shadowed', 'out_of_scope')
    source = text.replace(fixed, '').replace(blocked, '')
    if re.search(r'\b(SystolicArray\w*|RelayOperatorCall|AcceleratorCall)\b', source):
        raise CandidateRejection('generic rewrites cannot construct accelerator or opaque calls', 'content_policy')
    extensions = tree / PROJECT / 'src/language/solver_extensions'
    if extensions.exists():
        for path in extensions.rglob('*'):
            if path.is_file():
                source += '\n' + path.read_text()
    source = _without_rust_comments(source)
    # Files, environment, network, processes and clocks are isolate's business (the
    # build and every probe run in a fresh confined instance). What stays is language-level:
    # stdout is the probe's result channel (print/println/io), and the probe times the search in
    # its own process, so safe Rust with no foreign or symbol-level code is the timer's boundary.
    forbidden = r'\b(unsafe|extern|asm|global_asm|print|println|io)\b'
    if re.search(forbidden, source):
        raise CandidateRejection('only safe Rust that leaves the probe output channel alone is permitted', 'content_policy')
    for module_path in re.findall(r'\bpath\s*=\s*"([^"]+)"', source):
        parts = Path(module_path)
        if parts.is_absolute() or '..' in parts.parts or not module_path.startswith('solver_extensions/') or parts.suffix != '.rs':
            raise CandidateRejection('external source redirection is not permitted', 'content_policy')
    if re.search(r'\b(link|link_section|no_mangle|export_name)\b', source):
        raise CandidateRejection('external source or symbol redirection is not permitted', 'content_policy')


# The offline Cargo home and the dependency build seed (built against this CARGO_HOME, whose
# path is part of Cargo's fingerprints) are read-only image paths every build reads in place.
CARGO_HOME = Path('/opt/cargo')
DEPENDENCY_SEED = Path('/opt/frontier-cargo-seed')
PROBE_LIMITS = {'memory_bytes': 3 << 30, 'file_bytes': 64 << 20}


class IsolatedProcessError(RuntimeError):
    """A sandboxed command's failed run: its exit status and bounded output."""

    def __init__(self, returncode, stdout, stderr):
        super().__init__("isolated process failed: " + stderr)
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _run_isolated(argv, *, timeout, syscalls, stdin=b'', tree=None, env=None, read_paths=(),
                  outputs=(), output_bytes=2_097_153):
    """Run argv in a fresh isolate instance (no judge files, environment, network or
    state from another run). A timeout is a deadline fault; any other failure is a receipt."""
    seconds = _remaining(timeout)
    result = isolate.run(argv, timeout_seconds=seconds, cpu_seconds=math.ceil(seconds),
                              tree=tree, stdin=stdin, env=env or {}, read_paths=read_paths,
                              outputs=outputs, output_bytes=output_bytes, syscalls=syscalls,
                              **PROBE_LIMITS)
    if result.timed_out:
        raise RuntimeError('sandbox deadline exceeded')
    if result.returncode or result.limit:
        raise IsolatedProcessError(result.returncode,
                                   result.stdout[:2_097_152].decode(errors='replace'),
                                   result.stderr[:16384].decode(errors='replace'))
    return result


def _toolchain() -> 'Path':
    compiler = shutil.which('rustc')
    if not compiler:
        raise RuntimeError('Rust toolchain is unavailable')
    compiler = Path(compiler).resolve()
    if compiler.name == 'rustup':
        compiler = Path(subprocess.check_output(['rustup', 'which', 'rustc'], text=True).strip())
    return compiler.parent.parent


def _reference_tree() -> 'Path':
    """The pristine starter with reference.patch applied (trusted, judge-owned); the caller
    removes its parent."""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='reference-', dir=WORK_ROOT))
    tree = scratch / 'tree'
    shutil.copytree(APP, tree, symlinks=True)
    for path in (tree, *tree.rglob('*')):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    env = {**os.environ, 'GIT_CEILING_DIRECTORIES': str(scratch.resolve())}
    subprocess.run(['git', 'apply', '--whitespace=nowarn', str(REFERENCE_PATCH)], cwd=tree,
                   check=True, capture_output=True, timeout=60, env=env)
    return tree


def _is_reference(tree: 'Path') -> bool:
    """Whether the judged tree is exactly the reference (pristine + reference.patch): the
    judge's own reference is exempt from the candidate-only source checks, as before, when
    the submitted patch was byte-identical to it."""
    reference = _reference_tree()
    try:
        names = {p.relative_to(root) for root in (tree, reference) for p in root.rglob('*')
                 if p.is_file() or p.is_symlink()}
        return all((tree / name).is_file() and (reference / name).is_file()
                   and (tree / name).read_bytes() == (reference / name).read_bytes()
                   for name in names)
    finally:
        shutil.rmtree(reference.parent, ignore_errors=True)


def check_tree(tree: str, pristine: str, changes: list) -> tuple[str, list[str]] | None:
    """Shared-verifier hook: the source rules task.toml cannot express. The fixed mapping
    functions and shape constraint stay exact and unshadowed; candidate code is safe Rust that
    leaves the probe's stdout alone (no unsafe/foreign code, printing or std::io, external-source
    or symbol-redirection constructs) and constructs no accelerator call."""
    tree = Path(tree)
    try:
        _validate_candidate_tree(tree, trusted=_is_reference(tree))
    except CandidateRejection as exc:
        if exc.category is None:
            raise
        return exc.category, [c.path for c in changes if c.status != 'D'] or [c.path for c in changes]
    return None


def _compile_library(source: 'Path', action) -> None:
    failures = []

    def captured_action():
        try:
            action()
        except Exception as exc:
            failures.append(exc)
            raise

    target = CargoLibraryTarget(APP / ENTRYPOINT, source, captured_action)
    settings = {
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        'PYTEST_ADDOPTS': '-c /dev/null --noconftest',
        'PYTEST_PLUGINS': '',
    }
    previous = {name: os.environ.get(name) for name in settings}
    try:
        os.environ.update(settings)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = pytest.main([str(APP / ENTRYPOINT), '-q', '-p', 'no:cacheprovider'], plugins=[target])
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if failures:
        raise failures[0]
    if status != 0 or target.completed != 1:
        raise RuntimeError('retained Cargo library build failed')


def _candidate_compile_diagnostic(receipt: IsolatedProcessError) -> bool:
    # Cargo exit 101 alone also covers disk/dependency faults. Require rustc error
    # diagnostics located on the task's explicit editable source surface.
    if receipt.returncode != 101:
        return False
    diagnostics = []
    for line in receipt.stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('reason') == 'compiler-message' and row.get('message', {}).get('level') == 'error':
            message = row['message']
            if (message.get('code') or {}).get('code') in {'E0460', 'E0461', 'E0462', 'E0463', 'E0514'}:
                return False
            diagnostics.append(message)
    def editable(span):
        name = span.get('file_name', '')
        name = name.removeprefix('/build/code/').removeprefix('code/')
        return name == 'src/language/rewrites.rs' or name.startswith('src/language/solver_extensions/')
    return bool(diagnostics) and all(
        any(span.get('is_primary') is True and editable(span) for span in message.get('spans', []))
        for message in diagnostics
    )


def _build(tree: 'Path', label: str, trusted: bool = False) -> 'Path':
    _validate_candidate_tree(tree, trusted)
    if (tree / ENTRYPOINT).read_bytes() != (APP / ENTRYPOINT).read_bytes():
        raise ValueError('submission patch changed the retained library entrypoint')
    signature = hashlib.sha256()
    for path in sorted((tree / PROJECT).rglob('*')):
        if path.is_file():
            signature.update(str(path.relative_to(tree)).encode())
            signature.update(path.read_bytes())
    for path in [TASK / 'probe.rs', TASK / 'analysis_contract.rs', TASK / 'evaluator.py', TASK / 'cargo_target.py', APP / 'cpu-feature-gates.patch', APP / REWRITES]:
        signature.update(path.read_bytes())
    signature.update(str(_toolchain()).encode())
    binary = WORK_ROOT / (label + '-probe') / 'probe'
    stamp = WORK_ROOT / (label + '-build-signature')
    if binary.is_file() and stamp.is_file() and stamp.read_text() == signature.hexdigest():
        return binary
    build = WORK_ROOT / (label + '-build')
    build.mkdir(exist_ok=True)
    source = build / 'code'
    if source.exists():
        shutil.rmtree(source)
    shutil.copytree(tree / PROJECT, source)
    repair = (APP / 'cpu-feature-gates.patch').read_text()
    sections = repair.split('--- a/')
    repair = ''.join('--- a/' + section for section in sections[1:] if not section.startswith('src/language/rewrites.rs\n'))
    subprocess.run(['patch', '--batch', '-p1'], input=repair, text=True, cwd=source, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rewritten = source / 'src/language/rewrites.rs'
    text = rewritten.read_text().replace('pub fn lstm_to_flexasr()', '#[cfg(feature = "tvm")]\npub fn lstm_to_flexasr()')
    text = text.replace('$(#[$meta])*\n            #[test]', '$(#[$meta])*\n            #[cfg(feature = "tvm")]\n            #[test]')
    rewritten.write_text(text)
    (source / 'examples').mkdir(exist_ok=True)
    shutil.copyfile(TASK / 'probe.rs', source / 'examples/frontier_probe.rs')
    shutil.copyfile(TASK / 'analysis_contract.rs', source / 'examples/analysis_contract.rs')
    starter = (APP / REWRITES).read_text()
    constraint = starter[starter.index('fn constrain_access('):starter.index('pub fn flatten_unflatten_any_access')].strip()
    fixed = '''use glenside::language::{Language, MyAnalysis, MyAnalysisData};
use egg::{rewrite, Applier, EGraph, Id, Pattern, PatternAst, Rewrite, Subst, Symbol, Var};
use ndarray::Dimension;
type EG = EGraph<Language, MyAnalysis>;
mod language { pub use glenside::language::*; }
mod implementation {
use super::*;
''' + constraint + '\n' + _function(starter, 'systolic_array') + '\n' + _function(starter, 'systolic_array_with_blocking') + '\n}\npub use implementation::{systolic_array, systolic_array_with_blocking};\n'
    (source / 'examples/fixed_mapping.rs').write_text(fixed)
    toolchain = _toolchain()
    # The instance gets its own writable copy of the source (tree) and of the dependency seed
    # (timestamps kept for Cargo's fingerprints; the image copy is read-only), builds, and hands
    # back only the probe binary.
    command = ['sh', '-c', 'cp -a "$0/." "$CARGO_TARGET_DIR/" && chmod -R u+w "$CARGO_TARGET_DIR" && "$@" && '
               'cp "$CARGO_TARGET_DIR/debug/examples/frontier_probe" "{out}/probe"',
               str(DEPENDENCY_SEED), str(toolchain / 'bin/cargo'), 'build', '--manifest-path',
               '{work}/code/Cargo.toml', '--locked', '--offline', '--no-default-features', '--lib',
               '--example', 'frontier_probe', '--message-format=json']
    built = {}

    def compile_target():
        try:
            built['probe'] = _run_isolated(command, tree=build, timeout=180, syscalls='default', env={
                'PATH': f'{toolchain}/bin:/usr/bin:/bin', 'CARGO_HOME': str(CARGO_HOME),
                'CARGO_TARGET_DIR': '{work}/target', 'CARGO_BUILD_JOBS': '1',
                'CARGO_PROFILE_DEV_OPT_LEVEL': '1', 'CARGO_PROFILE_DEV_DEBUG': '0',
                'CARGO_NET_OFFLINE': 'true',
            }, read_paths=[str(CARGO_HOME), str(DEPENDENCY_SEED)], outputs=['probe'],
                output_bytes=1 << 30).outputs['probe']
        except IsolatedProcessError as exc:
            if not trusted and label == 'candidate' and _candidate_compile_diagnostic(exc):
                raise CandidateRejection('candidate source does not compile') from exc
            raise

    _compile_library(source, compile_target)
    if not built.get('probe'):
        raise RuntimeError('the probe build produced no binary')
    binary.parent.mkdir(exist_ok=True)
    binary.write_bytes(built['probe'])
    binary.chmod(0o755)
    stamp.write_text(signature.hexdigest())
    return binary


_TRUSTED_BINARY: Path | None = None


def _trusted_binary() -> 'Path':
    global _TRUSTED_BINARY
    if _TRUSTED_BINARY is None:
        tree = _reference_tree()
        try:
            try:
                _TRUSTED_BINARY = _build(tree, 'trusted', trusted=True)
            except CandidateRejection:
                raise RuntimeError('trusted source validation failed') from None
        finally:
            shutil.rmtree(tree.parent)
    return _TRUSTED_BINARY


def _invoke(binary: 'Path', payload: dict, timeout: int = 60):
    # A fresh instance per call: its own copy of the probe, the workloads on stdin, the result
    # on stdout; single-threaded (syscalls="strict": no processes, threads or signals).
    result = _run_isolated(['{work}/probe'], tree=binary.parent, timeout=timeout, syscalls='strict',
                           stdin=json.dumps(payload).encode(), env={'PATH': '/usr/bin:/bin'})
    if len(result.stdout) > 2_097_152:
        raise RuntimeError('oversized process result')
    return json.loads(result.stdout.decode())


def _measure(binary: 'Path', workloads: list[dict], verifier: 'Path') -> dict[str, dict]:
    measurement_inputs = [{**workload, 'seed': 0} for workload in workloads]
    rows = _invoke(binary, {'mode': 'measure', 'workloads': measurement_inputs})
    if not isinstance(rows, list) or len(rows) != len(workloads):
        raise ValueError('invalid measurement output')
    for row, workload in zip(rows, workloads):
        if row['workload'] != workload['id'] or not math.isfinite(row['rewrite_time_seconds']) or row['rewrite_time_seconds'] <= 0:
            raise ValueError('invalid measurement fields')
        if type(row['egraph_node_growth']) is not int or row['egraph_node_growth'] < 0:
            raise ValueError('invalid e-graph growth')
    repeated_workloads = [workload for workload in workloads for _ in range(3)]
    guards = _invoke(verifier, {'mode': 'verify', 'workloads': repeated_workloads, 'witnesses': [witness for row in rows for witness in row['witnesses']]})
    if len(guards) != len(repeated_workloads) or any(type(guard) is not bool for guard in guards):
        raise ValueError('invalid semantic verification')
    return {row['workload']: {**row, 'witness': row['witnesses'][0], 'im2col_discovery': int(all(guards[index*3:index*3+3]))} for index, row in enumerate(rows)}


def _semantic_contract_row(row: dict) -> dict:
    """Discard implementation metadata only where independent checks cover semantics."""
    def fact(value, allow_settled):
        if isinstance(value, list):
            return [fact(item, allow_settled) for item in value]
        if isinstance(value, dict):
            return {key: fact(item, allow_settled) for key, item in value.items()
                    if not (key in ('settled', 'zero_regions') and not allow_settled)}
        return value
    projected = {}
    for key, value in row.items():
        if key == 'flags' and row.get('case') in ('both-unsettled', 'zero-region-retained'):
            continue
        # Rust independently verifies propagation at every expression node, allowing
        # conservative reshape and sound (not necessarily complete) zero inference;
        # merge settledness and target zero-region preservation stay checked.
        projected[key] = fact(value, key != 'facts')
    return projected


def _check_analysis_contract(candidate: Path, trusted: Path) -> None:
    payload = {'mode': 'analysis-contract', 'workloads': []}
    expected = _invoke(trusted, payload)
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(row, dict) or not isinstance(row.get('case'), str) for row in expected)
            or len({row['case'] for row in expected}) != len(expected)):
        raise RuntimeError('trusted analysis contract failed')
    try:
        actual = _invoke(candidate, payload)
    except IsolatedProcessError as exc:
        # This is the compiled candidate probe, not Cargo or a tool launcher.
        # Accept only Rust panic receipts on the trusted contract or editable core.
        panic_headers = re.findall(r"(?:^|\n)thread '[^'\n]+' panicked at ([^\n]+):\n", exc.stderr)
        if exc.returncode == 101 and panic_headers and re.fullmatch(
                r"(?:[^\n]*?/)?(?:examples/analysis_contract\.rs|src/language/solver_extensions/[^\n]+\.rs|src/language/rewrites\.rs):\d+:\d+",
                panic_headers[-1]):
            raise CandidateRejection('candidate analysis contract failed') from exc
        raise RuntimeError('candidate analysis invocation failed') from exc
    except isolate.IsolateError:
        raise
    except Exception as exc:
        raise RuntimeError('candidate analysis invocation failed') from exc
    if (not isinstance(actual, list) or len(actual) != len(expected)
            or any(not isinstance(row, dict) for row in actual)
            or [_semantic_contract_row(row) for row in actual] != [_semantic_contract_row(row) for row in expected]):
        raise CandidateRejection('candidate analysis contract failed')


def _score(rows: dict) -> float:
    if not rows:
        raise ValueError('empty measurements')
    scores = []
    for candidate in rows.values():
        discovery = candidate['im2col_discovery']
        # Public Runner budgets set scales; they are not measured reference data.
        time_ratio = (10.0 / candidate['rewrite_time_seconds']) ** 0.125
        growth_ratio = (100001.0 / (candidate['egraph_node_growth'] + 1)) ** 0.125
        timing = time_ratio / (1 + time_ratio)
        growth = growth_ratio / (1 + growth_ratio)
        scores.append(discovery * (60 + 20 * timing + 20 * growth))
    return sum(scores) / len(scores)


def _aggregate(rows: dict, reference: dict) -> tuple[float, list[dict]]:
    if rows.keys() != reference.keys() or not rows:
        raise ValueError('workload membership mismatch')
    passes = []
    for workload, candidate in rows.items():
        ref = reference[workload]
        discovery = candidate['im2col_discovery']
        passes.append({'workload': workload, 'metrics': {
            'im2col_discovery': bool(discovery >= ref['im2col_discovery']),
            'rewrite_time_seconds': bool(candidate['rewrite_time_seconds'] <= ref['rewrite_time_seconds']),
            'egraph_node_growth': bool(candidate['egraph_node_growth'] <= ref['egraph_node_growth']),
        }})
    return _score(rows), passes


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    """Score the judged tree (the starter with the submission's allowed files overlaid; the
    shared verifier has applied the patch and run check_tree)."""
    global _DEADLINE
    started = time.monotonic()
    _DEADLINE = started + CONFIG['runtime']['timeout_seconds']
    try:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(WORK_ROOT)
        suite_name, workloads = _suite()
        candidate_tree = Path(tree)
        authenticated = _is_reference(candidate_tree)
        try:
            _validate_candidate_tree(candidate_tree, trusted=authenticated)
        except CandidateRejection:
            if authenticated:
                raise RuntimeError('trusted source validation failed') from None
            raise RuntimeError('check_tree admitted a tree the source check rejects') from None
        trusted = _trusted_binary()
        candidate = _build(candidate_tree, 'candidate', trusted=authenticated)
        _check_analysis_contract(candidate, trusted)
        reference_rows, candidate_rows = {}, {}
        for index, workload in enumerate(workloads):
            pair = [(trusted, reference_rows), (candidate, candidate_rows)]
            for binary, rows in pair[::1 if index % 2 == 0 else -1]:
                rows.update(_measure(binary, [workload], trusted))
        if time.monotonic() - started > CONFIG['runtime']['timeout_seconds']:
            raise RuntimeError('evaluation deadline exceeded')
        score, passes = _aggregate(candidate_rows, reference_rows)
        reference_score, _ = _aggregate(reference_rows, reference_rows)
        metrics = {
            'beat_spec_workload_passes': passes,
            'beats_reference': int(all(value for row in passes for value in row['metrics'].values())),
            'reference_score': reference_score, 'margin': score - reference_score,
            'development_im2col_discovered_count': sum(row['im2col_discovery'] for row in candidate_rows.values()),
            'development_rewrite_time_seconds_median': median(row['rewrite_time_seconds'] for row in candidate_rows.values()),
            'development_egraph_node_growth_median': median(row['egraph_node_growth'] for row in candidate_rows.values()),
        }
        for prefix, rows in [('raw_per_workload_', candidate_rows), ('reference_', reference_rows)]:
            for metric in METRIC_NAMES:
                metrics[prefix + metric] = {name: row[metric] for name, row in rows.items()}
        return score, score, f'{suite_name}: {metrics["development_im2col_discovered_count"]}/{len(workloads)} valid discoveries', metrics
    except CandidateRejection as exc:
        # The candidate does not compile on its editable surface, or fails the analysis
        # contract: a scored zero with no measurement, exactly as before (train returns only
        # score_unbounded). No shared rejection category names a build or precondition
        # failure, so none is reported.
        return 0.0, 0.0, str(exc)[:160], {}
    finally:
        _DEADLINE = None
