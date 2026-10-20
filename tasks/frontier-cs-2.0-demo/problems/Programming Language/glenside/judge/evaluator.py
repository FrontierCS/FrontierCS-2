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

from submission_patch import apply_submission_patch
from sandbox import IsolatedProcessError, run_isolated
from cargo_target import CargoLibraryTarget

APP = TASK / 'harbor/app'
PROJECT = Path('code')
ENTRYPOINT = 'code/src/lib.rs'
REWRITES = PROJECT / 'src/language/rewrites.rs'
ANALYSIS = PROJECT / 'src/language/solver_extensions/access_analysis.rs'
REFERENCE_PATCH = TASK / 'reference.patch'
BASELINE_PATCH = TASK / 'baseline_naive.patch'
WORK_ROOT = Path(os.environ.get('GLENSIDE_WORK_ROOT', str(TASK / '.evaluator-work'))).resolve()
CONFIG = yaml.safe_load((TASK / 'config.yaml').read_text())
METRIC_NAMES = tuple(metric['name'] for metric in json.loads((TASK / 'beat_spec.json').read_text())['metrics'])
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
    """A deterministic rejection of submitted source, not a trusted runtime fault."""


def _validate_candidate_tree(tree: 'Path', trusted: bool = False) -> None:
    extension_prefix = PROJECT / 'src/language/solver_extensions'
    for original in APP.rglob('*'):
        if original.is_file() and original.relative_to(APP) not in (REWRITES, ANALYSIS):
            copied = tree / original.relative_to(APP)
            if not copied.is_file() or copied.read_bytes() != original.read_bytes():
                raise CandidateRejection('submission patch changed immutable source')
    for path in tree.rglob('*'):
        if path.is_symlink():
            raise CandidateRejection('symbolic links are not permitted')
        if path.is_file() and not (APP / path.relative_to(tree)).exists() and not path.relative_to(tree).is_relative_to(extension_prefix):
            raise CandidateRejection('submission patch added a file outside the editable surface')
    text = (tree / REWRITES).read_text()
    starter = (APP / REWRITES).read_text()
    fixed = _function(starter, 'systolic_array')
    if _function(text, 'systolic_array') != fixed:
        raise CandidateRejection('the fixed downstream mapping function must remain unchanged')
    blocked = _function(starter, 'systolic_array_with_blocking')
    if _function(text, 'systolic_array_with_blocking') != blocked:
        raise CandidateRejection('the fixed blocked mapping function must remain unchanged')
    constraint = starter[starter.index('fn constrain_access('):starter.index('pub fn flatten_unflatten_any_access')].strip()
    if constraint not in text:
        raise CandidateRejection('the fixed downstream shape constraint must remain unchanged')
    if trusted:
        return
    if len(re.findall(r'pub\s+fn\s+systolic_array\s*\(', text)) != 1:
        raise CandidateRejection('the fixed downstream mapping function cannot be shadowed')
    if len(re.findall(r'pub\s+fn\s+systolic_array_with_blocking\s*\(', text)) != 1:
        raise CandidateRejection('the fixed blocked mapping function cannot be shadowed')
    source = text.replace(fixed, '').replace(blocked, '')
    if re.search(r'\b(SystolicArray\w*|RelayOperatorCall|AcceleratorCall)\b', source):
        raise CandidateRejection('generic rewrites cannot construct accelerator or opaque calls')
    extensions = tree / PROJECT / 'src/language/solver_extensions'
    if extensions.exists():
        for path in extensions.rglob('*'):
            if path.is_file():
                if path.suffix != '.rs':
                    raise CandidateRejection('extension files must be Rust source')
                source += '\n' + path.read_text()
    source = _without_rust_comments(source)
    forbidden = r'\b(unsafe|extern|include|include_str|include_bytes|env|option_env|asm|global_asm|print|println|eprint|eprintln|dbg|fs|io|net|process|thread|libc|Command|File|OpenOptions|SystemTime|Instant)\b'
    if re.search(forbidden, source):
        raise CandidateRejection('only pure generic rewrite and analysis code is permitted; no I/O')
    for module_path in re.findall(r'\bpath\s*=\s*"([^"]+)"', source):
        parts = Path(module_path)
        if parts.is_absolute() or '..' in parts.parts or not module_path.startswith('solver_extensions/') or parts.suffix != '.rs':
            raise CandidateRejection('external source redirection is not permitted')
    if re.search(r'\b(link|link_section|no_mangle|export_name)\b', source):
        raise CandidateRejection('external source or symbol redirection is not permitted')


def _cargo_home() -> 'Path':
    configured = os.environ.get('CARGO_HOME')
    if configured and (Path(configured) / 'registry').is_dir():
        return Path(configured).resolve()
    if Path('/opt/cargo/registry').is_dir():
        return Path('/opt/cargo')
    cargo = WORK_ROOT / 'cargo-home'
    if not (cargo / 'registry').is_dir():
        cargo.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '-xJf', str(TASK / 'environment/cargo-cache.tar.xz'), '-C', str(cargo)], check=True, timeout=60)
    return cargo


def _build_cargo_home(label):
    source = _cargo_home()
    target = WORK_ROOT / (label + '-cargo-home')
    if not target.exists():
        shutil.copytree(source, target, symlinks=False)
        # Image dependencies are read-only to the non-root Verifier. Cargo's
        # lock/cache bookkeeping belongs in each build's private writable copy.
        target.chmod(target.stat().st_mode | 0o700)
        for path in target.rglob('*'):
            path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o600))
    return target


def _toolchain() -> 'Path':
    compiler = shutil.which('rustc')
    if not compiler:
        raise RuntimeError('Rust toolchain is unavailable')
    compiler = Path(compiler).resolve()
    if compiler.name == 'rustup':
        compiler = Path(subprocess.check_output(['rustup', 'which', 'rustc'], text=True).strip())
    return compiler.parent.parent


def _apply(patch: 'Path') -> 'Path':
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(WORK_ROOT)
    result = apply_submission_patch(patch, TASK, CONFIG, pristine=APP)
    if not result.ok or result.workdir is None:
        if result.candidate_rejection:
            raise CandidateRejection('submission patch was rejected: invalid format or scope')
        raise RuntimeError('submission patch application failed without candidate-origin evidence')
    return Path(result.workdir)


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
    binary = WORK_ROOT / (label + '-probe')
    stamp = WORK_ROOT / (label + '-build-signature')
    if binary.is_file() and stamp.is_file() and stamp.read_text() == signature.hexdigest():
        return binary
    build = WORK_ROOT / (label + '-build')
    target = WORK_ROOT / (label + '-target')
    build.mkdir(exist_ok=True)
    if not target.exists():
        seed = Path('/opt/frontier-dependency-target-' + label)
        if seed.is_dir():
            shutil.copytree(seed, target)
        else:
            target.mkdir()
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
    command = ['/toolchain/bin/cargo', 'build', '--manifest-path', '/build/code/Cargo.toml', '--locked', '--offline', '--no-default-features', '--lib', '--example', 'frontier_probe', '--message-format=json']

    def compile_target():
        try:
            run_isolated(command, mounts=[(str(build), '/build', True), (str(target), '/target', True), (str(_build_cargo_home(label)), '/cargo', True), (str(_toolchain()), '/toolchain', False)], cwd='/build/code', timeout=_remaining(180), environment={
            'PATH': '/toolchain/bin:/usr/bin:/bin', 'CARGO_HOME': '/cargo', 'CARGO_TARGET_DIR': '/target',
            'CARGO_BUILD_JOBS': '1', 'CARGO_PROFILE_DEV_OPT_LEVEL': '1', 'CARGO_PROFILE_DEV_DEBUG': '0',
            'CARGO_NET_OFFLINE': 'true', 'HOME': '/tmp',
            })

        except IsolatedProcessError as exc:
            if not trusted and label == 'candidate' and _candidate_compile_diagnostic(exc):
                raise CandidateRejection('candidate source does not compile') from exc
            raise

    _compile_library(source, compile_target)
    shutil.copyfile(target / 'debug/examples/frontier_probe', binary)
    binary.chmod(0o755)
    stamp.write_text(signature.hexdigest())
    return binary


_TRUSTED_BINARY: Path | None = None


def _trusted_binary() -> 'Path':
    global _TRUSTED_BINARY
    if _TRUSTED_BINARY is None:
        try:
            tree = _apply(REFERENCE_PATCH)
        except CandidateRejection:
            raise RuntimeError('trusted patch validation failed') from None
        try:
            try:
                _TRUSTED_BINARY = _build(tree, 'trusted', trusted=True)
            except CandidateRejection:
                raise RuntimeError('trusted source validation failed') from None
        finally:
            shutil.rmtree(tree.parent)
    return _TRUSTED_BINARY


def _invoke(binary: 'Path', payload: dict, timeout: int = 60):
    output = run_isolated(['/probe'], mounts=[(str(binary), '/probe', False)], cwd='/', timeout=_remaining(timeout), payload=json.dumps(payload), environment={'PATH': '/usr/bin:/bin', 'HOME': '/tmp'})
    return json.loads(output)


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
    except Exception as exc:
        raise RuntimeError('candidate analysis invocation failed') from exc
    if (not isinstance(actual, list) or len(actual) != len(expected)
            or any(not isinstance(row, dict) for row in actual)
            or [_semantic_contract_row(row) for row in actual] != [_semantic_contract_row(row) for row in expected]):
        raise CandidateRejection('candidate analysis contract failed')


def _run_patch(patch: 'Path', workloads: list[dict], label: str = 'candidate') -> dict:
    trusted = _trusted_binary()
    tree = _apply(patch)
    try:
        authenticated = patch.read_bytes() == REFERENCE_PATCH.read_bytes()
        binary = _build(tree, label, trusted=authenticated)
        return _measure(binary, workloads, trusted)
    finally:
        shutil.rmtree(tree.parent)


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


def evaluate(solution_path: str) -> tuple[float, float, str, dict]:
    global _DEADLINE
    started = time.monotonic()
    _DEADLINE = started + CONFIG['runtime']['timeout_seconds']
    try:
        solution = Path(solution_path)
        if not solution.is_file():
            raise FileNotFoundError('accepted submission file does not exist')
        suite_name, workloads = _suite()
        authenticated = solution.read_bytes() == REFERENCE_PATCH.read_bytes()
        try:
            candidate_tree = _apply(solution)
        except CandidateRejection:
            if authenticated:
                raise RuntimeError('trusted patch validation failed') from None
            raise
        try:
            try:
                _validate_candidate_tree(candidate_tree, trusted=authenticated)
            except CandidateRejection:
                if authenticated:
                    raise RuntimeError('trusted source validation failed') from None
                raise
            trusted = _trusted_binary()
            candidate = _build(candidate_tree, 'candidate', trusted=authenticated)
            _check_analysis_contract(candidate, trusted)
            reference_rows, candidate_rows = {}, {}
            for index, workload in enumerate(workloads):
                pair = [(trusted, reference_rows), (candidate, candidate_rows)]
                for binary, rows in pair[::1 if index % 2 == 0 else -1]:
                    rows.update(_measure(binary, [workload], trusted))
        finally:
            shutil.rmtree(candidate_tree.parent)
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
        return 0.0, 0.0, str(exc)[:160], {'candidate_rejection': True, 'beats_reference': 0}
    finally:
        _DEADLINE = None


if __name__ == '__main__':
    import sys
    result = evaluate(sys.argv[1] if len(sys.argv) > 1 else str(REFERENCE_PATCH))
    print(json.dumps(dict(zip(('score', 'score_unbounded', 'message', 'metrics'), result)), sort_keys=True))
