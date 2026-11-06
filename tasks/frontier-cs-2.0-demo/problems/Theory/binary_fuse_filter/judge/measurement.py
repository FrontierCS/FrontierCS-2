"""Measurement procedure of binary_fuse_filter (the code the calibration binding covers).

Moved verbatim out of evaluator.py in the task.toml migration (#757) so that the reference-only
statistical calibration and its runtime receipt bind the measurement code itself
(evaluator.measurement_binding), not the submission glue or the generated config.yaml.
Nothing here reads the submission patch: measure() receives the judged tree.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

import isolate  # the shared launcher, beside edit_scope in the Verifier base

from accuracy_metrics import CalibrationRequired, METRIC_NAMES, non_inferior, summarize_accuracy

TASK = Path(__file__).resolve().parent

METRICS = ('resident_bytes', 'build_ns', 'lookup_ns')

ALL_METRICS = METRICS + METRIC_NAMES

MIXTURES = (0, 50, 100, 25)  # Existing indices 0..2 are unchanged; mix3 is shuffled.

REPEATS = 5

QUERIES = 1048576

# Keep all original nominal-FPR gates and their thresholds byte-equivalent.
FAMILY_TESTS = 1200

ALPHA = 1e-6

# The new stream has its own tighter per-sample gate; <=384 new gates/final.
NEW_STREAM_FAMILY_TESTS = 400

# The original <=1152 gates use <=0.96e-6 of their 1e-6 family bound.
# <=384 new gates use <=0.0384e-6: total remains <1e-6, original bars unchanged.
NEW_STREAM_ALPHA = 4e-8

TRAIN_SEED = 20261007

FINAL_SEED = 81985529216486895

class InvalidCandidate(Exception):
    pass

class InvalidReference(Exception):
    """A failed author baseline is not evidence of solver difficulty."""
    pass

def command(args, *, timeout=30, tree=None, inputs=None, outputs=(), read_paths=()):
    """One bounded command under the shared isolate launcher (#757): every compile and every run of
    candidate or reference code. The instance reads only the system, its inputs and the named
    paths, writes only its own directories, gets a clean environment (plus the variables below)
    and leaves nothing running. Limits as before: 4 GiB memory, 30 CPU seconds, 8 MiB files,
    the first 64 KiB of stdout and 4 KiB of stderr."""
    result = isolate.run(args, timeout_seconds=timeout, tree=tree, inputs=inputs,
                              outputs=outputs, read_paths=read_paths, path='/usr/bin:/bin',
                              env={'LANG': 'C', 'LC_ALL': 'C', 'OMP_NUM_THREADS': '1',
                                   'OPENBLAS_NUM_THREADS': '1'},
                              memory_bytes=4*2**30, cpu_seconds=30, file_bytes=8*2**20)
    if result.timed_out:
        raise InvalidCandidate('bounded command timed out')
    if result.returncode or result.limit:
        raise InvalidCandidate('command failed: ' + result.stderr[:4096].decode('utf-8', 'replace')[-1000:])
    return result.stdout[:65536].decode('utf-8', 'strict'), result.outputs

def compile_runner(include, directory):
    """The trusted harness linked with the bridge over `include` (the judged tree's include/
    or the reference headers); returns the binary, written to `directory`."""
    flags = ['cc', '-std=c99', '-O3', '-fno-lto', '-DNDEBUG']
    # Compiler is fixed, helpers can be header-only under the usable include/ surface.
    _, out = command(flags + ['-I', '{in}/include', '-c', '{in}/bridge.c', '-o', '{out}/bridge.o'],
                     inputs={'include': include, 'bridge.c': TASK / 'bridge.c'}, outputs=['bridge.o'])
    bridge = out['bridge.o']
    _, out = command(flags + ['-c', '{in}/harness.c', '-o', '{out}/harness.o'],
                     inputs={'harness.c': TASK / 'harness.c'}, outputs=['harness.o'])
    harness = out['harness.o']
    _, out = command(['cc', '{in}/bridge.o', '{in}/harness.o', '-lm', '-o', '{out}/measure'],
                     inputs={'bridge.o': bridge, 'harness.o': harness}, outputs=['measure'])
    binary = directory / 'measure'
    binary.write_bytes(out['measure'])
    binary.chmod(0o755)
    return binary

def fpr_limit(m, bits, *, family_tests=FAMILY_TESTS, alpha=ALPHA):
    mean = m / (2**bits)
    logterm = math.log(family_tests / alpha)
    return math.ceil(mean + math.sqrt(2*mean*logterm) + 2*logterm/3)

def validate_sample(row, bits, q):
    if row.get('construction') is not True:
        raise InvalidCandidate('construction did not succeed')
    for field in ['false_negatives', 'mixed_false_negatives', 'leaked_bytes']:
        if type(row.get(field)) is not int or row[field] != 0:
            raise InvalidCandidate('correctness or lifetime failure: ' + field)
    if row.get('absent_queries') != q:
        raise InvalidCandidate('wrong query count')
    fp = row.get('false_positives')
    mixed = row.get('mixed_false_positives')
    if type(fp) is not int or not 0 <= fp <= fpr_limit(q, bits):
        raise InvalidCandidate('false-positive confidence gate failed')
    effective_fp = row.get('effective_false_positives')
    if (type(effective_fp) is not int or not fp <= effective_fp <= fp + 7*q or
            type(row.get('effective_absent_queries')) is not int or
            row['effective_absent_queries'] != 8*q):
        raise InvalidCandidate('wrong effective-FPR measurement counts')
    if (not isinstance(mixed, list) or len(mixed) != 4 or
        any(type(x) is not int or x < 0 for x in mixed) or
        mixed[0] > fpr_limit(q, bits) or mixed[1] > fpr_limit(q//2, bits) or mixed[2] != 0 or
        mixed[3] > fpr_limit(3*q//4, bits, family_tests=NEW_STREAM_FAMILY_TESTS, alpha=NEW_STREAM_ALPHA)):
        raise InvalidCandidate('mixed-query confidence gate failed')
    counts = row.get('mixed_absent_queries')
    if (q % 4 or not isinstance(counts, list) or
            any(type(x) is not int for x in counts) or counts != [q, q//2, 0, 3*q//4] or
            type(row.get('shuffled_hit_queries')) is not int or row['shuffled_hit_queries'] != q//4):
        raise InvalidCandidate('wrong query mixture counts')
    if type(row.get('resident_bytes')) is not int:
        raise InvalidCandidate('non-integral allocation bytes')
    values = [row.get('resident_bytes'), row.get('build_ns')]
    lookup = row.get('lookup_ns')
    if not isinstance(lookup, list) or len(lookup) != 4:
        raise InvalidCandidate('malformed timing output')
    values += lookup
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise InvalidCandidate('non-positive or non-finite measurement')

def run_sample(binary, bits, n, family, seed, q=QUERIES):
    # The instance works on a copy of the binary's directory (the binary alone).
    text, _ = command(['./' + binary.name, str(bits), str(n), str(family), str(seed), str(q)],
                      tree=binary.parent)
    try:
        row = json.loads(text)
        validate_sample(row, bits, q)
    except (ValueError, TypeError, KeyError) as exc:
        raise InvalidCandidate('malformed measurement') from exc
    return row

def configurations(final):
    return [(bits, n, family, seed)
            for seed in ([TRAIN_SEED, FINAL_SEED] if final else [TRAIN_SEED])
            for bits in (8, 16) for n in (10000, 100000, 1000000) for family in range(3)]

def workload_ids(final):
    return [f'b{bits}-n{n}-f{family}-s{seed}-mix{mix}'
            for bits, n, family, seed in configurations(final) for mix in range(len(MIXTURES))]

def original_entrypoint(repo):
    # One instance on a writable copy of the judged tree: the build, then its binary.
    output, _ = command(['sh', '-c', 'make -j1 query CC=cc >/dev/null && exec ./query'],
                        tree=repo, timeout=60)
    for bits in (8, 16):
        if f'binary_fuse{bits}:' not in output:
            raise InvalidCandidate('original query entrypoint failed binary fuse construction')
    return output

def paired_measure(candidate, reference, final):
    results, raw = [], []
    for config in configurations(final):
        samples = {'candidate': [], 'reference': []}
        for repetition in range(REPEATS):
            # Alternate execution order, same process/runtime limits and key bytes.
            order = [('candidate', candidate), ('reference', reference)]
            if repetition % 2: order.reverse()
            for name, binary in order:
                try:
                    row = run_sample(binary, *config)
                except InvalidCandidate as exc:
                    if name == 'reference':
                        raise InvalidReference(str(exc)) from exc
                    raise
                samples[name].append(row)
        bits, n, family, seed = config
        ident = f'b{bits}-n{n}-f{family}-s{seed}'
        raw.append({'id': ident, 'candidate': samples['candidate'], 'reference': samples['reference']})
        accuracy = {name: summarize_accuracy(samples[name], n) for name in samples}
        for mix in range(len(MIXTURES)):
            measures = {}
            for name in samples:
                measures[name] = {
                    'resident_bytes': max(row['resident_bytes'] for row in samples[name]),
                    'build_ns': statistics.median(row['build_ns'] for row in samples[name]),
                    'lookup_ns': statistics.median(row['lookup_ns'][mix] for row in samples[name]),
                    **accuracy[name]}
            results.append({'workload': ident + f'-mix{mix}', **measures})
    return results, raw

def score_rows(rows, calibration):
    if not rows:
        raise ValueError('empty suite')
    decisions = []
    for row in rows:
        workload = row['workload']
        if workload not in calibration or set(calibration[workload]) != set(METRIC_NAMES):
            raise CalibrationRequired('missing workload preregistered anchors')
        passed = {m: row['candidate'][m] < row['reference'][m] for m in METRICS}
        # The accuracy guards compare the candidate's certified bounds with the
        # preregistered reference-only anchor; the paired reference row is still
        # measured and retained, but it never moves an anchor.
        passed.update({m: non_inferior(row['candidate'][m], calibration[workload][m])
                       for m in METRIC_NAMES})
        decisions.append({'workload': workload, 'metrics': passed})
    # Preserve the original descriptive score. All FIVE predicates, including
    # unresolved-domain failures, separately control strict PASS; no tradeoffs.
    score = statistics.mean(100 * row['reference'][m] / (row['candidate'][m] + row['reference'][m]) for row in rows for m in METRICS)
    return score, decisions, int(all(all(row['metrics'].values()) for row in decisions))


def measure(tree, final, calibration, start):
    """The whole measurement of one judged tree: compile the candidate and the exact
    reference with the trusted harness, run the original `make query` entrypoint and the
    edge-cardinality gates, then the paired workload measurement and its scoring.
    Raises InvalidCandidate for a candidate failure, InvalidReference for the reference."""
    with tempfile.TemporaryDirectory(prefix='binary-fuse-measure-') as tmp:
        root = Path(tmp)
        cdir, rdir = root / 'candidate', root / 'reference'
        cdir.mkdir(); rdir.mkdir()
        candidate = compile_runner(tree / 'include', cdir)
        try:
            reference = compile_runner(TASK / 'reference' / 'include', rdir)
        except InvalidCandidate as exc:
            raise InvalidReference(str(exc)) from exc
        spine = original_entrypoint(tree)
        # API edge cardinalities are correctness gates, never beat metrics.
        for bits in (8, 16):
            for n in (2, 3, 128, 1025):
                for family in range(3):
                    run_sample(candidate, bits, n, family, TRAIN_SEED, 8192)
        rows, raw = paired_measure(candidate, reference, final)
        score, decisions, beats = score_rows(rows, calibration)
        metrics = {'beats_reference': beats, 'beat_spec_workload_passes': decisions,
                   # Equal paired measurements score 100*R/(R+R) = 50.
                   # Required private final fields; never public train feedback.
                   'reference_score': 50.0, 'margin': score - 50.0,
                   **{'mean_' + m: statistics.mean(row['candidate'][m] for row in rows) for m in METRICS},
                   'mean_effective_fpr': statistics.mean(row['candidate']['effective_fpr']['point'] for row in rows),
                   'space_overhead_defined_workloads': sum(row['candidate']['space_overhead_pct']['point'] is not None for row in rows),
                   'paired_measurements': rows,
                   'paired_samples': raw,
                   'original_query_output': spine,
                   'elapsed_seconds': time.monotonic() - start,
                   'correctness_passed': True}
        return score, score, ('strict PASS' if beats else 'valid; strict FAIL'), metrics
