"""Native CPU evaluator; exact pinned reference, strict per-row conjunction.

The three performance metrics require strict per-workload wins over the exact
reference. The two accuracy metrics are certified non-inferiority guards against
preregistered reference-only anchors (accuracy_metrics.py). The trusted harness
is a separate translation unit from the submitted header. Raw paired samples
remain in private evidence; no reference values reach train.
"""
from __future__ import annotations
import json
import hashlib
import math
import os
from pathlib import Path
import resource
import signal
import shutil
import statistics
import subprocess
import tempfile
import time

from submission_patch import apply_submission_patch
from accuracy_metrics import (CalibrationRequired, RuntimeQualificationRequired, METRIC_NAMES, RULE,
                              SAMPLE_ALPHA, non_inferior, reference_anchors, summarize_accuracy)

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
SUBMISSION = {'submission': {'allow_paths': ['include/'], 'max_changed_files': 32,
                            'max_patch_bytes': 524288, 'repo_root': '/app'}}
MANIFEST_SCHEMA = 'binary-fuse-calibration-manifest/v3'
PROFILE_SCHEMA = 'binary-fuse-reference-statistical-calibration/v3'
RUNTIME_SCHEMA = 'binary-fuse-runtime-qualification/v2'

class InvalidCandidate(Exception):
    pass

class InvalidReference(Exception):
    """A failed author baseline is not evidence of solver difficulty."""
    pass

def limits():
    resource.setrlimit(resource.RLIMIT_AS, (4*2**30, 4*2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*2**20, 8*2**20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

def command(args, cwd, timeout=30):
    # File-backed output is bounded by RLIMIT_FSIZE; no unbounded communicate buffer.
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(cwd), 'TMPDIR': str(cwd),
           'LANG': 'C', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'GIT_CEILING_DIRECTORIES': str(Path(cwd).parent)}
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=out, stderr=err,
                                    start_new_session=True, preexec_fn=limits)
            try:
                proc.wait(timeout=timeout)
            finally:
                # Reap the whole owned group even if the compiler/solution forks.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        except subprocess.TimeoutExpired as exc:
            raise InvalidCandidate('bounded command timed out') from exc
        out.seek(0); err.seek(0)
        stdout, stderr = out.read(65536), err.read(4096)
        if proc.returncode:
            raise InvalidCandidate('command failed: ' + stderr.decode('utf-8', 'replace')[-1000:])
        return stdout.decode('utf-8', 'strict')

def compile_runner(repo, directory):
    binary = directory / 'measure'
    flags = ['cc', '-std=c99', '-O3', '-fno-lto', '-DNDEBUG']
    # Compiler is fixed, helpers can be header-only under the usable include/ surface.
    command(flags + ['-I', str(repo / 'include'), '-c', str(TASK / 'bridge.c'),
                     '-o', str(directory / 'bridge.o')], directory)
    command(flags + ['-c', str(TASK / 'harness.c'), '-o', str(directory / 'harness.o')], directory)
    command(['cc', str(directory / 'bridge.o'), str(directory / 'harness.o'), '-lm', '-o', str(binary)], directory)
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
    text = command([str(binary), str(bits), str(n), str(family), str(seed), str(q)], binary.parent)
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

def measurement_binding():
    digest = hashlib.sha256()
    for name in ('harness.c', 'bridge.c', 'evaluator.py', 'accuracy_metrics.py', 'beat_spec.json',
                 'config.yaml', 'reference/include/binaryfusefilter.h', 'reference/include/xorfilter.h'):
        body = (TASK / name).read_bytes()
        digest.update(name.encode() + b'\0' + len(body).to_bytes(8, 'big') + body)
    return 'sha256:' + digest.hexdigest()

def _sha256_file(path):
    return 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()

def _digest_string(value):
    return (isinstance(value, str) and len(value) == 71 and value.startswith('sha256:') and
            all(c in '0123456789abcdef' for c in value[7:]))

def load_statistical_calibration(*, require_runtime=True):
    """Validate retained reference-only statistics and their preregistered anchors.

    measurement_binding() hashes the exact measurement sources; the private
    profile and the production runtime receipt must bind that identity. The
    author may inspect anchors without executing a candidate. evaluate() always
    uses require_runtime=True and fails before opening a patch. Private point
    values stay in the verifier-only reference directory.
    """
    binding = measurement_binding()
    manifest = json.loads((TASK / 'statistical_calibration.json').read_text())
    private_profile = TASK / 'reference/statistical-calibration.json'
    if (manifest.get('schema') != MANIFEST_SCHEMA or
            manifest.get('reference_statistics_status') != 'CALIBRATED_REFERENCE_ONLY' or
            manifest.get('measurement_sha256') != binding or
            not private_profile.is_file() or
            manifest.get('statistical_profile_sha256') != _sha256_file(private_profile)):
        raise CalibrationRequired('reference statistics are missing or their binding differs')
    profile = json.loads(private_profile.read_text())
    if (profile.get('schema') != PROFILE_SCHEMA or
            profile.get('status') != 'CALIBRATED_REFERENCE_ONLY' or
            profile.get('rule') != RULE or
            profile.get('measurement_sha256') != binding or
            profile.get('successor_solver_results_seen') is not False or
            profile.get('sample_alpha') != SAMPLE_ALPHA):
        raise CalibrationRequired('successor requires exact-source reference-only calibration')
    rounds = profile.get('reference_repeat_summaries')
    anchors = reference_anchors(rounds, workload_ids(True))
    evidence = profile.get('reference_evidence_sha256', [])
    runtime_receipt = profile.get('reference_runtime_receipt_sha256')
    if (not _digest_string(runtime_receipt) or
            len(evidence) != len(rounds) or len(set(evidence)) != len(evidence) or
            any(not _digest_string(h) for h in evidence) or
            profile.get('anchors') != anchors):
        raise CalibrationRequired('reference evidence or recomputed anchors disagree')
    if require_runtime:
        runtime = manifest.get('production_runtime', {})
        receipt_path = TASK / 'reference/runtime-qualification.json'
        if (runtime.get('status') != 'QUALIFIED_EXACT_RUNTIME' or
                not receipt_path.is_file() or
                runtime.get('receipt_sha256') != _sha256_file(receipt_path)):
            raise RuntimeQualificationRequired('reference statistics calibrated; exact production runtime pending')
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('schema') != RUNTIME_SCHEMA or
                receipt.get('status') != 'QUALIFIED_EXACT_RUNTIME' or
                receipt.get('measurement_sha256') != binding or
                receipt.get('full_entrypoint_train_final') is not True or
                receipt.get('answering_seconds') != 18000 or
                receipt.get('train_calls') != 100):
            raise RuntimeQualificationRequired('production runtime evidence is stale or incomplete')
        for name, maximum in (('final_wall_seconds', 180),
                              ('train_pool_fit_seconds', 12000),
                              ('peak_memory_bytes', 4*2**30)):
            value = receipt.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise RuntimeQualificationRequired('production runtime bound missing or exceeded: ' + name)
    return anchors

def original_entrypoint(repo):
    command(['make', '-j1', 'query', 'CC=cc'], repo)
    output = command([str(repo / 'query')], repo)
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

def evaluate(solution_path):
    start = time.monotonic()
    calibration = load_statistical_calibration()
    return _evaluate_submission(solution_path, start, calibration)

def evaluate_reference(solution_path):
    """Author-only packaging/runtime qualification of the exact private restore.

    The production entrypoint always calls evaluate(). This entrypoint cannot
    evaluate another submission and does not create or assert runtime evidence.
    It exercises the same patch, compilation, workload and scoring implementation.
    """
    start = time.monotonic()
    patch = Path(solution_path)
    reference = TASK / 'reference.patch'
    if (not patch.is_file() or patch.stat().st_size != reference.stat().st_size or
            patch.read_bytes() != reference.read_bytes()):
        raise InvalidReference('author qualification requires the exact private reference patch')
    calibration = load_statistical_calibration(require_runtime=False)
    return _evaluate_submission(solution_path, start, calibration)

def _evaluate_submission(solution_path, start, calibration):
    final = os.environ.get('FRONTIER_SUBMISSION_ROLE') == 'final'
    pristine = TASK / 'harbor/app'
    # In a checked-out native tree the public source has a single canonical copy.
    if not pristine.exists():
        pristine = TASK.parent / 'agent-base/frontier/task/repo'
    applied = apply_submission_patch(solution_path, TASK, SUBMISSION, pristine=pristine)
    if not applied.ok:
        return failure(applied.message, start)
    try:
        with tempfile.TemporaryDirectory(prefix='binary-fuse-measure-') as tmp:
            root = Path(tmp)
            cdir, rdir = root / 'candidate', root / 'reference'
            cdir.mkdir(); rdir.mkdir()
            candidate = compile_runner(applied.workdir, cdir)
            try:
                reference = compile_runner(TASK / 'reference', rdir)
            except InvalidCandidate as exc:
                raise InvalidReference(str(exc)) from exc
            spine = original_entrypoint(applied.workdir)
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
    except InvalidCandidate as exc:
        return failure(str(exc), start)
    finally:
        if applied.cleanup_root:
            shutil.rmtree(applied.cleanup_root)

def failure(message, start):
    return 0.0, 0.0, 'invalid candidate: ' + message[:1000], {'all_final_metrics': {
        'candidate_failure': {'message': message[:2000]}, 'runtime': time.monotonic()-start}}
