"""CPU SWE correctness containment; fixed sidecar request bytes stay unchanged.

This library needs an already prepared, trusted test backend. It does not create
containers, authenticate receipts, or turn a parsed report into a platform score.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

from instance_manifest import expected_groups
from serving_eval.swe_patch_disposition import (SWE_CORRECTNESS_REQUEST_BYTES, OVERSIZE_REQUEST_SCHEMA,
                                    correctness_envelope, oversize_request, validate_disposition)

SUPPORTED_HARNESS_VERSION = '4.1.0'
SWE_TEST_TIMEOUT_SECONDS = 1200
from serving_eval.accuracy import DATA_DIR


class CorrectnessRuntimeError(RuntimeError):
    """Unusable model-case output; the surviving CPU owner decides its scope."""


class TrustedCorrectnessError(CorrectnessRuntimeError):
    """CPU fixed-input or harness failure, never a model correctness result."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def load_swe_cases(mode):
    try:
        return _load_swe_cases(mode)
    except Exception as error:
        raise TrustedCorrectnessError("invalid CPU fixed SWE data") from error


def _load_swe_cases(mode):
    raw = (DATA_DIR / 'instances.json').read_bytes()
    provenance = json.loads((DATA_DIR / 'provenance.json').read_text())
    if _sha(raw) != provenance['native_instances_sha256']:
        raise TrustedCorrectnessError('fixed SWE data digest mismatch')
    payload = json.loads(raw)
    if payload.get('schema') != 'frontier.vllm-swe-fixed-data/v1':
        raise TrustedCorrectnessError('invalid fixed SWE data schema')
    rows = payload['instances']
    expected = expected_groups('eval')['swe/public']
    if tuple(row['instance_id'] for row in rows) != expected:
        raise TrustedCorrectnessError('fixed SWE data differs from original case list')
    for row in rows:
        for field in ('FAIL_TO_PASS', 'PASS_TO_PASS'):
            tests = row[field]
            if not isinstance(tests, list) or any(not isinstance(t, str) or not t for t in tests):
                raise TrustedCorrectnessError('invalid original SWE test identity list')
    selected = set(expected_groups(mode)['swe/public'])
    return [row for row in rows if row['instance_id'] in selected]


def _case(instance_id):
    for row in load_swe_cases('eval'):
        if row['instance_id'] == instance_id:
            return row
    raise TrustedCorrectnessError('unexpected SWE instance')


def _harness():
    try:
        import swebench
        from swebench.harness import grading
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        from swebench.harness.test_spec.python import make_eval_script_list_py
    except Exception as exc:
        raise TrustedCorrectnessError('supported SWE correctness harness is unavailable') from exc
    if swebench.__version__ != SUPPORTED_HARNESS_VERSION:
        raise TrustedCorrectnessError('SWE correctness harness version is unsupported')
    return grading, MAP_REPO_VERSION_TO_SPECS, make_eval_script_list_py


def build_test_request(instance_id, candidate_patch):
    if not isinstance(candidate_patch, str):
        raise CorrectnessRuntimeError('SWE prediction patch must be explicit text, including empty no-op')
    try:
        candidate_patch.encode()
    except UnicodeError as error:
        raise CorrectnessRuntimeError('SWE prediction patch has invalid text encoding') from error
    return _build_test_request(instance_id, candidate_patch)


def _build_test_request(instance_id, candidate_patch):
    row = _case(instance_id)
    _, specs, make_script = _harness()
    try:
        commands = make_script(row, specs[row['repo']][row['version']], 'testbed', '/testbed', row['base_commit'], row['test_patch'])
        if not isinstance(commands, list) or any(not isinstance(command, str) for command in commands):
            raise ValueError('fixed SWE script generator returned malformed commands')
        script = '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + commands) + '\n'
    except Exception as exc:
        raise TrustedCorrectnessError('fixed SWE case lacks a supported original test script') from exc
    request = {
        'schema': 'frontier.vllm-swe-correctness-request/v1',
        'instance_id': instance_id,
        'base_commit': row['base_commit'],
        'candidate_patch': candidate_patch,
        'candidate_patch_sha256': _sha(candidate_patch.encode()),
        'test_patch_sha256': _sha(row['test_patch'].encode()),
        'eval_script': script,
        'eval_script_sha256': _sha(script.encode()),
        'case_sha256': _json_sha(row),
        'harness_version': SUPPORTED_HARNESS_VERSION,
        'timeout_seconds': SWE_TEST_TIMEOUT_SECONDS,
        'protocol': 'Fresh matching testbed; apply candidate patch first, then run exact eval_script and retain combined stdout/stderr. Test-script setup remains outside measured conversation latency.',
    }
    request['request_sha256'] = _json_sha(request)
    return request


def build_correctness_request(instance_id, candidate_patch, *, identity, variant):
    """Keep oversized model output out of the bounded correctness transport."""
    request = build_test_request(instance_id, candidate_patch)
    size = len(correctness_envelope(request, identity, variant))
    if size <= SWE_CORRECTNESS_REQUEST_BYTES:
        return request
    empty = build_test_request(instance_id, '')
    if (not candidate_patch or len(correctness_envelope(empty, identity, variant))
            > SWE_CORRECTNESS_REQUEST_BYTES):
        raise TrustedCorrectnessError('fixed correctness instructions exceed the transport bound')
    return oversize_request(instance_id, candidate_patch, identity=identity, variant=variant,
                            original_request_bytes=size)


def validate_oversize_request(request):
    try:
        evidence = validate_disposition(request)
    except ValueError as error:
        raise TrustedCorrectnessError(str(error)) from error
    empty = build_test_request(request['instance_id'], '')
    empty_size = len(correctness_envelope(empty, request['correctness_identity'], request['variant']))
    # Both patch-related hashes always have 64 ASCII characters. Only the JSON
    # string occupied by candidate_patch changes the request's encoded length.
    expected = empty_size - len(json.dumps('').encode()) + request['candidate_patch_json_bytes']
    if empty_size > SWE_CORRECTNESS_REQUEST_BYTES or request['original_request_bytes'] != expected:
        raise TrustedCorrectnessError('oversize disposition differs from fixed instruction envelope')
    return evidence


def _validate_model_receipt_shape(receipt, *, control_receipt=False):
    if not isinstance(receipt, dict):
        raise CorrectnessRuntimeError('model correctness receipt is not an object')
    for field in ('log', 'patch_apply_log'):
        if isinstance(receipt.get(field), str):
            try:
                receipt[field].encode()
            except UnicodeError as error:
                raise CorrectnessRuntimeError('model receipt has invalid text encoding') from error
    for field in ('testbed', 'stage_exit_codes'):
        if field in receipt and not isinstance(receipt[field], dict):
            raise CorrectnessRuntimeError('model receipt has a malformed ' + field)
    if any(not isinstance(key, str) for key in receipt.get('stage_exit_codes', {})):
        raise CorrectnessRuntimeError('model receipt has malformed stage names')
    for field in ('candidate_syntax_checks', 'candidate_syntax_evidence'):
        if field in receipt and (not isinstance(receipt[field], list)
                or any(not isinstance(item, dict) for item in receipt[field])):
            raise CorrectnessRuntimeError('model receipt has malformed ' + field)
    if not control_receipt and 'startup_control' in receipt:
        control = receipt['startup_control']
        if not isinstance(control, dict) or ('reset' in control and not isinstance(control['reset'], dict)):
            raise CorrectnessRuntimeError('model receipt has malformed startup control')
        if 'receipt' in control:
            _validate_model_receipt_shape(control['receipt'], control_receipt=True)


def parse_test_receipt(request, receipt):
    if request.get('schema') == OVERSIZE_REQUEST_SCHEMA:
        evidence = validate_oversize_request(request)
        base = _case(request['instance_id'])['base_commit']
        _validate_model_receipt_shape(receipt)
        if (not isinstance(receipt, dict) or receipt.get('instance_id') != request['instance_id']
                or receipt.get('request_sha256') != request['request_sha256']
                or receipt.get('execution') != 'candidate_patch_oversize'
                or receipt.get('oversize_evidence') != evidence
                or any(receipt.get(key) is not False for key in
                       ('candidate_patch_applied', 'test_patch_applied', 'tests_started', 'tests_completed'))
                or receipt.get('testbed', {}).get('base_commit') != base
                or receipt.get('testbed', {}).get('content_matches_fixed_base') is not True
                or receipt.get('testbed', {}).get('fresh_worktree') is not True):
            raise CorrectnessRuntimeError('oversize receipt lacks its bound disposition or prepared testbed')
        return {'instance_id': request['instance_id'], 'correct': False,
                'reason': 'candidate_patch_oversize', 'request_sha256': request['request_sha256'],
                'oversize_evidence': evidence}
    if request != build_test_request(request.get('instance_id'), request.get('candidate_patch')):
        raise TrustedCorrectnessError('CPU SWE request differs from fixed test instructions')
    _validate_model_receipt_shape(receipt)
    if not isinstance(receipt, dict) or receipt.get('request_sha256') != request['request_sha256']:
        raise CorrectnessRuntimeError('SWE correctness receipt is missing or belongs to another request')
    if receipt.get('instance_id') != request['instance_id']:
        raise CorrectnessRuntimeError('SWE correctness instance identity mismatch')
    execution = receipt.get('execution')
    if execution == 'candidate_patch_rejected':
        if receipt.get('candidate_patch_applied') is not False or not isinstance(receipt.get('patch_apply_log'), str) or not receipt['patch_apply_log']:
            raise CorrectnessRuntimeError('candidate patch rejection lacks its application evidence')
        return {'instance_id': request['instance_id'], 'correct': False, 'reason': execution,
                'request_sha256': request['request_sha256'], 'patch_apply_log_sha256': _sha(receipt['patch_apply_log'].encode())}
    if execution == 'setup_failed':
        raise CorrectnessRuntimeError('original SWE setup failed; dependency/runtime attribution is required')
    if execution != 'complete':
        raise CorrectnessRuntimeError('SWE test execution failed or needs candidate/platform attribution')
    if not all(receipt.get(field) is True for field in ('candidate_patch_applied', 'test_patch_applied', 'tests_started', 'tests_completed')):
        raise CorrectnessRuntimeError('SWE backend did not complete patch/test stages')
    log = receipt.get('log')
    if not isinstance(log, str) or not log:
        raise CorrectnessRuntimeError('SWE test backend returned no original test log')
    row = _case(request['instance_id'])
    grading, _, _ = _harness()
    spec = SimpleNamespace(**row)
    with tempfile.TemporaryDirectory(prefix='frontier-vllm-correctness-') as temp:
        log_path = Path(temp) / 'test_output.txt'
        log_path.write_text(log)
        try:
            statuses, found = grading.get_logs_eval(spec, str(log_path))
            if not isinstance(statuses, dict) or type(found) is not bool:
                raise ValueError('CPU SWE log grader returned an invalid shape')
        except Exception as error:
            raise CorrectnessRuntimeError('CPU SWE log grader failed') from error
        if not found or not statuses:
            evidence = _attributed_syntax_failure(receipt, log)
            if evidence is not None:
                return {'instance_id': row['instance_id'], 'correct': False,
                        'reason': 'candidate_syntax_error', 'syntax_evidence': evidence,
                        'request_sha256': request['request_sha256'], 'test_log_sha256': _sha(log.encode())}
            controlled = _attributed_startup_failure(request, receipt)
            if controlled is not None:
                return controlled
            raise CorrectnessRuntimeError('original harness did not find usable test execution output')
        try:
            reports = grading.get_eval_report(spec, {'instance_id': row['instance_id'], 'model_name_or_path': 'vllm-serving-episode', 'model_patch': request['candidate_patch']}, str(log_path), include_tests_status=True)
            if not isinstance(reports, dict):
                raise ValueError('CPU SWE report grader returned an invalid shape')
        except Exception as error:
            raise CorrectnessRuntimeError('CPU SWE report grader failed') from error
    report = reports.get(row['instance_id'])
    if not isinstance(report, dict) or type(report.get('resolved')) is not bool:
        raise CorrectnessRuntimeError('CPU SWE report grader returned no valid instance-bound result')
    if not report.get('patch_successfully_applied'):
        raise CorrectnessRuntimeError('original harness did not find successful patch/test output')
    # Missing tests remain failures under the original harness. A status (such
    # as a skipped gold test) omitted from both lists needs explicit review; it
    # cannot disappear from the expected denominator and create a passing report.
    if not isinstance(report.get('tests_status'), dict):
        raise CorrectnessRuntimeError('CPU SWE report lacks structured test statuses')
    for field in ('FAIL_TO_PASS', 'PASS_TO_PASS'):
        group = report['tests_status'].get(field)
        if (not isinstance(group, dict) or any(not isinstance(group.get(name), list)
                or any(not isinstance(test, str) for test in group[name])
                for name in ('success', 'failure'))):
            raise CorrectnessRuntimeError('CPU SWE report returned malformed test status lists')
        observed = group['success'] + group['failure']
        if sorted(observed) != sorted(row[field]):
            raise CorrectnessRuntimeError('SWE report omitted an expected test disposition')
    return {'instance_id': row['instance_id'], 'correct': report['resolved'],
            'request_sha256': request['request_sha256'], 'test_log_sha256': _sha(log.encode()),
            'harness_version': SUPPORTED_HARNESS_VERSION, 'report': report,
            'missing_test_statuses': [test for field in ('FAIL_TO_PASS', 'PASS_TO_PASS') for test in row[field] if test not in statuses]}


def startup_failure_evidence(receipt):
    """Select a narrow startup exception in a modified, previously valid file."""
    from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT

    if receipt.get('execution') != 'complete' or not all(receipt.get(key) is True for key in
            ('candidate_patch_applied', 'test_patch_applied', 'tests_started', 'tests_completed')):
        return None
    exits = receipt.get('stage_exit_codes', {})
    setup = [value for key, value in exits.items() if key.startswith('setup_')]
    if (not setup or any(value != 0 for value in setup)
            or any(exits.get(key) != 0 for key in ('test_reset', 'test_patch', 'test_cleanup'))
            or exits.get('tests') in (None, 0)):
        return None
    log = receipt.get('log', '')
    if START_TEST_OUTPUT not in log or END_TEST_OUTPUT not in log:
        return None
    output = log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    error = re.search(r'^(TypeError|NameError): [^\n]{1,512}$', output, re.MULTILINE)
    if error is None:
        return None
    frames = re.findall(r'File "/testbed/([^"\n]+)", line ([0-9]+)', output)
    if not frames:
        return None
    path, line = frames[-1]
    if Path(path).is_absolute() or '..' in Path(path).parts:
        return None
    for check in receipt.get('candidate_syntax_checks', []):
        if (check.get('path') == path and check.get('baseline_exists') is True
                and check.get('baseline_compiles') is True and check.get('patched_compiles') is True
                and all(isinstance(check.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', check[key])
                        for key in ('baseline_sha256', 'patched_sha256'))
                and check.get('baseline_sha256') != check.get('patched_sha256')):
            return {'path': path, 'lineno': int(line), 'error_type': error.group(1),
                    'baseline_sha256': check.get('baseline_sha256'),
                    'patched_sha256': check.get('patched_sha256')}
    return None


def needs_startup_control(request, receipt):
    if startup_failure_evidence(receipt) is None:
        return False
    grading, _, _ = _harness()
    with tempfile.TemporaryDirectory(prefix='frontier-vllm-startup-check-') as temp:
        path = Path(temp) / 'test_output.txt'
        path.write_text(receipt['log'])
        try:
            statuses, found = grading.get_logs_eval(SimpleNamespace(**_case(request['instance_id'])), str(path))
        except Exception as error:
            raise CorrectnessRuntimeError('CPU SWE log grader failed') from error
    return not found or not statuses


def _attributed_startup_failure(request, receipt):
    evidence = startup_failure_evidence(receipt)
    control = receipt.get('startup_control')
    if evidence is None or not isinstance(control, dict) or control.get('state') != 'complete':
        return None
    if (control.get('schema') != 'frontier.vllm-fixed-base-startup-control/v1'
            or control.get('request_sha256') != request['request_sha256']
            or control.get('variant') not in ('vanilla', 'reference', 'candidate')):
        return None
    elapsed = control.get('elapsed_seconds')
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or not 0 < elapsed <= request['timeout_seconds']):
        return None
    baseline = control.get('receipt')
    if not isinstance(baseline, dict) or baseline.get('fixed_base_control') is not True:
        return None
    expected_base = _case(request['instance_id'])['base_commit']
    for identity in (receipt.get('testbed', {}), baseline.get('testbed', {})):
        if (identity.get('base_commit') != expected_base or identity.get('content_matches_fixed_base') is not True
                or identity.get('fresh_worktree') is not True):
            return None
    if control.get('reset', {}).get('head') != baseline.get('testbed', {}).get('observed_head'):
        return None
    baseline_request = build_test_request(request['instance_id'], '')
    exits = baseline.get('stage_exit_codes', {})
    setup = [value for name, value in exits.items() if name.startswith('setup_')]
    if (not setup or any(value != 0 for value in setup)
            or any(exits.get(key) != 0 for key in ('test_reset', 'test_patch', 'test_cleanup'))
            or not isinstance(exits.get('tests'), int) or isinstance(exits.get('tests'), bool)):
        return None
    # A control cannot recursively supply another control as its own evidence.
    if 'startup_control' in baseline:
        return None
    try:
        parsed = parse_test_receipt(baseline_request, baseline)
    except TrustedCorrectnessError:
        raise
    except CorrectnessRuntimeError:
        return None
    if 'report' not in parsed or parsed.get('missing_test_statuses'):
        return None
    return {'instance_id': request['instance_id'], 'correct': False,
            'reason': 'candidate_test_startup_error', 'startup_evidence': evidence,
            'request_sha256': request['request_sha256'], 'test_log_sha256': _sha(receipt['log'].encode()),
            'fixed_base_control_sha256': _json_sha(baseline),
            'fixed_base_test_log_sha256': parsed['test_log_sha256']}


def _attributed_syntax_failure(receipt, log):
    from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT

    if START_TEST_OUTPUT not in log or END_TEST_OUTPUT not in log:
        return None
    test_output = log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    exits = receipt.get('stage_exit_codes', {})
    if exits.get('tests') in (None, 0) or exits.get('test_cleanup') != 0:
        return None
    setup = [value for name, value in exits.items() if name.startswith('setup_')]
    if not setup or any(value != 0 for value in setup):
        return None
    for evidence in receipt.get('candidate_syntax_evidence', []):
        if not isinstance(evidence, dict) or evidence.get('baseline_compiles') is not True:
            continue
        path, line, error = evidence.get('path'), evidence.get('lineno'), evidence.get('error_type')
        if (not isinstance(path, str) or not path.endswith('.py') or path.startswith('/')
                or '..' in Path(path).parts or not isinstance(line, int) or line <= 0
                or error not in ('SyntaxError', 'IndentationError', 'TabError')
                or any(not re.fullmatch('[0-9a-f]{64}', str(evidence.get(key, '')))
                       for key in ('baseline_sha256', 'patched_sha256'))):
            continue
        # Both a baseline-vs-patch compile proof in the instance interpreter and
        # the same file/line/error in actual test output are required. Dependency
        # errors and unrelated collection failures still need runtime attribution.
        pattern = (r'File "/testbed/' + re.escape(path) + r'", line ' + str(line)
                   + r'\b[^\n]*\n[\s\S]{0,2000}?\n' + re.escape(error) + ':')
        if re.search(pattern, test_output):
            return evidence
    return None


def verify_swe_case(instance_id, candidate_patch, backend):
    if backend is None or not callable(getattr(backend, 'run_correctness', None)):
        raise CorrectnessRuntimeError('prepared real SWE correctness backend is unavailable')
    request = build_test_request(instance_id, candidate_patch)
    try:
        receipt = backend.run_correctness(request)
    except Exception as exc:
        raise CorrectnessRuntimeError('prepared SWE correctness backend failed') from exc
    result = parse_test_receipt(request, receipt)
    return {'request': request, 'receipt': receipt, 'result': result}


def summarize_swe_correctness(mode, instance_results):
    expected = expected_groups(mode)['swe/public']
    records = {}
    for result in instance_results:
        ident = result.get('instance_id')
        if ident not in expected or ident in records or not isinstance(result.get('correct'), bool):
            raise CorrectnessRuntimeError('incorrect or duplicate SWE result identity')
        records[ident] = result
    if set(records) != set(expected):
        raise CorrectnessRuntimeError('SWE correctness is missing an original instance')
    return {'expected_instances': len(expected), 'correct_instances': sum(records[i]['correct'] for i in expected),
            'accuracy': sum(records[i]['correct'] for i in expected) / len(expected), 'instances': [records[i] for i in expected]}
