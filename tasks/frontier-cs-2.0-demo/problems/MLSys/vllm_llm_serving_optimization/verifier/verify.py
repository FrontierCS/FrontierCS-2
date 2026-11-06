"""Standalone CPU Verifier of vllm_llm_serving_optimization (task.toml [verifier] command); the
Driver owns credentials, GPUs and the final clock. Scoring note (statement: repo/README.md):

raw_score = score_unbounded = 0.8*SWE + 0.2*BFCL on 0-100 (judge/continuous.py). Per load group,
r_i = vanilla latency (median of 3 captures) / candidate latency, 1/8 for a candidate-caused
missing result; S = exp(mean log r_i); final S_w = sqrt(S_public*S_hidden); L(S) = 0 if S<=1
else 100*log S/(log 10+log S); times quality = 1 if the pooled correct-fraction drop vs vanilla
is <= 0.05 absolute or relative, else 0.05/relative_drop. A rule violation is a scored 0 with
metrics.rejected (judge/submission.py categories).
Per-group comparisons against the optimized reference (judge/fixed_targets.py), each workload
= one benchmark x load group (train: swe/public, bfcl/public; final: x public, hidden):
- paired_timing: mean(log ref_i - log cand_i) > 0, ref_i = fastest of 3 captures; higher better.
- p90 / p95 / p99: candidate nearest-rank tail latency strictly below the smallest capture
  value of that percentile; lower better.
- correctness: correct count >= median(R1,R2,R3) - max(1, ceil(0.05*N)); higher better.
A group also needs complete usable candidate results. beats_reference = every comparison
passes in every group (conjunction). beat_spec_progress counts the comparisons per metric.
Provenance: docs/task-authoring/vllm_llm_serving_optimization/provenance/beat_spec.json.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time

sys.path.insert(0,str(Path(__file__).parents[1]/'judge'))
from evidence import write_evidence
from safe_error_frames import safe_error_frames
from instance_manifest import expected_groups
from batch_coordinator import FINAL_HIDDEN_JPS
from cpu_reference_verdict import TrustedMeasurementError
from parallel_phase import PhaseInfrastructureError, PhaseQuotaError, run_parallel_phase
from runtime_config import canonical_digest, phase_limits
from runtime_plan import validate_plan
from submission import SubmissionInfrastructureError, inspect_submission
from fixed_target_release import load_release


def extract_patch(archive, destination, maximum):
    if archive.is_symlink() or not archive.is_file() or not 0<archive.stat().st_size<=maximum:
        raise ValueError('invalid submission archive')
    seen=set()
    patch=None
    with tarfile.open(archive,mode='r:') as tar:
        for member in tar:
            if member.name in seen:
                raise ValueError('duplicate payload member')
            seen.add(member.name)
            if member.name=='payload' and member.isdir():
                continue
            if member.name!='payload/solution.patch' or not member.isfile() or not 0<=member.size<=1500000:
                raise ValueError('archive must contain only the original regular patch entry')
            stream=tar.extractfile(member)
            if stream is None:
                raise ValueError('patch member is unreadable')
            patch=stream.read(1500001)
            if len(patch)!=member.size:
                raise ValueError('incomplete patch archive')
    if patch is None:
        raise ValueError('patch entry is absent')
    if patch and not patch.decode('utf-8', errors='strict').strip():
        raise ValueError('nonempty patch entry is whitespace-only')
    destination.write_bytes(patch)
    return patch


def submission_input(archive, destination, maximum, clean_source):
    """Contain all candidate input failures before any phase or GPU work.

    Only explicit clean-source infrastructure failures escape this boundary.
    Unknown parser, decoding and patch inspection exceptions default to invalid
    submission; their messages and candidate bytes never enter public feedback.
    """
    identity = {}
    try:
        if archive.is_symlink() or not archive.is_file() or not 0 < archive.stat().st_size <= maximum:
            raise ValueError('invalid submission archive')
        with archive.open('rb') as stream:
            raw = stream.read(maximum + 1)
        if not 0 < len(raw) <= maximum:
            raise ValueError('invalid submission archive size')
        identity = {'payload_sha256': 'sha256:' + hashlib.sha256(raw).hexdigest(),
                    'payload_bytes': len(raw)}
        patch = extract_patch(archive, destination, maximum)
        inspected = inspect_submission(destination, clean_source)
        if not isinstance(inspected, dict) or type(inspected.get('valid_submission')) is not bool:
            raise ValueError('invalid patch inspection response')
        inspected = dict(inspected)
        if not inspected['valid_submission']:
            inspected.update(disposition='invalid_submission', origin='submission_input_boundary')
        inspected.update(identity)
        json.dumps(inspected, allow_nan=False)
        return patch, inspected
    except SubmissionInfrastructureError:
        raise
    except Exception as error:
        return None, {'valid_submission': False, 'disposition': 'invalid_submission',
                      'origin': 'submission_input_boundary', 'category': 'malformed',
                      'error_type': type(error).__name__,
                      'message': 'Submission input is invalid.', **identity}


def dataset_selector(path, mode):
    value=json.loads(path.read_bytes())
    expected=expected_groups(mode)
    correct={'schema':'frontier.vllm-dataset/v1','problem_id':'vllm_llm_serving_optimization','mode':mode,
             'swe_instance_ids':list(expected['swe/public']),
             'bfcl_instance_ids':list(expected['bfcl/public']),
             'rate_groups':['public'] if mode=='train' else ['public','hidden']}
    if value!=correct:
        raise ValueError('selected dataset differs from the complete original phase')


def beat_spec_progress(result, mode):
    """Count existing predicates; None means the complete breakdown is absent.

    This is an operator-only projection, not a second scoring implementation.
    Missing result evidence must not become failed scientific comparisons.
    Candidate omissions already evaluated by reference_verdict are different:
    those have explicit booleans and retain their full workload denominator.
    """
    expected = expected_groups(mode)
    metric_names = ('paired_timing', 'p90', 'p95', 'p99', 'correctness')
    reference = result.get('reference_verdict')
    validity = result.get('validity')
    if not isinstance(reference, dict) or not isinstance(validity, dict):
        return None
    groups, gates = reference.get('groups'), validity.get('groups')
    no_flip = result.get('bfcl_flip_gate_enabled') is False
    bfcl_groups = (set() if no_flip else
                   {key for key in expected if key.startswith('bfcl/')})
    if not isinstance(groups, dict) or not isinstance(gates, dict):
        return None
    if set(groups) - set(expected) or set(gates) - bfcl_groups:
        raise ValueError('unexpected operator result group')
    if set(groups) != set(expected) or set(gates) != bfcl_groups:
        return None
    values = {name: [] for name in (metric_names if no_flip else (*metric_names, 'bfcl_regressions'))}
    group_passes = []
    for key, ids in expected.items():
        group = groups[key]
        if not isinstance(group, dict) or not isinstance(group.get('comparisons'), dict):
            return None
        comparisons = group['comparisons']
        if any(type(comparisons.get(name)) is not bool for name in metric_names):
            return None
        if type(group.get('complete')) is not bool or type(group.get('pass')) is not bool:
            return None
        if group.get('expected_instances') != len(ids):
            return None
        passed = group['complete'] and all(comparisons[name] for name in metric_names)
        if group['pass'] != passed:
            raise ValueError('inconsistent operator group conjunction')
        group_passes.append(passed)
        for name in metric_names:
            values[name].append(comparisons[name])
        if key in bfcl_groups:
            gate = gates[key]
            if not isinstance(gate, dict) or type(gate.get('pass')) is not bool:
                return None
            if gate.get('expected_instances') != len(ids):
                return None
            values['bfcl_regressions'].append(gate['pass'])
    reference_pass, valid = all(group_passes), all(values.get('bfcl_regressions', ()))
    aggregates = ((reference.get('pass'), reference_pass), (validity.get('pass'), valid),
                  (result.get('valid'), valid), (result.get('pass'), valid and reference_pass))
    if any(type(observed) is not bool for observed, _ in aggregates):
        return None
    if any(observed != computed for observed, computed in aggregates):
        raise ValueError('inconsistent operator aggregate conjunction')
    progress = {}
    for name, outcomes in values.items():
        total, passed = len(outcomes), sum(outcomes)
        failed = total - passed
        progress[name] = {'passed': passed, 'failed': failed, 'total': total,
                          'pass_percent': round(100 * passed / total, 2),
                          'fail_percent': round(100 * failed / total, 2)}
    return progress


# task.toml [feedback], in order: the only evaluator metrics a train result returns.
FEEDBACK = (
    'swe_public_geometric_mean_latency_seconds', 'swe_public_speedup_over_vanilla',
    'swe_public_p90_latency_seconds', 'swe_public_p95_latency_seconds',
    'swe_public_p99_latency_seconds', 'swe_public_correct',
    'bfcl_public_geometric_mean_latency_seconds', 'bfcl_public_speedup_over_vanilla',
    'bfcl_public_p90_latency_seconds', 'bfcl_public_p95_latency_seconds',
    'bfcl_public_p99_latency_seconds', 'bfcl_public_correct',
)


def train_feedback(result):
    """Whitelist public candidate aggregates, never copy private result objects."""
    metrics = {}
    groups = result.get('reference_verdict', {}).get('groups', {})
    continuous = result.get('continuous', {})
    for workload in ('swe', 'bfcl'):
        group = groups.get(workload + '/public', {})
        prefix = workload + '_public_'
        count = group.get('candidate_correct')
        if type(count) is int and 0 <= count <= len(expected_groups('train')[workload + '/public']):
            metrics[prefix + 'correct'] = count
        timing = {'geometric_mean_latency_seconds': group.get('candidate_geometric_mean_latency_seconds')}
        tails = group.get('candidate_tails')
        if isinstance(tails, dict):
            timing.update({p + '_latency_seconds': tails.get(p) for p in ('p90', 'p95', 'p99')})
        paired = continuous.get(workload, {}).get('groups', {}).get('public', {})
        if paired.get('timing_complete') is True:
            timing['speedup_over_vanilla'] = paired.get('speedup')
        for name, value in timing.items():
            if type(value) in (int, float) and math.isfinite(value) and value > 0:
                metrics[prefix + name] = value
    if not set(metrics) <= set(FEEDBACK):
        raise ValueError('train feedback outside task.toml [feedback]')
    progress = beat_spec_progress(result, 'train')
    if progress is not None:
        metrics['beat_spec_progress'] = {name: progress[name] for name in
            ('paired_timing', 'p90', 'p95', 'p99', 'correctness')}
    return metrics


def result_projection(result,mode):
    """The platform result in the shared contract (tasks/shared/verifier/fcs2_verifier.py):
    metrics carry score_unbounded (the 0-100 score itself), train adds the [feedback] metrics
    and beat_spec_progress, final keeps beats_reference and beat_spec_progress private."""
    score=result.get('score')
    if mode not in {'train','eval'} or isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=100 or type(result.get('pass')) is not bool:
        raise ValueError('invalid complete operator result')
    value={'status':'passed','raw_score':score,'summary':'Verification completed.',
           'metrics':{'score_unbounded':score}}
    if mode=='train':
        value['metrics'].update(train_feedback(result))
    if mode=='eval':
        value['private_metrics']={'beats_reference':int(result['pass'])}
        progress = beat_spec_progress(result, mode)
        if progress is not None:
            value['private_metrics'].update(beat_spec_progress=progress)
        elif result['pass']:
            raise ValueError('passing operator result lacks complete comparison evidence')
    return value


def rejection_projection(category,mode):
    """A submission outside the rules: a scored zero with its shared public category."""
    if mode not in {'train','eval'}:
        raise ValueError('unknown verification mode')
    value={'status':'passed','raw_score':0.0,'summary':'Verification completed.',
           'metrics':{'score_unbounded':0.0,'rejected':category}}
    if mode=='eval':
        value['private_metrics']={'beats_reference':0,'rejected':category}
    return value



def main():
    started=time.monotonic()
    result_path=Path(os.environ['FRONTIER_RESULT_PATH'])
    evidence_path=Path(os.environ['FRONTIER_EVIDENCE_PATH'])
    mode=os.environ['FRONTIER_VERIFICATION_MODE']
    root=Path(tempfile.mkdtemp(prefix='vllm-verifier-',dir='/frontier'))
    evidence=root/'evidence'; evidence.mkdir()
    result={'status':'failed','failure':{'code':'vllm_verifier_runtime_error',
        'message':'Verification could not complete.','retryable':False}}
    try:
        config=json.loads(Path('/opt/frontier/runtime-config.json').read_text())
        plan=validate_plan(json.loads(Path('/opt/frontier/runtime-plan.json').read_text()),config)
        # Missing baselines are trusted infrastructure failures before candidate
        # bytes are inspected or any GPU/testbed can be allocated. No fallback.
        fixed_targets=load_release(Path(__file__).parent, config)
        if mode not in {'train','eval'}:
            raise ValueError('unknown verification mode')
        dataset=Path(os.environ['FRONTIER_DATASET_PATH']); dataset_selector(dataset,mode)
        payload=Path(os.environ['FRONTIER_PAYLOAD_PATH'])
        patch_path=root/'solution.patch'
        patch,inspected=submission_input(payload,patch_path,plan['max_payload_bytes'],
                                       Path('/opt/frontier/vllm-clean'))
        selection={'payload_sha256':inspected.get('payload_sha256'),
                   'patch_sha256':('sha256:'+hashlib.sha256(patch).hexdigest()) if patch is not None else None,
                   'dataset_sha256':'sha256:'+hashlib.sha256(dataset.read_bytes()).hexdigest(),
                   'runtime_projection_sha256':canonical_digest(config),'mode':mode}
        (evidence/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        (evidence/'submission.json').write_text(json.dumps(inspected,indent=2)+'\n')
        if not inspected['valid_submission']:
            result=rejection_projection(inspected['category'],mode)
        else:
            phase=plan['phases'][mode]
            limits=phase_limits(config,mode,requested_gpu_seconds=max(
                cap['gpu_seconds'] for cap in phase['groups'].values()))
            end=started+limits['timeout_seconds']
            # Final runs the public and configured hidden rates as four groups.
            operator=run_parallel_phase(config,plan,mode,environment=os.environ,
                candidate_patch=patch,root=root,evidence=evidence/'phase',deadline=end,
                fixed_targets=fixed_targets,
                hidden_jps=FINAL_HIDDEN_JPS if mode=='eval' else None)
            result=result_projection(operator,mode)
    except PhaseQuotaError as error:
        result={'status':'failed','failure':{'code':'vllm_resource_allocation_exhausted',
            'message':'Declared GPU resource allocation exhausted.', 'retryable':False}}
        try:
            (evidence/'quota-failure.json').write_text(json.dumps({
                'origin':'typed_driver_quota_ledger', 'score_ready':False,
                'automatic_retry':False, 'failed_groups':error.failures},
                allow_nan=False, sort_keys=True)+'\n')
        except Exception:
            pass
    except PhaseInfrastructureError as error:
        result={'status':'failed','failure':{'code':'vllm_platform_infrastructure_error',
            'message':'Owned platform provisioning could not complete.',
            'retryable':not error.cleanup_required}}
        try:
            (evidence/'infrastructure-failure.json').write_text(json.dumps({
                'origin':'typed_driver_provisioning', 'score_ready':False,
                'cleanup_required':error.cleanup_required, 'failed_groups':error.failures},
                allow_nan=False, sort_keys=True)+'\n')
        except Exception:
            pass
    except Exception as error:
        if isinstance(error, TrustedMeasurementError):
            result = {'status': 'failed', 'failure': {
                'code': 'vllm_trusted_measurement_unavailable',
                'message': 'Trusted verification measurements are unavailable.',
                'retryable': True}}
        try:
            (evidence/'failure.json').write_text(json.dumps({'error_type':type(error).__name__,
                'origin':'native_runtime_unclassified','score_ready':False,
                'phase':mode if mode in {'train','eval'} else None,
                'group':None,'variant':None,'frames':safe_error_frames(error),
                **(error.private_classification() if isinstance(error, TrustedMeasurementError) else {})})+'\n')
        except Exception:
            pass
    try:
        maximum=int(os.environ['FRONTIER_MAX_EVIDENCE_BYTES'])
        if maximum<=0: raise ValueError('missing evidence allocation')
        write_evidence(evidence,evidence_path,maximum)
    except Exception:
        result={'status':'failed','failure':{'code':'vllm_evidence_error',
            'message':'Verification evidence could not be retained.','retryable':False}}
    result_path.write_text(json.dumps(result,allow_nan=False)+'\n')
    # The platform wrapper reads structured status only after a zero task exit.
    # Nonzero is reserved for inability to produce the protocol result itself.
    return 0


if __name__=='__main__': raise SystemExit(main())
