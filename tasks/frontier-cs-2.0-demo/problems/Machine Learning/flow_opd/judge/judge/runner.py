"""Verifier-side orchestration for a subsequently qualified private manifest.

This module never provisions hardware. It runs inside the Driver Verifier GPU.
Execution checks validate the runtime report and retained endpoint artifacts.
They are not a second sandbox; platform isolation and actual-trial review
remain the execution and acceptance boundaries.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from .scoring import QualificationError, TrustedEvaluationFailure, aggregate, failure_metrics, reward_ok
from .execution_checks import validate_execution

TASK=Path(__file__).resolve().parents[1]
ENTRYPOINT='scripts/single_node/sd3_opd_example.sh'


def private_file(relative):
    path=(TASK/relative).resolve()
    if not path.is_relative_to(TASK/'judge') or not path.is_file():
        raise QualificationError('Missing private qualification artifact.')
    return path


def checked_json(record):
    path=private_file(record['path'])
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:
        raise QualificationError('Qualification artifact changed after freeze.')
    return json.loads(raw)


def validate_reference(cache,manifest):
    if cache.get('manifest_sha256')!=manifest['identity_sha256']:
        raise QualificationError('Reference uses a different frozen workload.')
    if cache.get('reference_patch_sha256')!=hashlib.sha256((TASK/'reference.patch').read_bytes()).hexdigest():
        raise QualificationError('Reference patch identity mismatch.')
    if cache.get('valid') is not True:
        raise QualificationError('Missing valid reference evidence.')
    return cache


def comparison_context(q, *, final):
    manifest=checked_json(q['suite_manifest'])
    cache=validate_reference(checked_json(q['reference_cache']),manifest)
    if manifest.get('frozen_before_candidates') is not True:
        raise QualificationError('Workload was not frozen before candidate comparisons.')
    selected=['final'] if final else ['development','train_grading']
    rows=[row for row in manifest['workloads'] if row['suite'] in selected]
    grading=[row for row in rows if row['suite']==('final' if final else 'train_grading')]
    ids=[row['id'] for row in grading]
    if not ids: raise QualificationError('Empty private evaluation suite.')
    if len(ids) != len(set(ids)):
        raise QualificationError('Duplicate private workload identity.')
    reference = {i:cache['workloads'][i]['ocr_reward'] for i in ids}
    feedback_reference = [cache['workloads'][row['id']]['ocr_reward']
                          for row in rows if row['suite'] == 'development']
    if (any(not reward_ok(v) for v in reference.values())
            or (not final and (not feedback_reference
                or any(not reward_ok(v) for v in feedback_reference)))):
        raise QualificationError('Invalid frozen reference measurements.')
    return manifest, cache, rows, ids, reference, feedback_reference


def run_evaluation(tree,q,*,final,deadline,model_root):
    manifest,cache,rows,ids,reference,feedback_reference = comparison_context(q, final=final)
    rewards={}; public=[]; public_reference=[]; updates=0; valid=True
    observed_runs={}
    # All workload prompts stay in the private judge side. Runs sharing a
    # training specification share a checkpoint, with independently reset eval RNG.
    run_ids=sorted({row['run'] for row in rows})
    for run_id in run_ids:
        run=manifest['runs'][run_id]
        profile=checked_json(run['profile'])
        run_rows=[row for row in rows if row['run']==run_id]
        profile['evaluation']=[{'prompt':row['prompt'],'seed':row['seed']} for row in run_rows]
        # Retain the canonical dataset loader; only its file location changes.
        evidence_root=Path(os.environ.get('FRONTIER_EVIDENCE_ROOT',tempfile.gettempdir()))
        output=Path(tempfile.mkdtemp(prefix='flow-opd-run-',dir=evidence_root))
        dataset=output/'corpus'; dataset.mkdir()
        training=private_file(run['training_file'])
        shutil.copyfile(training,dataset/'train.txt')
        (dataset/'test.txt').write_text(''.join(row['prompt']+'\n' for row in run_rows))
        profile['dataset']=str(dataset)
        try:
            result=yield profile,output
            observed_runs[run_id]=result
            if any(row['suite']=='development' for row in run_rows):
                updates+=int(result.get('verified_optimizer_updates',0))
            accepted=validate_execution(tree,output,profile,result,deadline=deadline)
            valid=valid and accepted and result.get('complete') is True
            if result.get('verified_optimizer_updates')!=profile['updates']:
                valid=False
            values=result.get('ocr_rewards',[])
            if len(values)!=len(run_rows):
                valid=False
                break
            initial=result.get('initial_evaluation_latent_sha256',[])
            for i,(row,value) in enumerate(zip(run_rows,values)):
                ref=cache['workloads'][row['id']]
                if i>=len(initial) or initial[i]!=ref['initial_latent_sha256']:
                    valid=False
                if row['suite']=='development':
                    public.append(value); public_reference.append(ref['ocr_reward'])
                else: rewards[row['id']]=value
        except (TimeoutError,ValueError,RuntimeError) as error:
            # A launcher or infrastructure error is not an observed candidate
            # rejection. Preserve that distinction instead of fabricating a zero.
            raise TrustedEvaluationFailure('workload_runtime_unclassified') from error
        # Preserve state/evidence on both success and failure; owned cleanup is
        # explicit after review. The outer evaluator deletes only patched source.
    if not valid:
        metrics=failure_metrics(ids,reference=reference,final=final,updates=updates,
                                feedback_reference=feedback_reference,
                                status='invalid_execution_report')
        metrics['private_execution_runs'] = {
            run_id: {'report_present': True, 'complete': report.get('complete') is True,
                     'reward_count': len(report['ocr_rewards'])
                        if isinstance(report.get('ocr_rewards'), list) else None,
                     'reward_values_valid': isinstance(report.get('ocr_rewards'), list)
                        and all(reward_ok(v) for v in report['ocr_rewards'])}
            for run_id, report in observed_runs.items()
        }
        return 0.0,0.0,'Invalid or incomplete execution.',metrics
    score,unbounded,message,metrics=aggregate(
        ids,rewards,reference,candidate_valid=True,reference_valid=True,
        final=final,development_rewards=public,
        development_reference_rewards=public_reference,updates=updates)
    # The native adapter filters public feedback; this trace remains in private
    # evidence so selected payloads can be audited against actual execution.
    metrics['private_execution_runs']=observed_runs
    return score,unbounded,message,metrics
