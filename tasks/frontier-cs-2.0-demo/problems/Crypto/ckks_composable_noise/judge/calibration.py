"""Scaled native calibration successor; no solver or formal-acceptance path.
P/K/V and fail-stop rules are preregistered before native collection. B=1 keeps
exactly the diagnostic's original Stats aggregation and original invocations.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
import time
import lzma

from ckks_evidence_transport import (
    MAX_DECODED, packed, unpack, reserve_next_pair, verify_source_binding,
)

def canonical(value):
    return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def differences(a,b,cells):
    if set(a)!=set(cells) or set(b)!=set(cells):raise ValueError('incomplete cell manifest')
    values={k:a[k]-b[k] for k in cells}
    if not all(math.isfinite(v) for v in values.values()):raise ValueError('nonfinite discrepancy difference')
    return values

def pilot_scales(pairs,cells):
    if len(pairs)!=8:raise ValueError('incorrect pilot size')
    scales={k:statistics.stdev(p[k] for p in pairs) for k in cells}
    if not all(math.isfinite(v) and v>0 for v in scales.values()):raise ValueError('unresolved pilot scale; collection aborted without floor')
    return scales

def tolerance(pairs,scales,cells):
    if len(pairs)!=19:raise ValueError('incorrect calibration size')
    maxima=[max(p[k]/scales[k] for k in cells) for p in pairs]
    if not all(math.isfinite(v) for v in maxima):raise ValueError('nonfinite calibration maximum')
    q=max(0.0,sorted(maxima)[18])
    return q,{k:q*scales[k] for k in cells},maxima

def verdict(pair,bands,cells):
    if set(pair)!=set(cells) or set(bands)!=set(cells):raise ValueError('incomplete verdict')
    return {k:math.isfinite(pair[k]) and pair[k]<=bands[k] for k in cells}

def validate_scaled_collection(plan):
    expected={'batch_invocations':1,'pilot_pairs':8,'calibration_pairs':19,
              'validation_pairs':5,'alpha':0.05,'rank':19,
              'scale':'sample_sd_signed_pilot_differences','scale_floor':None,
              'sensitivity':['independent_incomplete'],'sensitivity_pairs_per_omission':5}
    if any(plan.get(k)!=v for k,v in expected.items()):
        raise ValueError('scaled preregistered collection counts/rule differ')

def collect_native(solution_path):
    import evaluator as e
    evidence={'status':'running','formal_acceptance':False,'conditional_coverage':'numeric compatibility conditional on complete valid executions; execution reliability reported separately','events':[],'pairs':[]}
    cleanups=[];started=time.monotonic();deadline=started+e.CONFIG['runtime']['timeout_seconds']-30
    last_rows=({},{});names=e.FINAL_SUITE;cells=[w+'/'+m for w in names for m in e.METRICS]
    def event(kind,**extra):evidence['events'].append({'event':kind,'elapsed_seconds':time.monotonic()-started,**extra})
    def load(patch):
        with e.private_child_environment(e.clean_env()):r=e.apply_submission_patch(patch,e.TASK_DIR,e.CONFIG)
        if not r.ok:raise e.EvaluationError('calibration fixture setup failed')
        cleanups.append(r.cleanup_root);return r.workdir
    def numeric(rows):return {w+'/'+m:rows[w][m] for w in names for m in e.METRICS}
    def pair(stage,index,a_tree,b_tree,directory):
        nonlocal last_rows
        # Layered archive/member/decoded budget; retain the complete prefix.
        reserve_next_pair(evidence)
        record={'stage':stage,'index':index,'status':'pending','sides':{}}
        evidence['pairs'].append(record)
        completed={}
        for side,tree,coverage in [('A',a_tree,False),('B',b_tree,True)]:
            with tempfile.TemporaryDirectory(prefix=f'{stage}-{index}-{side}-',dir=directory) as scratch:
                rows,detail=e.measure_suite(tree,names,Path(scratch),deadline,coverage=coverage)
            record['sides'][side]=detail
            if set(rows)!=set(names):
                record['status']='invalid';raise e.EvaluationError('invalid full-suite calibration execution; no replacement')
            completed[side]=rows
        record['differences']=differences(numeric(completed['A']),numeric(completed['B']),cells)
        record['status']='ok';last_rows=(completed['A'],completed['B'])
        return record['differences']
    try:
        if e.RELEASE['mode']!='calibration':raise e.EvaluationError('not a calibration release')
        plan=json.loads((e.TASK_DIR/'judge/calibration_preregistration.json').read_text())
        validate_scaled_collection(plan['collection'])
        if plan['collection']!=e.RELEASE['collection'] or plan['cells']!=cells:raise e.EvaluationError('calibration plan mismatch')
        for name,sha in plan['scientific_file_sha256'].items():
            if not verify_source_binding(e.TASK_DIR,name,sha):raise e.EvaluationError('calibration source mismatch')
        if hashlib.sha256(Path(solution_path).read_bytes()).hexdigest()!=plan['reference_patch_sha256']:raise e.EvaluationError('calibration-only release requires exact reference payload')
        evidence.update(preregistration=plan,preregistration_sha256=digest(plan),toolchain=e.run(['go','version'],e.TASK_DIR,deadline).strip())
        if e.RELEASE['toolchain'] not in evidence['toolchain']:raise e.EvaluationError('incorrect Go toolchain')
        candidate=load(solution_path)
        restored=e.restore_reference();cleanups.append(restored.cleanup_root);reference=restored.workdir
        with tempfile.TemporaryDirectory(prefix='calibration-',dir=e.scratch_root()) as scratch:
            d=Path(scratch);e.guards(candidate,d,deadline)
            pilot=[pair('pilot',i,candidate,reference,d) for i in range(8)]
            scales=pilot_scales(pilot,cells);evidence['scales']=scales
            event('scales_frozen_before_calibration',scales_sha256=digest(scales),pilot_pairs=8)
            calibration=[pair('calibration',i,candidate,reference,d) for i in range(19)]
            q,bands,maxima=tolerance(calibration,scales,cells)
            frozen={'alpha':0.05,'K':19,'rank':19,'q':q,'scales':scales,'bands':bands,'maxima':maxima,'cells':cells,'batch_invocations':1}
            evidence['frozen']=frozen;event('bands_frozen_before_validation',frozen_sha256=digest(frozen))
            for i in range(5):
                diff=pair('validation',i,candidate,reference,d);evidence['pairs'][-1]['cell_passes']=verdict(diff,bands,cells)
            # A validation rejection is recorded, never used to redraw or refit.
            for name in plan['collection']['sensitivity']:
                if name != 'independent_incomplete': raise e.EvaluationError('unrecognized scaled control')
                tree=load(e.TASK_DIR/'baseline_naive.patch');e.guards(tree,d,deadline)
                for i in range(5):
                    diff=pair('sensitivity_'+name,i,tree,reference,d);r=evidence['pairs'][-1]
                    r.update(cell_passes=verdict(diff,bands,cells),candidate_score=e.accuracy(list(last_rows[0].values())),reference_score=e.accuracy(list(last_rows[1].values())))
        evidence['status']='collected';event('collection_finished')
    except Exception as exc:
        evidence.update(status='invalid',failure={'type':type(exc).__name__,'reason':str(exc)[:2000],'receipt':getattr(exc,'receipt',{})})
        event('collection_aborted_without_replacement')
    finally:
        for path in cleanups:
            if path:shutil.rmtree(path,ignore_errors=True)
    # Collection is not a scored candidate episode. Numeric top-level zero is
    # an explicit transport placeholder; all reference/omission rows are private.
    result=e.failed_result(names, {}, 'calibration-only transport', calibration_invalid=True)
    metrics=result[3];metrics['status']='calibration_collected' if evidence['status']=='collected' else 'calibration_invalid'
    metrics['operator_evidence']=packed(evidence)
    metrics['calibration_transport_placeholder']=True
    return 0.0,0.0,'calibration-only collection; score zero is a transport placeholder; formal acceptance disabled',metrics
