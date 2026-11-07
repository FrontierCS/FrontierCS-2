"""Proposed formal scoring layer; adopt only with audited native calibration.
This module neither draws a reference nor chooses a workload. The existing
original-main evaluator supplies fresh candidate/reference observations.
"""
import hashlib
import json
import math
from pathlib import Path

METRICS=('avg_log2_prec_l2_abs_discrepancy_bits','std_log2_prec_l2_abs_discrepancy_bits')

def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()

def read_calibration(path, *, expected_sha256, final_workloads, expected_binding):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('frozen calibration file digest mismatch')
    record=json.loads(raw)
    if record['status']!='audited_complete_native_collection' or record['measurement_binding']!=expected_binding:raise ValueError('unaudited or incompatible calibration')
    frozen=record['frozen'];cells=[w+'/'+m for w in final_workloads for m in METRICS]
    if frozen['cells']!=cells or frozen['batch_invocations']!=1:raise ValueError('different calibration cell/batch manifest')
    if frozen['alpha']!=0.05 or frozen['K']!=19 or frozen['rank']!=19:raise ValueError('different preregistered predictive rule')
    if set(frozen['bands'])!=set(cells) or set(frozen['scales'])!=set(cells):raise ValueError('missing/extra calibration cells')
    q=frozen['q']
    if type(q) not in (int,float) or not math.isfinite(q) or q<0:raise ValueError('invalid calibration multiplier')
    for cell in cells:
        scale=frozen['scales'][cell];band=frozen['bands'][cell]
        if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0:raise ValueError('invalid calibration scale')
        if type(band) not in (int,float) or not math.isfinite(band) or band<0 or band!=q*scale:raise ValueError('invalid calibration band')
    development=record['development_workloads']
    if not isinstance(development,list) or not development or len(development)!=len(set(development)) or not set(development)<set(final_workloads):raise ValueError('invalid development manifest')
    return record

def score(rows,names):
    if len(names)!=len(set(names)) or set(rows)!=set(names):raise ValueError('incomplete or duplicate scoring suite')
    values=[rows[w][m] for w in names for m in METRICS]
    if not all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values):raise ValueError('invalid candidate discrepancy')
    return 100/(1+sum(values)/len(values))

TIME_METRIC='estimator_seconds'

def faster_in_every_pair(candidate_ns,reference_ns,pairs):
    """Strictly lower trusted estimator clock in each interleaved pair."""
    if len(candidate_ns)!=pairs or len(reference_ns)!=pairs or pairs<1:raise ValueError('incomplete estimator clock pairs')
    if not all(type(v) is int and v>0 for v in (*candidate_ns,*reference_ns)):raise ValueError('invalid estimator clock')
    return all(c<r for c,r in zip(candidate_ns,reference_ns))

def compare(candidate,reference,names,calibration,*,mode='final',pairs=3,development=None):
    """Rows hold the mean discrepancy over the interleaved invocations and their clocks."""
    if mode not in ('final','development'):raise ValueError('invalid trusted verifier mode')
    final_cells=calibration['frozen']['cells']
    # Development feedback may measure a declared subset of the calibrated development rows.
    feedback=list(development) if development is not None else list(calibration['development_workloads'])
    if not feedback or len(set(feedback))!=len(feedback) or not set(feedback)<=set(calibration['development_workloads']):raise ValueError('invalid development feedback manifest')
    expected=final_cells if mode=='final' else [w+'/'+m for w in feedback for m in METRICS]
    if not names or [w+'/'+m for w in names for m in METRICS]!=expected:raise ValueError('comparison does not match frozen mode manifest')
    candidate_score=score(candidate,names);reference_score=score(reference,names)
    bands=calibration['frozen']['bands'];passes=[]
    for w in names:
        row={m:candidate[w][m]-reference[w][m]<=bands[w+'/'+m] for m in METRICS}
        row[TIME_METRIC]=faster_in_every_pair(candidate[w]['estimator_ns'],reference[w]['estimator_ns'],pairs)
        passes.append({'workload':w,'metrics':row})
    # Inclusive per-cell accuracy conjunction and strict per-pair time
    # conjunction, with no cross-workload or cross-condition compensation.
    return {'score':candidate_score,'reference_score':reference_score,'margin':candidate_score-reference_score,'beats_reference':int(all(v for row in passes for v in row['metrics'].values())),'beat_spec_workload_passes':passes,'calibration_pending':False}
