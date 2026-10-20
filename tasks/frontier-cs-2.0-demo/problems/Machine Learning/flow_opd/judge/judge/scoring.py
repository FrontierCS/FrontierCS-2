"""Bookkeeping only. All rewards must originate in a qualified GPU/OCR run."""
import math


class QualificationError(ValueError):
    pass


class TrustedEvaluationFailure(RuntimeError):
    def __init__(self, reason):
        super().__init__('Trusted OCR evaluation did not establish a scientific result.')
        self.trusted_evaluation_failure = {'reason': reason}


MAX_REWARD = 1.0


def reward_ok(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def beats_workload(candidate_value, reference_value):
    """Strictly better below the scorer's maximum; at the maximum the candidate
    must also reach it (equality passes only on ceiling rows). A reference copy
    therefore still fails every non-ceiling workload, so the gate stays meaningful
    while becoming attainable."""
    if reference_value >= MAX_REWARD:
        return candidate_value >= reference_value
    return candidate_value > reference_value


def failure_metrics(ids, *, reference, final=False, updates=0,
                    status='invalid_submission', feedback_reference=None):
    if (not ids or len(ids) != len(set(ids)) or set(reference) != set(ids)
            or any(not reward_ok(reference[i]) for i in ids)):
        raise QualificationError('Invalid frozen reference for candidate rejection.')
    values = list(reference.values()) if final else feedback_reference
    if not values or any(not reward_ok(v) for v in values):
        raise QualificationError('Missing genuine feedback reference.')
    reference_score = 100 * math.fsum(values) / len(values)
    return {'beats_reference':0,
            'beat_spec_workload_passes':[{'workload':i, 'metrics':{'ocr_reward':False}} for i in ids],
            'reference_score':reference_score, 'margin':-reference_score,
            'dev_eval_reward_ocr':None,
            'dev_verified_optimizer_updates':None if final else updates,
            'evaluation_status':status,
            'candidate_rejection':{'status':status, 'measurements':'not_established'}}


def aggregate(ids, candidate, reference, *, candidate_valid, reference_valid,
              final, development_rewards=None, development_reference_rewards=None, updates=0):
    if not ids or len(ids)!=len(set(ids)) or any(not isinstance(i,str) or not i for i in ids):
        raise QualificationError('invalid frozen workload identities')
    if (not reference_valid or set(reference)!=set(ids)
            or any(not reward_ok(reference[i]) for i in ids)):
        raise QualificationError('reference evidence missing or invalid')
    complete = (candidate_valid and set(candidate)==set(ids)
                and all(reward_ok(candidate[i]) for i in ids))
    rows = [{'workload':i, 'metrics':{'ocr_reward':bool(complete and beats_workload(candidate[i], reference[i]))}}
            for i in ids]
    reference_score = 100 * math.fsum(reference[i] for i in ids)/len(ids)
    score = 100 * math.fsum(candidate[i] for i in ids)/len(ids) if complete else 0.0
    # Public feedback is a SEPARATE public development run, never private grading rows.
    dev_mean = None
    if not final and development_rewards is not None:
        if development_rewards and all(reward_ok(x) for x in development_rewards):
            dev_mean = math.fsum(development_rewards)/len(development_rewards)
    if not final:
        if (development_reference_rewards is None or not development_reference_rewards
                or any(not reward_ok(x) for x in development_reference_rewards)):
            raise QualificationError('Public development reference evidence missing.')
        reference_score = 100*math.fsum(development_reference_rewards)/len(development_reference_rewards)
        score = 100*dev_mean if dev_mean is not None else 0.0
    metrics = {'beats_reference':int(complete and all(r['metrics']['ocr_reward'] for r in rows)),
               'beat_spec_workload_passes':rows, 'reference_score':reference_score,
               'margin':score-reference_score, 'dev_eval_reward_ocr':dev_mean,
               'dev_verified_optimizer_updates':None if final else updates,
               'evaluation_status':'complete' if complete else 'invalid_submission'}
    # In development, only the public mean can contribute to the user-visible score.
    # The private train-grading rows feed the separate criterion only.
    visible_score = score if final else (100*dev_mean if dev_mean is not None else 0.0)
    return visible_score, visible_score, 'Completed OCR evaluation.' if complete else 'Invalid or incomplete run.', metrics
