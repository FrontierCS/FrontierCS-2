"""Preregistered effective-FPR and theoretical-space measurements.

Intervals invert the Bernoulli KL Chernoff bound, not a normal approximation.
Their coverage assumes IID Bernoulli absent-query outcomes within a sample;
deterministic hashing/structured keys do not establish that assumption.
Repeated executions of the same keys are never pooled as new query trials.

Both accuracy metrics are judged as certified non-inferiority guards against
preregistered reference-only anchors: a workload is rejected only when the
candidate's certified lower bound exceeds its anchor. Overlapping intervals are
never read as equal true rates, and no anchor relaxes a strict performance win.
"""
from __future__ import annotations

import math
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
from functools import lru_cache

METRIC_NAMES = ('effective_fpr', 'space_overhead_pct')
RULE = 'certified-non-inferiority/v1'
# 36 final configurations * 5 repetitions * candidate/reference. Mixture rows
# reuse the same untimed absent stream and do not add independent intervals.
INTERVAL_FAMILY_SIZE = 360
INTERVAL_FAMILY_ALPHA = 1e-6
SAMPLE_ALPHA = INTERVAL_FAMILY_ALPHA / INTERVAL_FAMILY_SIZE
MIN_REFERENCE_ROUNDS = 5


class CalibrationRequired(RuntimeError):
    """Author calibration is missing/invalid; never blame the candidate."""


class RuntimeQualificationRequired(CalibrationRequired):
    """Reference statistics exist, but the exact production runtime is unqualified."""


# All certification operations use explicit contexts, independent of callers'
# decimal settings. Decimal.ln is correctly rounded (ROUND_HALF_EVEN); adjacent
# representable decimals therefore enclose its exact value. Basic arithmetic
# below is directed. Precision controls cost/tightness, NOT a scoring tolerance.
_DOWN = Context(prec=96, rounding=ROUND_FLOOR)
_UP = Context(prec=96, rounding=ROUND_CEILING)


@lru_cache(maxsize=4096)
def _log_ratio_bounds(numerator, denominator):
    """Certified decimal enclosure of ln(numerator/denominator), integers >0."""
    a, b = Decimal(numerator), Decimal(denominator)
    lower, upper = _DOWN.divide(a, b), _UP.divide(a, b)
    return (_DOWN.next_minus(_DOWN.ln(lower)),
            _UP.next_plus(_UP.ln(upper)))


def _contour_bounds(k, m, q):
    """Enclose m*KL(k/m || q) without rounding k/m or 1-q to binary64."""
    qn, qd = q.as_integer_ratio()
    lower = upper = Decimal(0)
    for weight, numerator, denominator in (
            (k, k*qd, m*qn), (m-k, (m-k)*qd, m*(qd-qn))):
        if weight:
            lo, hi = _log_ratio_bounds(numerator, denominator)
            lower = _DOWN.add(lower, _DOWN.multiply(Decimal(weight), lo))
            upper = _UP.add(upper, _UP.multiply(Decimal(weight), hi))
    return lower, upper


def _certify_endpoint(k, m, alpha, proposal, lower):
    """Move a float proposal OUTWARD until an interval proof succeeds.

    Convexity/monotonicity of KL on each side of k/m then proves root enclosure.
    Inconclusive finite precision never accepts an endpoint. A domain endpoint
    (0 or1) is always a safe outer bound, including unrepresentable roots.
    """
    an, ad = alpha.as_integer_ratio()
    threshold_upper = _log_ratio_bounds(2*ad, an)[1]
    outer = 0.0 if lower else 1.0
    value = min(1.0, max(0.0, proposal))
    step = math.ulp(value)
    for _ in range(64):
        if value == outer:
            return value
        qn, qd = value.as_integer_ratio()
        correct_side = qn*m <= k*qd if lower else qn*m >= k*qd
        if 0 < value < 1 and correct_side:
            if _contour_bounds(k, m, value)[0] >= threshold_upper:
                return value
        value = math.nextafter(max(0.0, value-step) if lower
                               else min(1.0, value+step), outer)
        step *= 2
    return outer  # mathematically conservative; never an uncertified proposal


def _float_outward(value, lower):
    result = float(value)
    exact = Decimal.from_float(result)
    if (lower and exact > value) or (not lower and exact < value):
        result = math.nextafter(result, -math.inf if lower else math.inf)
    return result


def overhead_bound(resident_bytes, n, epsilon, *, lower):
    """Directed enclosure of the SAME paper formula at a certified FPR bound."""
    if epsilon == 0:
        return -100.0
    if epsilon == 1:
        return None
    en, ed = epsilon.as_integer_ratio()
    log_lo, log_hi = _log_ratio_bounds(ed, en)  # positive ln(1/epsilon)
    two_lo, two_hi = _log_ratio_bounds(2, 1)
    ctx = _DOWN if lower else _UP
    bits = ctx.divide(Decimal(8*resident_bytes), Decimal(n))
    numerator = ctx.multiply(bits, two_lo if lower else two_hi)
    value = ctx.divide(numerator, log_hi if lower else log_lo)
    value = ctx.multiply(Decimal(100), ctx.subtract(value, Decimal(1)))
    return _float_outward(value, lower)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def bernoulli_kl(p, q):
    if p == 0:
        return -math.log1p(-q) if q < 1 else math.inf
    if p == 1:
        return -math.log(q) if q > 0 else math.inf
    if q in (0, 1):
        return math.inf
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def fpr_interval(false_positives, absent_queries, alpha=SAMPLE_ALPHA):
    """Return [L,U] with per-sample failure probability <= alpha under IID.

    For each tail, m*KL(k/m || p) > log(2/alpha) rejects p. Bisection
    supplies proposals; directed interval certification proves outer bounds.
    Zero/all counts keep 0/1
    endpoints; no pseudocount, nominal fingerprint rate or finite floor.
    """
    if (type(absent_queries) is not int or absent_queries <= 0 or
            type(false_positives) is not int or
            not 0 <= false_positives <= absent_queries or
            not _number(alpha) or not 0 < alpha < 1):
        raise ValueError('invalid binomial observation or alpha')
    p = false_positives / absent_queries
    cutoff = math.log(2 / alpha) / absent_queries
    if false_positives == 0:
        upper = min(1.0, math.nextafter(-math.expm1(-cutoff), 1.0))
        return 0.0, _certify_endpoint(0, absent_queries, alpha, upper, False)
    if false_positives == absent_queries:
        lower = max(0.0, math.nextafter(math.exp(-cutoff), 0.0))
        return _certify_endpoint(false_positives, absent_queries, alpha, lower, True), 1.0
    lo, hi = 0.0, p
    for _ in range(80):
        mid = (lo + hi) / 2
        if bernoulli_kl(p, mid) > cutoff:
            lo = mid
        else:
            hi = mid
    lower = max(0.0, math.nextafter(lo, 0.0))
    lo, hi = p, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if bernoulli_kl(p, mid) > cutoff:
            hi = mid
        else:
            lo = mid
    upper = min(1.0, math.nextafter(hi, 1.0))
    return (_certify_endpoint(false_positives, absent_queries, alpha, lower, True),
            _certify_endpoint(false_positives, absent_queries, alpha, upper, False))


def theoretical_overhead(resident_bytes, n, epsilon):
    """Paper Fig.5a quantity; None means outside its open-domain definition."""
    if (type(resident_bytes) is not int or resident_bytes <= 0 or
            type(n) is not int or n <= 0 or not _number(epsilon) or
            not 0 <= epsilon <= 1):
        raise ValueError('invalid space/FPR observation')
    if epsilon in (0, 1):
        return None
    return 100 * ((8 * resident_bytes / n) / (-math.log2(epsilon)) - 1)


def accuracy_sample(row, n):
    k, m, size = row['effective_false_positives'], row['effective_absent_queries'], row['resident_bytes']
    lower, upper = fpr_interval(k, m)
    epsilon = k / m
    point = theoretical_overhead(size, n, epsilon)
    # Endpoint limits are interval bounds only, never invented point estimates.
    overhead_lower = overhead_bound(size, n, lower, lower=True)
    overhead_upper = overhead_bound(size, n, upper, lower=False)
    return {
        'effective_fpr': {'point': epsilon, 'lower': lower, 'upper': upper},
        'space_overhead_pct': {'point': point, 'lower': overhead_lower,
                               'upper': overhead_upper},
        'counts': {'false_positives': k, 'absent_queries': m},
        'space_domain': 'interior' if point is not None else 'unresolved_endpoint',
    }


def summarize_accuracy(samples, n):
    """Worst of the five executions, with a simultaneous interval for that max.

    Bounds for a max are maxima of the corresponding bounds, not a CI of a
    pooled count. Undefined space points remain undefined for the whole row.
    """
    if not samples:
        raise ValueError('no accuracy samples')
    measurements = [accuracy_sample(row, n) for row in samples]
    result = {}
    for metric in METRIC_NAMES:
        values = [sample[metric] for sample in measurements]
        result[metric] = {
            field: (None if any(v[field] is None for v in values)
                    else max(v[field] for v in values))
            for field in ('point', 'lower', 'upper')
        }
    return result


def nominal_false_positive_rate(workload):
    """The fingerprint specification's rate 2^-b for workload id b{bits}-n...-mix..."""
    head = workload.split('-', 1)[0]
    if len(head) < 2 or head[0] != 'b' or not head[1:].isdigit():
        raise CalibrationRequired('workload identity does not name its fingerprint width')
    bits = int(head[1:])
    if bits not in (8, 16):
        raise CalibrationRequired('unsupported fingerprint width in workload identity')
    return 2.0 ** -bits


def non_inferior(candidate, anchor):
    """Certified non-inferiority against a preregistered reference-only anchor.

    The workload clears when the candidate's certified LOWER bound does not
    exceed the anchor. Under the interval's IID assumption a filter whose true
    value is at or below the anchor is therefore rejected with probability at
    most the per-sample alpha; a certified excess above the anchor fails. This
    is a non-inferiority test, not evidence of equality: overlapping intervals
    never prove equal rates, and the test's resolution is bounded by the sample.
    """
    if not _number(anchor):
        raise CalibrationRequired('invalid preregistered anchor')
    # An observed boundary does not establish an interior-domain space point.
    # It remains measurable evidence, but cannot clear the space predicate.
    if any(candidate.get(k) is None for k in ('point', 'lower', 'upper')):
        return False
    if not all(_number(candidate[k]) for k in ('point', 'lower', 'upper')):
        raise ValueError('non-finite statistical measurement')
    if not candidate['lower'] <= candidate['point'] <= candidate['upper']:
        raise ValueError('statistical point is outside its interval')
    # Compare the exact represented values: binary64 rounding must never move a
    # certified bound across the anchor.
    ln, ld = candidate['lower'].as_integer_ratio()
    an, ad = anchor.as_integer_ratio()
    return ln * ad <= an * ld


def reference_anchors(rounds, workload_ids):
    """Compute, but never guess, preregistered anchors from >=5 reference rounds.

    Input is one mapping per complete reference round: workload -> two point
    estimates. These are construction-run summaries, not new solver results.
    The effective-FPR anchor is the larger of the fingerprint's nominal rate and
    the worst retained reference point; the space anchor is the worst retained
    reference point. Endpoint space observations require more reference
    measurement/design review.
    """
    expected = set(workload_ids)
    if (not isinstance(rounds, list) or len(rounds) < MIN_REFERENCE_ROUNDS or
            not expected or any(not isinstance(r, dict) or set(r) != expected for r in rounds)):
        raise CalibrationRequired('need complete reference repeats for every workload')
    anchors = {}
    for workload in sorted(expected):
        anchors[workload] = {}
        for metric in METRIC_NAMES:
            values = [r[workload].get(metric) for r in rounds]
            if (any(not _number(v) for v in values) or
                    (metric == 'effective_fpr' and any(not 0 < v < 1 for v in values)) or
                    (metric == 'space_overhead_pct' and any(v <= -100 for v in values))):
                raise CalibrationRequired('reference statistical point is unresolved')
            worst = max(values)
            if metric == 'effective_fpr':
                worst = max(worst, nominal_false_positive_rate(workload))
            anchors[workload][metric] = worst
    return anchors
