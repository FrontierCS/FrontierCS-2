#!/usr/bin/env python3
"""The PASS rule of the task, in exact integer arithmetic.

Per graded program, candidate and reference are each run on the same K seeds,
interleaved on the same machine.  Two quantities are measured per run:

  coverage   branches covered when the run ends (integer);
  overhead   search overhead: CPU time of strategy code plus constraint
             solving, excluding the program under test (integer milliseconds).

Both rules are the same test, a win by more than MARGIN_SIGMAS standard
errors of a difference of two K-run means:

  (a)  mean(coverage_c) - mean(coverage_r) > MARGIN_SIGMAS * s_cov * sqrt(2/K)
  (b)  mean(overhead_r) - mean(overhead_c) > MARGIN_SIGMAS * s_ovh * sqrt(2/K)

where s^2 = (var(candidate) + var(reference)) / 2 is the pooled per-run
sample variance (divisor K - 1) of that quantity in this very judgment.  The
margins are therefore derived from the spread measured in the judgment, which
contains both the seed-to-seed variation and the run-to-run measurement noise.

PASS needs both rules on every graded program, and a valid candidate.
"""

# Runs (seeds) per program, for the candidate and for the reference.
K_SEEDS = 30
# Number of standard errors a win must exceed.  With the reference's spreads
# measured at qualification over 30 seeds
#   coverage s.d.  41.5 / 30.6 / 223 branches  (development / final B / final A)
#   overhead s.d.  6.1 / 2.4 / 53 s
# the margins come to about 32 / 24 / 173 branches (2.0 %, 3.6 %, 2.2 % of the
# reference's coverage) and 4.7 / 1.9 / 41 s (6.5 %, 4.7 %, 28 % of its
# overhead).  A reference-equivalent strategy clears either rule by chance in
# about 0.2 % of judgments and both rules on one program in fewer than 1 in
# 20,000 (random splits of 60 reference runs, per program).
MARGIN_SIGMAS = 3


def _sum_and_q(values, k):
    """Returns (sum, k * sum of squares - sum^2); the second term is
    k * (k - 1) times the sample variance, an integer for integer data."""
    total = sum(values)
    return total, k * sum(v * v for v in values) - total * total


def significantly_greater(high, low, sigmas=MARGIN_SIGMAS):
    """True iff mean(high) - mean(low) > sigmas * s * sqrt(2 / k), exactly.

    With Q = k (k - 1) var, s^2 = (Qh + Ql) / (2 k (k - 1)) and D = sum(high)
    - sum(low), the inequality D / k > sigmas * s * sqrt(2 / k) is, after
    squaring (the right side is non-negative),
        D > 0  and  D^2 * (k - 1) > sigmas^2 * (Qh + Ql).
    Ties and negative differences fail; equal samples fail.
    """
    k = len(high)
    if k < 2 or len(low) != k:
        raise ValueError('the rule needs equally many, at least two, runs per side')
    if not all(type(v) is int and v >= 0 for v in list(high) + list(low)):
        raise ValueError('measurements must be non-negative integers')
    sum_h, q_h = _sum_and_q(high, k)
    sum_l, q_l = _sum_and_q(low, k)
    d = sum_h - sum_l
    return d > 0 and d * d * (k - 1) > sigmas * sigmas * (q_h + q_l)


def margin(a, b, sigmas=MARGIN_SIGMAS):
    """The right-hand side of the rule as a float, for reports only."""
    k = len(a)
    _, q_a = _sum_and_q(a, k)
    _, q_b = _sum_and_q(b, k)
    return sigmas * ((q_a + q_b) / float(k * k * (k - 1))) ** 0.5


def coverage_rule(candidate, reference):
    """Rule (a): the candidate covers significantly more branches."""
    return significantly_greater(candidate, reference)


def overhead_rule(candidate_ms, reference_ms):
    """Rule (b): the candidate's search overhead is significantly lower."""
    return significantly_greater(reference_ms, candidate_ms)


def decide(programs):
    """programs: list of dicts with keys
         workload, candidate_coverage, reference_coverage,
         candidate_overhead_ms, reference_overhead_ms, candidate_valid.
    Returns (beats_reference, rows); rows is the per-workload decision table
    in the format the verifier adapter expects."""
    rows = []
    for p in programs:
        valid = bool(p['candidate_valid'])
        rows.append({'workload': p['workload'], 'metrics': {
            'branches_covered': valid and coverage_rule(p['candidate_coverage'], p['reference_coverage']),
            'search_overhead_seconds': valid and overhead_rule(p['candidate_overhead_ms'],
                                                               p['reference_overhead_ms']),
        }})
    beats = int(bool(rows) and all(v for row in rows for v in row['metrics'].values()))
    return beats, rows
