"""The PASS arithmetic: exact thresholds, ties, symmetry of the two rules."""
import deployment_test_guard  # noqa: F401  # isort: skip

from fractions import Fraction
import math
import random

import pytest
from tests.maintenance.cute_crest.runtime_imports import MODULES

gate = MODULES["gate"]


def reference_formula(high, low, sigmas=3):
    """mean(high) - mean(low) > sigmas * s * sqrt(2/k), in exact rationals where
    possible: compare squares of both sides."""
    k = len(high)
    mean_h, mean_l = Fraction(sum(high), k), Fraction(sum(low), k)
    var_h = sum((Fraction(v) - mean_h) ** 2 for v in high) / (k - 1)
    var_l = sum((Fraction(v) - mean_l) ** 2 for v in low) / (k - 1)
    difference = mean_h - mean_l
    return difference > 0 and difference ** 2 > sigmas ** 2 * (var_h + var_l) / 2 * Fraction(2, k)


def test_constants():
    assert gate.K_SEEDS == 30
    assert gate.MARGIN_SIGMAS == 3


def test_identical_samples_fail():
    values = [1500 + i for i in range(30)]
    assert not gate.coverage_rule(values, values)
    assert not gate.overhead_rule(values, values)


def test_constant_samples_tie_and_strict_win():
    assert not gate.coverage_rule([7] * 30, [7] * 30)
    assert gate.coverage_rule([8] * 30, [7] * 30)      # zero variance: any gain is a win
    assert not gate.coverage_rule([7] * 30, [8] * 30)
    assert gate.overhead_rule([7] * 30, [8] * 30)
    assert not gate.overhead_rule([8] * 30, [7] * 30)


def test_exact_threshold_is_not_a_win():
    # k = 2, reference {0, 2}: Q_r = 2*4 - 4 = 4.  Candidate {c, c}: Q_c = 0,
    # D = 2c - 2.  Rule: D^2 * 1 > 9 * 4, i.e. D > 6.  D = 6 is the boundary.
    assert not gate.significantly_greater([4, 4], [0, 2])   # D = 6: equality fails
    assert gate.significantly_greater([5, 4], [0, 2]) == reference_formula([5, 4], [0, 2])
    assert gate.significantly_greater([5, 5], [0, 2])       # D = 8


def test_matches_rational_formula_on_random_data():
    rng = random.Random(465)
    for _ in range(2000):
        k = rng.choice([2, 3, 5, 30])
        low = [rng.randint(600, 700) for _ in range(k)]
        high = [v + rng.randint(-10, 40) for v in low]
        assert gate.significantly_greater(high, low) == reference_formula(high, low)


def test_margin_matches_measured_scale():
    # Two samples with s.d. about 21 (the development program's measured spread).
    rng = random.Random(1)
    a = [int(rng.gauss(1586, 21)) for _ in range(30)]
    b = [int(rng.gauss(1586, 21)) for _ in range(30)]
    m = gate.margin(a, b)
    assert 10 < m < 25
    k = 30
    mean_a, mean_b = sum(a) / k, sum(b) / k
    var = (sum((v - mean_a) ** 2 for v in a) + sum((v - mean_b) ** 2 for v in b)) / (2 * (k - 1))
    assert math.isclose(m, 3 * math.sqrt(var) * math.sqrt(2 / k), rel_tol=1e-12)


def test_overhead_rule_is_the_mirror_of_the_coverage_rule():
    rng = random.Random(7)
    for _ in range(500):
        a = [rng.randint(60000, 80000) for _ in range(30)]
        b = [v - rng.randint(-2000, 9000) for v in a]
        assert gate.overhead_rule(b, a) == gate.coverage_rule(a, b)


def test_bad_inputs_are_errors_not_verdicts():
    with pytest.raises(ValueError):
        gate.coverage_rule([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        gate.coverage_rule([1], [1])
    with pytest.raises(ValueError):
        gate.coverage_rule([1.5, 2], [1, 2])
    with pytest.raises(ValueError):
        gate.overhead_rule([True, 2], [1, 2])
    with pytest.raises(ValueError):
        gate.coverage_rule([-1, 2], [1, 2])


def program(c_cov, r_cov, c_ms, r_ms, valid=True, name="w0"):
    return {"workload": name, "candidate_valid": valid, "candidate_coverage": c_cov,
            "reference_coverage": r_cov, "candidate_overhead_ms": c_ms, "reference_overhead_ms": r_ms}


def test_decide_needs_both_rules_on_every_program():
    low, high = [100 + i % 3 for i in range(30)], [130 + i % 3 for i in range(30)]
    win = program(high, low, low, high)
    assert gate.decide([win, dict(win, workload="w1")])[0] == 1
    # Coverage win, overhead parity.
    beats, rows = gate.decide([program(high, low, low, low)])
    assert beats == 0 and rows[0]["metrics"] == {"branches_covered": True, "search_overhead_seconds": False}
    # Overhead win, coverage parity.
    beats, rows = gate.decide([program(low, low, low, high)])
    assert beats == 0 and rows[0]["metrics"] == {"branches_covered": False, "search_overhead_seconds": True}
    # One program wins, the other does not: no compensation across programs.
    assert gate.decide([win, program(low, low, low, low, name="w1")])[0] == 0
    # An invalid candidate passes nothing even with winning numbers.
    beats, rows = gate.decide([dict(win, candidate_valid=False)])
    assert beats == 0 and not any(rows[0]["metrics"].values())
    assert gate.decide([])[0] == 0
