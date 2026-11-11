"""The gate is main's strict rule; category_means is candidate-only feedback.

Counters below are invented comparison inputs, except where a test reads the
pinned reference table; no predictor is built or run.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import json
from decimal import Decimal
from fractions import Fraction

import pytest

from tests.maintenance.runlts_predictor.runtime_imports import (
    DESIGN_SPEC,
    FEEDBACK,
    MODULES,
    TASK,
)

e = MODULES["evaluator"]
feedback = MODULES["beat_spec_feedback"]
BITS = MODULES["source_policy"].REFERENCE_BITS
SPEC = DESIGN_SPEC
MANIFEST = json.loads((TASK / "judge/workload_manifest.json").read_text())["workloads"]
PINNED = {
    row["id"]: e.pinned_measurement(row)
    for row in json.loads((TASK / "judge/reference_counts.json").read_text())["rows"]
}
DEVELOPMENT = [row for row in MANIFEST if row["development"]]
N = 1_000_000


def cell(misp, wp, instr=N):
    return e.Measurement(
        instr,
        misp,
        wp,
        2 * instr - 1,
        Decimal(f"{1000 * misp / instr:.4f}"),
        Decimal(f"{1000 * wp / instr:.4f}"),
    )


def main_rule(candidate, reference, bits):
    """The rule of origin/main, written out independently of grade()."""
    return int(
        bits <= 1572864
        and all(
            candidate[w].misp_br * reference[w].instr
            < reference[w].misp_br * candidate[w].instr
            and candidate[w].cyc_wp * reference[w].instr
            < reference[w].cyc_wp * candidate[w].instr
            for w in reference
        )
    )


def final_suite(candidate_of, reference=(100_000, 1_000_000)):
    ref = {row["id"]: cell(*reference) for row in MANIFEST}
    cand = {row["id"]: cell(*candidate_of(index)) for index, row in enumerate(MANIFEST)}
    metrics = e.grade(MANIFEST, cand, ref, True, logical_storage_bits=BITS)[3]
    assert (
        metrics["beats_reference"]
        == metrics["final_beats_reference"]
        == main_rule(cand, ref, BITS)
    )
    return metrics


CASES = {
    "equal to the reference": (lambda i: (100_000, 1_000_000), 0),
    "one event better on every trace and count": (lambda i: (99_999, 999_999), 1),
    "ten percent better everywhere": (lambda i: (90_000, 900_000), 1),
    "better everywhere, one misprediction tie": (
        lambda i: (100_000 if i == 50 else 90_000, 900_000),
        0,
    ),
    "better everywhere, one wrong-path tie": (
        lambda i: (90_000, 1_000_000 if i == 50 else 900_000),
        0,
    ),
    "better everywhere, one trace one event worse": (
        lambda i: (100_001 if i == 104 else 90_000, 900_000),
        0,
    ),
    "better means, eleven traces worse": (
        lambda i: (101_000 if i < 11 else 80_000, 800_000),
        0,
    ),
    "mispredictions better, cycles worse": (lambda i: (90_000, 1_010_000), 0),
    "naive-like, worse on both": (lambda i: (104_000, 1_013_500), 0),
}


@pytest.mark.parametrize("name", list(CASES))
def test_gate_decision_is_the_strict_per_trace_rule_of_main(name):
    candidate_of, expected = CASES[name]
    metrics = final_suite(candidate_of)
    assert metrics["beats_reference"] == expected
    assert metrics["beats_reference"] == int(
        all(
            v
            for row in metrics["beat_spec_workload_passes"]
            for v in row["metrics"].values()
        )
    )


def test_criteria_and_constants_are_those_of_main():
    assert e.METRICS == ("br_mis_pki", "cyc_wp_pki", "logical_storage_bits")
    for name in ("MEAN_IMPROVEMENT", "TRACE_TOLERANCE"):
        assert not hasattr(e, name)
    spec = json.loads(SPEC.read_text())
    assert [m["name"] for m in spec["metrics"]] == list(e.METRICS)
    assert (
        "Equality fails" in spec["metrics"][0]["clears_when"]
        and "No mean" in spec["beats_reference"]
    )
    metrics = final_suite(lambda i: (90_000, 900_000))
    for gone in (
        "significance_gate",
        "strict_workload_passes",
        "mean_gate",
        "tolerance_exceeded_traces",
        "strict_progress",
    ):
        assert gone not in metrics


def test_real_reference_counts_against_themselves_fail():
    metrics = e.grade(MANIFEST, PINNED, PINNED, True, logical_storage_bits=BITS)[3]
    assert metrics["beats_reference"] == 0 == main_rule(PINNED, PINNED, BITS)


def test_perturbed_real_counts_follow_the_strict_rule():
    # Real reference counters moved a little either way, as a shifted random stream does.
    def moved(scale_of):
        return {
            wid: e.Measurement(
                r.instr,
                int(r.misp_br * scale_of(i)),
                int(r.cyc_wp * scale_of(i)),
                r.full_instr,
                r.printed_br,
                r.printed_wp,
            )
            for i, (wid, r) in enumerate(PINNED.items())
        }

    mixed = moved(lambda i: Fraction(99 if i % 2 else 101, 100))
    better = moved(lambda i: Fraction(97, 100))
    for candidate, expected in ((mixed, 0), (better, 1)):
        metrics = e.grade(MANIFEST, candidate, PINNED, True, logical_storage_bits=BITS)[
            3
        ]
        assert (
            metrics["beats_reference"] == expected == main_rule(candidate, PINNED, BITS)
        )


def development():
    reference = {row["id"]: PINNED[row["id"]] for row in DEVELOPMENT}
    candidate = {
        wid: e.Measurement(
            r.instr,
            r.misp_br * 97 // 100,
            r.cyc_wp * 97 // 100,
            r.full_instr,
            r.printed_br,
            r.printed_wp,
        )
        for wid, r in reference.items()
    }
    return (
        e.grade(DEVELOPMENT, candidate, reference, False, logical_storage_bits=BITS),
        candidate,
        reference,
    )


def test_category_means_are_the_candidates_own_and_nothing_else():
    result, candidate, reference = development()
    shown = feedback.public_metrics(
        result[3], visible=e.CONFIG["evaluation"]["solver_visible_metrics"]
    )
    assert set(shown) == {
        "train_br_mis_pki_mean",
        "train_cyc_wp_pki_mean",
        "all_metrics_progress",
        "category_means",
        "candidate_failure",
        "beat_spec_progress",
    }
    assert shown["candidate_failure"] == 0
    assert set(shown["beat_spec_progress"]) == set(e.METRICS)

    def exact(rows, field):
        return float(
            sum(Fraction(1000 * getattr(m, field), m.instr) for m in rows) / len(rows)
        )

    assert {k: v["traces"] for k, v in shown["category_means"].items()} == {
        "compress": 2,
        "fp": 3,
        "infra": 4,
        "int": 8,
        "media": 1,
        "web": 6,
    }
    secret = set()
    for category, entry in shown["category_means"].items():
        members = [row["id"] for row in DEVELOPMENT if row["category"] == category]
        assert entry == {
            "traces": len(members),
            "br_mis_pki_mean": exact([candidate[i] for i in members], "misp_br"),
            "cyc_wp_pki_mean": exact([candidate[i] for i in members], "cyc_wp"),
        }
        secret |= {
            exact([reference[i] for i in members], "misp_br"),
            exact([reference[i] for i in members], "cyc_wp"),
        }
    for r in reference.values():
        secret |= {r.misp_br, r.cyc_wp, exact([r], "misp_br"), exact([r], "cyc_wp")}
    secret |= {
        exact(list(reference.values()), "misp_br"),
        exact(list(reference.values()), "cyc_wp"),
    }

    def walk(value, key=""):
        yield key, value
        if isinstance(value, dict):
            for k, v in value.items():
                yield from walk(v, k)

    nodes = list(walk(shown))
    assert not any(isinstance(v, list) for _, v in nodes)
    assert (
        not {
            v
            for _, v in nodes
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        & secret
    )
    encoded = json.dumps(shown)
    for word in ("reference", "target", "ratio", "margin", "final", "beats"):
        assert word not in encoded
    for row in MANIFEST:
        assert row["id"] not in encoded
    spec = json.loads(SPEC.read_text())
    for entry in spec["solver_feedback"]:
        assert entry["reveals_reference"].startswith(("no \u2014 ", "no; ")), entry[
            "name"
        ]
    # [feedback] is the design record's list plus the candidate-only failure code.
    assert FEEDBACK == [entry["name"] for entry in spec["solver_feedback"]] + [
        "candidate_failure"
    ]


def test_final_metrics_have_no_category_means():
    assert "category_means" not in final_suite(lambda i: (90_000, 900_000))


def test_statement_keeps_the_strict_rule_and_states_the_measured_noise():
    # The statement (repo/README.md); train feedback names, category_means among them, are the
    # beat spec's solver_feedback (test_category_means_are_the_candidates_own_and_nothing_else).
    text = " ".join((TASK / "repo/README.md").read_text().split())
    for needle in (
        "105 original CBP traces",
        "| final | all 105, rerun afresh (includes the train subset) |",
        "**Beats the reference** when, on **every** final trace, all three hold: `br_mis_pki` "
        "(mispredictions per instruction) strictly lower than the private reference's and `cyc_wp_pki` "
        "(wrong-path cycles per instruction) strictly lower, both compared as exact integer counters "
        "with no tolerance (a tie fails); and `logical_storage_bits <= 1,572,864`.",
        "A one-call shift of the shared random stream moves one trace's mispredictions by about 1% "
        "(up to 6.6%) and wrong-path cycles by about 0.7%, so winning every trace needs a clearly "
        "larger improvement.",
    ):
        assert needle in text, needle
    for stale in (
        "remains incomplete",
        "repeatability remain pending",
        "mean_gate",
        "0.002",
    ):
        assert stale not in text, stale
