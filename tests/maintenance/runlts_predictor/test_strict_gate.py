"""The gate is main's strict rule; category_means is candidate-only feedback.

Counters below are invented comparison inputs, except where a test reads the
pinned reference table; no predictor is built or run.
"""
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import os
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from tests.maintenance.runlts_predictor.runtime_imports import MODULES, TASK

e = MODULES["evaluator"]
feedback = MODULES["beat_spec_feedback"]
BITS = MODULES["source_policy"].REFERENCE_BITS
SPEC = TASK / "judge/beat_spec.json"
MANIFEST = json.loads((TASK / "judge/workload_manifest.json").read_text())["workloads"]
PINNED = {row["id"]: e.pinned_measurement(row)
          for row in json.loads((TASK / "judge/reference_counts.json").read_text())["rows"]}
DEVELOPMENT = [row for row in MANIFEST if row["development"]]
N = 1_000_000


def cell(misp, wp, instr=N):
    return e.Measurement(instr, misp, wp, 2 * instr - 1,
                         Decimal(f"{1000 * misp / instr:.4f}"), Decimal(f"{1000 * wp / instr:.4f}"))


def main_rule(candidate, reference, bits):
    """The rule of origin/main, written out independently of grade()."""
    return int(bits <= 1572864 and all(
        candidate[w].misp_br * reference[w].instr < reference[w].misp_br * candidate[w].instr
        and candidate[w].cyc_wp * reference[w].instr < reference[w].cyc_wp * candidate[w].instr
        for w in reference))


def final_suite(candidate_of, reference=(100_000, 1_000_000)):
    ref = {row["id"]: cell(*reference) for row in MANIFEST}
    cand = {row["id"]: cell(*candidate_of(index)) for index, row in enumerate(MANIFEST)}
    metrics = e.grade(MANIFEST, cand, ref, True, logical_storage_bits=BITS)[3]
    assert metrics["beats_reference"] == metrics["final_beats_reference"] == main_rule(cand, ref, BITS)
    return metrics


CASES = {
    "equal to the reference": (lambda i: (100_000, 1_000_000), 0),
    "one event better on every trace and count": (lambda i: (99_999, 999_999), 1),
    "ten percent better everywhere": (lambda i: (90_000, 900_000), 1),
    "better everywhere, one misprediction tie": (lambda i: (100_000 if i == 50 else 90_000, 900_000), 0),
    "better everywhere, one wrong-path tie": (lambda i: (90_000, 1_000_000 if i == 50 else 900_000), 0),
    "better everywhere, one trace one event worse": (lambda i: (100_001 if i == 104 else 90_000, 900_000), 0),
    "better means, eleven traces worse": (lambda i: (101_000 if i < 11 else 80_000, 800_000), 0),
    "mispredictions better, cycles worse": (lambda i: (90_000, 1_010_000), 0),
    "naive-like, worse on both": (lambda i: (104_000, 1_013_500), 0),
}


@pytest.mark.parametrize("name", list(CASES))
def test_gate_decision_is_the_strict_per_trace_rule_of_main(name):
    candidate_of, expected = CASES[name]
    metrics = final_suite(candidate_of)
    assert metrics["beats_reference"] == expected
    assert metrics["beats_reference"] == int(all(
        v for row in metrics["beat_spec_workload_passes"] for v in row["metrics"].values()))


def test_criteria_and_constants_are_those_of_main():
    assert e.METRICS == ("br_mis_pki", "cyc_wp_pki", "logical_storage_bits")
    for name in ("MEAN_IMPROVEMENT", "TRACE_TOLERANCE"):
        assert not hasattr(e, name)
    spec = json.loads(SPEC.read_text())
    assert [m["name"] for m in spec["metrics"]] == list(e.METRICS)
    assert "Equality fails" in spec["metrics"][0]["clears_when"] and "No mean" in spec["beats_reference"]
    metrics = final_suite(lambda i: (90_000, 900_000))
    for gone in ("significance_gate", "strict_workload_passes", "mean_gate", "tolerance_exceeded_traces", "strict_progress"):
        assert gone not in metrics


def test_real_reference_counts_against_themselves_fail():
    metrics = e.grade(MANIFEST, PINNED, PINNED, True, logical_storage_bits=BITS)[3]
    assert metrics["beats_reference"] == 0 == main_rule(PINNED, PINNED, BITS)


def test_perturbed_real_counts_follow_the_strict_rule():
    # Real reference counters moved a little either way, as a shifted random stream does.
    def moved(scale_of):
        return {wid: e.Measurement(r.instr, int(r.misp_br * scale_of(i)), int(r.cyc_wp * scale_of(i)),
                                   r.full_instr, r.printed_br, r.printed_wp)
                for i, (wid, r) in enumerate(PINNED.items())}
    mixed = moved(lambda i: Fraction(99 if i % 2 else 101, 100))
    better = moved(lambda i: Fraction(97, 100))
    for candidate, expected in ((mixed, 0), (better, 1)):
        metrics = e.grade(MANIFEST, candidate, PINNED, True, logical_storage_bits=BITS)[3]
        assert metrics["beats_reference"] == expected == main_rule(candidate, PINNED, BITS)


def development():
    reference = {row["id"]: PINNED[row["id"]] for row in DEVELOPMENT}
    candidate = {wid: e.Measurement(r.instr, r.misp_br * 97 // 100, r.cyc_wp * 97 // 100, r.full_instr,
                                    r.printed_br, r.printed_wp) for wid, r in reference.items()}
    return e.grade(DEVELOPMENT, candidate, reference, False, logical_storage_bits=BITS), candidate, reference


def test_category_means_are_the_candidates_own_and_nothing_else():
    result, candidate, reference = development()
    shown = feedback.public_metrics(result[3], beat_spec_path=SPEC,
                                    configured_visible=e.CONFIG["evaluation"]["solver_visible_metrics"])
    assert set(shown) == {"train_br_mis_pki_mean", "train_cyc_wp_pki_mean", "all_metrics_progress",
                          "category_means", "beat_spec_progress"}
    assert set(shown["beat_spec_progress"]) == set(e.METRICS)
    exact = lambda rows, field: float(sum(Fraction(1000 * getattr(m, field), m.instr) for m in rows) / len(rows))
    assert {k: v["traces"] for k, v in shown["category_means"].items()} == {
        "compress": 2, "fp": 3, "infra": 4, "int": 8, "media": 1, "web": 6}
    secret = set()
    for category, entry in shown["category_means"].items():
        members = [row["id"] for row in DEVELOPMENT if row["category"] == category]
        assert entry == {"traces": len(members),
                         "br_mis_pki_mean": exact([candidate[i] for i in members], "misp_br"),
                         "cyc_wp_pki_mean": exact([candidate[i] for i in members], "cyc_wp")}
        secret |= {exact([reference[i] for i in members], "misp_br"), exact([reference[i] for i in members], "cyc_wp")}
    for r in reference.values():
        secret |= {r.misp_br, r.cyc_wp, exact([r], "misp_br"), exact([r], "cyc_wp")}
    secret |= {exact(list(reference.values()), "misp_br"), exact(list(reference.values()), "cyc_wp")}

    def walk(value, key=""):
        yield key, value
        if isinstance(value, dict):
            for k, v in value.items():
                yield from walk(v, k)

    nodes = list(walk(shown))
    assert not any(isinstance(v, list) for _, v in nodes)
    assert not {v for _, v in nodes if isinstance(v, (int, float)) and not isinstance(v, bool)} & secret
    encoded = json.dumps(shown)
    for word in ("reference", "target", "ratio", "margin", "final", "beats"):
        assert word not in encoded
    for row in MANIFEST:
        assert row["id"] not in encoded
    spec = json.loads(SPEC.read_text())
    for entry in spec["solver_feedback"]:
        assert entry["reveals_reference"].startswith(("no \u2014 ", "no; ")), entry["name"]


def test_final_metrics_have_no_category_means():
    assert "category_means" not in final_suite(lambda i: (90_000, 900_000))


def test_statements_keep_the_strict_rule_and_state_the_measured_noise():
    sentence = ("Measured on official final evaluations, a change unrelated to component quality (shifting by one call "
                "the shared random stream that the retained predictor uses) moves a single trace's mispredictions by "
                "about 1% (standard deviation; up to 6.6%) and its wrong-path cycles by about 0.7%, so winning every "
                "trace requires an improvement clearly larger than that on each trace.")
    for path in (TASK / "judge/readme", TASK / "agent-base/frontier/task/repo/README.md"):
        text = path.read_text()
        for needle in ("strictly lower value for both quantities on every one of the 105 final traces",
                       "A tie or regression anywhere fails that criterion.", sentence, "category_means"):
            assert needle in text, (path.name, needle)
        for stale in ("remains incomplete", "repeatability remain pending", "mean_gate", "0.002"):
            assert stale not in text, (path.name, stale)


@pytest.mark.skipif(not os.environ.get("RUNLTS_OFFICIAL_METRICS"),
                    reason="set RUNLTS_OFFICIAL_METRICS to private final metrics.json paths (os.pathsep-separated)")
def test_recorded_official_finals_keep_their_recorded_verdict():
    import importlib.util
    script = Path(__file__).resolve().parents[3] / "scripts/maintenance/runlts_predictor/rejudge_metrics.py"
    spec = importlib.util.spec_from_file_location("runlts_rejudge_metrics", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in os.environ["RUNLTS_OFFICIAL_METRICS"].split(os.pathsep):
        row = module.rejudge(e, Path(name))
        print(Path(name).parent.name, row)
        assert row["beats_reference"] == row["recorded_beats_reference"] == int(row["strict_wins_both"] == 105)
