import deployment_test_guard  # noqa: F401

import sys
from pathlib import Path

TASK = (
    Path(__file__).resolve().parents[3]
    / "tasks/problems/Bioinformatics/edlib_exact_alignment"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "tasks/shared/lib")
)  # isolate, as in the Verifier base
import task_manifest  # noqa: E402

sys.path.insert(0, str(task_manifest.judge_bundle(TASK)))
"""Affected resource and workload contracts; not scientific measurements."""
import evaluator as e


def test_changed_workload_full_membership():
    public = e.rows(False)
    final = e.rows(True)
    expected = {
        (n, s, m)
        for n in (10000, 30000)
        for s in (60, 70, 80, 90, 99)
        for m in ("NW", "SHW", "HW")
    }
    assert len(public) == 30 and len(final) == 60
    assert {x["id"] for x in public} < {x["id"] for x in final}
    for split in ("public", "heldout"):
        selected = [x for x in final if x["split"] == split]
        assert len(selected) == 30
        assert {(x["length"], x["similarity"], x["mode"]) for x in selected} == expected
    assert all(x["split"] == "public" for x in public)


def test_changed_fixture_hashes():
    from judge.workloads import digest, generate

    for row in e.rows(True):
        q, t = generate(row)
        assert (digest(q), digest(t)) == (row["query_sha256"], row["target_sha256"])
