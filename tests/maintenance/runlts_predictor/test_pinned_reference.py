"""The pinned reference table: when it is used, when it is refused, how it is audited.

The shipped table is checked against the task tree and the archived reference
receipts. Evaluations below use invented simulator outputs and a temporary table;
no predictor is built or run.
"""
import deployment_test_guard  # noqa: F401  # isort: skip

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import types

import pytest
from tests.maintenance.runlts_predictor.runtime_imports import MODULES

e = MODULES["evaluator"]
adapter = MODULES["fcs2_verifier"]
policy = MODULES["source_policy"]
apply_submission_patch = MODULES["submission_patch"].apply_submission_patch
REPO = Path(__file__).resolve().parents[3]
GENERATOR = REPO / "scripts/maintenance/runlts_predictor/pin_reference_counts.py"
HEADER = "       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI"
N = 1000000
SHIPPED_TABLE = e.PINNED_REFERENCE


def output(misp=3000, wp=140000):
    row = f"{N} 500000 2.0000 100000 {misp} 0.2000 0.0060 3.0000% {1000 * misp / N:.4f} {wp} 46.6667 {1000 * wp / N:.4f}"
    return "\n".join([
        "--DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (50 Perc instructions)--", HEADER, row,
        "--DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)--",
        HEADER, row])


def workloads():
    return json.loads((e.ROOT / "workload_manifest.json").read_text())["workloads"]


@pytest.fixture
def restored():
    result = apply_submission_patch(e.ROOT / "reference.patch", e.ROOT, e.CONFIG)
    assert result.ok, result.message
    yield result.workdir
    shutil.rmtree(result.cleanup_root)


def refused(reason, suite, tree):
    with pytest.raises(e.EvaluationError) as caught:
        e.pinned_reference(suite, tree)
    assert str(caught.value) == "pinned reference " + reason
    evidence = caught.value.private_execution_evidence
    assert evidence["execution_valid"] is False and evidence["reference_phase"] == "pinned_table"
    return evidence


# ----------------------------------------------------------- the shipped table

def test_shipped_table_is_pinned_and_covers_every_workload(restored):
    assert hashlib.sha256(e.PINNED_REFERENCE.read_bytes()).hexdigest() == e.PINNED_REFERENCE_SHA256
    rows = workloads()
    reference = e.pinned_reference(rows, restored)
    assert list(reference) == [row["id"] for row in rows] and len(reference) == 105
    for row in rows:
        measured = reference[row["id"]]
        assert (measured.instr, measured.full_instr) == (row["measured_instr"], row["full_instr"])
        assert 0 < measured.misp_br and 0 < measured.cyc_wp  # strict improvement stays attainable
    development = [row for row in rows if row["development"]]
    assert set(e.pinned_reference(development, restored)) == {row["id"] for row in development}


def test_shipped_identity_is_the_pinned_reference_and_documents(restored):
    identity = json.loads(e.PINNED_REFERENCE.read_text())["identity"]
    assert identity == e.reference_identity(restored)
    provenance = json.loads((e.ROOT / "provenance.json").read_text())
    assert identity["reference_core_sha256"] == provenance["submission_files"]["my_cond_branch_predictor.cc"]
    assert identity["workload_manifest_sha256"] == provenance["data_checksums"]["manifest_sha256"]
    assert identity["invocation"]["build_argv"] == ["make", "-j1", "OPT=-O3 -D_GLIBCXX_ASSERTIONS"]
    # The restored reference is the tree the logical-storage audit recognises.
    assert policy.audit(restored)["logical_bits"] == policy.REFERENCE_BITS


def test_shipped_table_is_reproduced_from_the_archived_receipts():
    done = subprocess.run([sys.executable, "-B", str(GENERATOR), "--check"], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    assert e.PINNED_REFERENCE_SHA256 in done.stdout


def test_archived_copies_of_the_reference_run_agree_and_match_the_native_score(restored):
    # progress.json and result.recovered.json are copies of the per-trace receipts,
    # not repeat executions. The recorded native reference run agrees in its score.
    doc = json.loads(e.PINNED_REFERENCE.read_text())
    table = {row["id"]: row for row in doc["rows"]}
    receipts = REPO / doc["provenance"]["receipts"]
    for name in ("progress.json", "result.recovered.json"):
        for row in json.loads((receipts / name).read_text())["rows"]:
            half, pinned = row["latter_half"], table[row["id"]]
            assert (half["instr"], half["misp_br"], half["cyc_wp"], row["full"]["instr"]) == (
                pinned["instr"], pinned["misp_br"], pinned["cyc_wp"], pinned["full_instr"])
    native = doc["provenance"]["native_reference_run"]
    assert native["score_equals_table"] is True and native["reference_patch_equals_table"] is True
    assert e.score_of(list(e.pinned_reference(workloads(), restored).values())) == native["score"]


# ------------------------------------------------------ refusals of the table

def test_changed_simulator_or_reference_source_is_an_identity_mismatch(restored):
    suite = workloads()[:1]
    with (restored / "lib/bp.cc").open("a") as stream:
        stream.write("\n")
    assert refused("identity mismatch", suite, restored)["mismatched"] == ["reference_tree_sha256"]
    with (restored / "my_cond_branch_predictor.cc").open("a") as stream:
        stream.write("\n")
    assert refused("identity mismatch", suite, restored)["mismatched"] == ["reference_core_sha256", "reference_tree_sha256"]


def test_added_source_file_or_changed_command_is_an_identity_mismatch(restored, monkeypatch):
    suite = workloads()[:1]
    (restored / "lib/extra.h").write_text("")
    assert refused("identity mismatch", suite, restored)["mismatched"] == ["reference_tree_sha256"]
    (restored / "lib/extra.h").unlink()
    assert e.pinned_reference(suite, restored)
    monkeypatch.setattr(e, "BUILD_ARGV", ["make", "-j1", "OPT=-O2"])
    assert refused("identity mismatch", suite, restored)["mismatched"] == ["invocation"]


@pytest.mark.parametrize("document", ["reference.patch", "integrity.json", "source_contract.json", "workload_manifest.json"])
def test_changed_pinned_document_is_an_identity_mismatch(restored, monkeypatch, tmp_path, document):
    for name in ("reference.patch", "integrity.json", "source_contract.json", "workload_manifest.json"):
        shutil.copy(e.ROOT / name, tmp_path / name)
    with (tmp_path / document).open("a") as stream:
        stream.write("\n")
    monkeypatch.setattr(e, "ROOT", tmp_path)
    assert len(refused("identity mismatch", workloads_of(tmp_path)[:1], restored)["mismatched"]) == 1


def workloads_of(root):
    return json.loads((root / "workload_manifest.json").read_text())["workloads"]


def retable(monkeypatch, tmp_path, change, pin=True):
    doc = json.loads(SHIPPED_TABLE.read_text())
    change(doc)
    path = tmp_path / "reference_counts.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(e, "PINNED_REFERENCE", path)
    if pin:
        monkeypatch.setattr(e, "PINNED_REFERENCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())


def test_edited_or_absent_table_is_refused(restored, monkeypatch, tmp_path):
    suite = workloads()[:1]
    retable(monkeypatch, tmp_path, lambda doc: doc["rows"][0].update(misp_br=doc["rows"][0]["misp_br"] + 1), pin=False)
    refused("table checksum mismatch", suite, restored)
    monkeypatch.setattr(e, "PINNED_REFERENCE", tmp_path / "absent.json")
    refused("table missing", suite, restored)


def test_missing_row_is_refused(restored, monkeypatch, tmp_path):
    rows = workloads()
    retable(monkeypatch, tmp_path, lambda doc: doc["rows"].pop(3))
    refused("table coverage invalid", rows, restored)
    retable(monkeypatch, tmp_path, lambda doc: doc["rows"][3].update(id="int/another_trace"))
    assert refused("row missing", rows, restored)["workload"] == rows[3]["id"]
    assert e.pinned_reference(rows[:3], restored)  # rows that are present still load


@pytest.mark.parametrize("change,reason", [
    (lambda row: row.update(trace_sha256="0" * 64), "row does not match the workload manifest"),
    (lambda row: row.update(path="int/other.gz"), "row does not match the workload manifest"),
    (lambda row: row.update(misp_br=str(row["misp_br"])), "row invalid"),
    (lambda row: row.update(misp_br=row["misp_br"] + 1000), "row invalid"),  # no longer the printed rate
    (lambda row: row.pop("printed_cyc_wp_pki"), "row invalid"),
    (lambda row: row.update(instr=row["instr"] + 1), "row does not match the workload manifest"),
], ids=["trace_sha256", "path", "string_counter", "counter_off_printed_rate", "rate_absent", "instr_off_manifest"])
def test_row_that_is_not_this_trace_or_not_a_measurement_is_refused(restored, monkeypatch, tmp_path, change, reason):
    rows = workloads()
    retable(monkeypatch, tmp_path, lambda doc: change(doc["rows"][0]))
    assert refused(reason, rows[:1], restored)["workload"] == rows[0]["id"]


# ------------------------------------------------------------ whole evaluations

@pytest.fixture
def stage(tmp_path, monkeypatch):
    """_evaluate over a temporary task root with a real, self-consistent table."""
    root = tmp_path / "task"
    root.mkdir()
    for name in ("reference.patch", "integrity.json", "source_contract.json", "workload_manifest.json"):
        (root / name).write_text(name)
    trees = {}
    for name in ("candidate", "reference"):
        tree = tmp_path / name
        (tree / "lib").mkdir(parents=True)
        (tree / "lib/cbp.cc").write_text("// fixture\n")
        (tree / "my_cond_branch_predictor.cc").write_text("// " + name + "\n")
        trees[name] = tree
    suite = [{"id": f"unit/w{i}", "path": f"unit/w{i}.gz", "sha256": f"{i:064x}", "full_instr": N, "measured_instr": N}
             for i in range(105)]
    state = types.SimpleNamespace(root=root, trees=trees, suite=suite[:2], calls=[],
                                  outputs={"candidate": output(2999, 139999), "reference": output()})
    monkeypatch.setattr(e, "ROOT", root)
    table = {"schema": e.PINNED_REFERENCE_SCHEMA, "identity": e.reference_identity(trees["reference"]),
             "rows": [{"id": row["id"], "path": row["path"], "trace_sha256": row["sha256"], "instr": N, "misp_br": 3000,
                       "cyc_wp": 140000, "full_instr": N, "printed_br_mis_pki": "3.0000", "printed_cyc_wp_pki": "140.0000"}
                      for row in suite]}
    def install(doc):
        path = root / "reference_counts.json"
        path.write_text(json.dumps(doc))
        monkeypatch.setattr(e, "PINNED_REFERENCE", path)
        monkeypatch.setattr(e, "PINNED_REFERENCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    state.table, state.install = table, install
    install(table)

    def patched(path, task_root, config):
        side = "reference" if Path(path).name == "reference.patch" else "candidate"
        return types.SimpleNamespace(ok=True, workdir=trees[side], cleanup_root=None)

    def command(argv, cwd, timeout, deadline):
        side = "reference" if Path(cwd) == trees["reference"] else "candidate"
        step = "audit" if argv[0] == sys.executable else "build" if argv[0] == "make" else "trace"
        state.calls.append((side, step))
        if step == "audit":
            return json.dumps({"valid": True, "logical_bits": policy.REFERENCE_BITS, "method": "fixture"})
        return state.outputs[side] if step == "trace" else ""

    monkeypatch.setattr(e, "apply_submission_patch", patched)
    monkeypatch.setattr(e, "run_command", command)
    monkeypatch.setattr(e, "manifest", lambda final: list(state.suite))
    monkeypatch.delenv("FRONTIER_SUBMISSION_ROLE", raising=False)
    monkeypatch.delenv(e.REFERENCE_AUDIT_ENV, raising=False)
    return state


def test_default_evaluation_never_builds_or_runs_the_reference(stage):
    score, _, message, metrics = e.evaluate("candidate.patch")
    assert message == "evaluation complete" and score > 0 and metrics["beats_reference"] == 1
    assert stage.calls == [("candidate", "audit"), ("candidate", "build"), ("candidate", "trace"), ("candidate", "trace")]
    assert [row["role"] for row in metrics["private_runtime_telemetry"]["traces"]] == ["candidate", "candidate"]
    assert metrics["pinned_reference"] == {"table_sha256": e.PINNED_REFERENCE_SHA256, "live_check": "none",
                                           "workloads_equal_to_live_execution": 0}
    # The comparison still uses the reference's exact counters, from the table.
    assert [(row["reference_instr"], row["reference_misp_br"], row["reference_cyc_wp"])
            for row in metrics["paired_measurements"]] == [(N, 3000, 140000)] * 2
    adapter.validated_runtime_telemetry(metrics["private_runtime_telemetry"])


def test_tie_with_the_pinned_counts_does_not_clear_the_strict_comparison(stage):
    stage.outputs["candidate"] = output(3000, 139999)
    metrics = e.evaluate("candidate.patch")[3]
    assert metrics["beats_reference"] == 0
    assert metrics["beat_spec_workload_passes"][0]["metrics"] == {
        "br_mis_pki": False, "cyc_wp_pki": True, "logical_storage_bits": True}


def test_candidate_instruction_count_is_still_paired_against_the_pinned_value(stage):
    with pytest.raises(e.CandidateError) as caught:
        e.grade(stage.suite[:1], {"unit/w0": e.Measurement(N - 1, 1, 1, N, e.Decimal("0.0010"), e.Decimal("0.0010"))},
                e.pinned_reference(stage.suite[:1], stage.trees["reference"]), False,
                logical_storage_bits=policy.REFERENCE_BITS)
    assert caught.value.category == "instruction_count_mismatch"


@pytest.mark.parametrize("role", ["development", "final"])
def test_identity_mismatch_is_an_infrastructure_error_not_a_candidate_score(stage, monkeypatch, role):
    monkeypatch.setenv("FRONTIER_SUBMISSION_ROLE", role)
    (stage.trees["reference"] / "lib/cbp.cc").write_text("// a different simulator\n")
    with pytest.raises(e.EvaluationError, match="pinned reference identity mismatch") as caught:
        e.evaluate("candidate.patch")
    evidence = caught.value.private_execution_evidence
    assert evidence["mismatched"] == ["reference_tree_sha256"] and "private_runtime_telemetry" in evidence
    assert ("candidate", "build") not in stage.calls and ("candidate", "trace") not in stage.calls


def test_missing_row_is_an_infrastructure_error(stage):
    stage.table["rows"][1]["id"] = "unit/renamed"
    stage.install(stage.table)
    with pytest.raises(e.EvaluationError, match="pinned reference row missing") as caught:
        e.evaluate("candidate.patch")
    assert caught.value.private_execution_evidence["workload"] == "unit/w1"
    stage.suite = stage.suite[:1]
    assert e.evaluate("candidate.patch")[2] == "evaluation complete"


def test_live_audit_passes_when_the_reference_reproduces_the_table(stage, monkeypatch):
    monkeypatch.setenv(e.REFERENCE_AUDIT_ENV, "live")
    score, _, message, metrics = e.evaluate("candidate.patch")
    assert message == "evaluation complete" and metrics["beats_reference"] == 1
    assert stage.calls == [("candidate", "audit"), ("reference", "build"), ("candidate", "build"),
                           ("reference", "trace"), ("candidate", "trace"), ("reference", "trace"), ("candidate", "trace")]
    assert metrics["pinned_reference"]["live_check"] == "live_reference"
    assert metrics["pinned_reference"]["workloads_equal_to_live_execution"] == 2
    monkeypatch.delenv(e.REFERENCE_AUDIT_ENV)
    assert e.evaluate("candidate.patch")[0] == score  # the audit does not change the result


@pytest.mark.parametrize("live,field", [((3001, 140000), "misp_br"), ((3000, 139999), "cyc_wp")])
def test_live_audit_fails_as_infrastructure_when_one_counter_differs(stage, monkeypatch, live, field):
    monkeypatch.setenv(e.REFERENCE_AUDIT_ENV, "live")
    stage.outputs["reference"] = output(*live)
    with pytest.raises(e.EvaluationError, match="pinned reference counts differ from live reference execution") as caught:
        e.evaluate("candidate.patch")
    evidence = caught.value.private_execution_evidence
    assert evidence["reference_phase"] == "pinned_audit" and evidence["audit_source"] == "live_reference"
    assert evidence["workload"] == "unit/w0" and field in evidence["fields"]
    assert evidence["fields"][field]["pinned"] != evidence["fields"][field]["live"]
    assert ("candidate", "trace") not in stage.calls


def test_unknown_audit_mode_is_refused(stage, monkeypatch):
    monkeypatch.setenv(e.REFERENCE_AUDIT_ENV, "yes")
    with pytest.raises(e.EvaluationError, match="unknown reference audit mode"):
        e.evaluate("candidate.patch")


def test_reference_submission_is_checked_against_the_table_without_a_second_run(stage):
    # The reference qualification run: the candidate tree is the reference tree.
    (stage.trees["candidate"] / "my_cond_branch_predictor.cc").write_text("// reference\n")
    stage.outputs["candidate"] = output()
    metrics = e.evaluate("reference.as.candidate.patch")[3]
    assert metrics["beats_reference"] == 0 and metrics["margin"] == 0
    assert metrics["pinned_reference"]["live_check"] == "reference_submission"
    assert metrics["pinned_reference"]["workloads_equal_to_live_execution"] == 2
    assert ("reference", "trace") not in stage.calls and ("reference", "build") not in stage.calls
    stage.outputs["candidate"] = output(2999, 140000)
    with pytest.raises(e.EvaluationError, match="pinned reference counts differ from live reference execution") as caught:
        e.evaluate("reference.as.candidate.patch")
    assert caught.value.private_execution_evidence["audit_source"] == "reference_submission"


def test_adapter_reports_a_refused_table_as_a_failed_verification(stage, monkeypatch, tmp_path):
    zstandard = pytest.importorskip("zstandard")
    (stage.trees["reference"] / "lib/cbp.cc").write_text("// a different simulator\n")
    payload = tmp_path / "payload.tar"
    payload.write_bytes(b"fixture-only")
    result_path, evidence_path = tmp_path / "result.json", tmp_path / "evidence.tar.zst"
    scratch = tmp_path / "adapter-scratch"
    scratch.mkdir()
    for key, value in {"FRONTIER_VERIFIER_SCRATCH": scratch, "FRONTIER_RESULT_PATH": result_path,
                       "FRONTIER_EVIDENCE_PATH": evidence_path, "FRONTIER_PAYLOAD_PATH": payload,
                       "FRONTIER_DATASET_PATH": tmp_path / "dataset.json", "FRONTIER_VERIFICATION_MODE": "train"}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(adapter, "extract_payload", lambda *a: None)
    monkeypatch.setattr(adapter, "entrypoint_path", lambda *a: tmp_path / "solution.patch")
    monkeypatch.setattr(adapter, "stage_task", lambda *a: tmp_path)
    monkeypatch.setattr(adapter, "load_evaluator", lambda *a: e.evaluate)
    assert adapter.main(tmp_path, "payload/solution.patch") == 0
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed" and "raw_score" not in result and result["metrics"] == {}
    assert "candidate_diagnostic" not in result_path.read_text()
    raw = zstandard.ZstdDecompressor().decompress(evidence_path.read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        private = json.loads(archive.extractfile("metrics.json").read())["metrics"]["private_execution"]
    assert private["reference_phase"] == "pinned_table" and private["mismatched"] == ["reference_tree_sha256"]
