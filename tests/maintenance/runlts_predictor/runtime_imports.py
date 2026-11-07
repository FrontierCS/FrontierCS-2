"""Load task modules without leaking generic names into repository test imports.

The judge is imported as the Verifier image holds it (task_manifest.judge_bundle: judge/ with
the generated config.yaml and the starter at pristine/), beside the shared verifier and the
shared edit scope.
"""
from contextlib import contextmanager
import importlib
import math
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASK = REPOSITORY_ROOT / "tasks/frontier-cs-2.0-demo/problems/Arch/runlts_predictor"
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
import task_manifest  # noqa: E402

JUDGE = task_manifest.judge_bundle(TASK)
NAMES = ("evaluator", "source_policy", "beat_spec_feedback", "fcs2_verifier", "verify", "edit_scope")


def load_runtime():
    old_path = list(sys.path)
    saved = {name: sys.modules.get(name) for name in NAMES}
    try:
        for name in NAMES:
            sys.modules.pop(name, None)
        sys.path[:0] = [str(JUDGE), str(task_manifest.SHARED / "verifier"), str(task_manifest.SHARED / "lib")]
        return {name: importlib.import_module(name) for name in
                ("evaluator", "source_policy", "beat_spec_feedback", "fcs2_verifier", "edit_scope")}
    finally:
        sys.path[:] = old_path
        for name in NAMES:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


MODULES = load_runtime()
EDIT_SCOPE = MODULES["edit_scope"]
# The shared edit scope with this task's task.toml [submission] rules, as the verifier runs it.
RULES = EDIT_SCOPE.Rules.from_mapping(task_manifest.edit_scope_rules(task_manifest.read_task(TASK)))
PRISTINE = MODULES["evaluator"].PRISTINE
# The design record of the beat spec (metric names, withheld fields), archived with the
# provenance; the train-visible metrics are task.toml [feedback].
DESIGN_SPEC = REPOSITORY_ROOT / "docs/task-authoring/runlts_predictor/provenance/beat_spec.json"
FEEDBACK = list(task_manifest.read_task(TASK)["feedback"])
FAILURE_STAGES = ("semantic_audit", "storage_cap", "source_audit_incomplete", "build_failed",
                  "simulator_run_failed", "output_format", "instruction_count_mismatch")


def failure_metrics(category):
    """The train metrics of a candidate failure, written out independently of the evaluator:
    24 train rows failing every comparison and the [feedback] candidate_failure code of the
    stage (1..7 in task.toml's order, 8 for any other candidate failure)."""
    names = ("br_mis_pki", "cyc_wp_pki", "logical_storage_bits")
    return {"beats_reference": 0, "beat_spec_workload_passes": [{"workload": f"t{i:03d}", "metrics": dict.fromkeys(names, False)}
                                          for i in range(24)],
            "candidate_failure": FAILURE_STAGES.index(category) + 1 if category in FAILURE_STAGES else 8}


@contextmanager
def judged(patch: bytes):
    """What the shared verifier does with a patch before evaluate(): the shared edit scope and,
    when it accepts, the judge's check_tree hook. Yields (scope, check_tree verdict)."""
    scope = EDIT_SCOPE.check(patch, PRISTINE, RULES)
    try:
        verdict = (MODULES["evaluator"].check_tree(str(scope.tree), str(PRISTINE), list(scope.changes))
                   if scope.ok else None)
        yield scope, verdict
    finally:
        scope.dispose()


def check_runtime_telemetry(value):
    """The shape of the evaluator's private runtime telemetry (runtime_snapshot). This was the
    validator of the former task-forked verifier; the shared verifier keeps the telemetry in
    evidence unvalidated, so the shape is now checked here, on the evaluator's output."""
    fields = {"schema", "submission_role", "elapsed_seconds", "traces", "longest_trace",
              "verifier_cgroup_memory_peak", "child_process_rss_peak"}
    assert isinstance(value, dict) and set(value) == fields
    assert value["schema"] == "runlts.private-runtime-telemetry/v1"
    assert value["submission_role"] in ("train", "development", "final")

    def elapsed(number):
        return type(number) in (int, float) and math.isfinite(number) and number >= 0
    rows = value["traces"]
    assert elapsed(value["elapsed_seconds"]) and isinstance(rows, list) and len(rows) <= 210
    for row in rows:
        assert isinstance(row, dict) and set(row) == {"role", "workload", "elapsed_seconds", "completed"}
        assert row["role"] in ("reference", "candidate")
        assert isinstance(row["workload"], str) and 0 < len(row["workload"]) <= 256
        assert elapsed(row["elapsed_seconds"]) and type(row["completed"]) is bool
    assert value["longest_trace"] == (max(rows, key=lambda row: row["elapsed_seconds"]) if rows else None)
    memory = value["verifier_cgroup_memory_peak"]
    assert memory.get("scope") == "Verifier cgroup high-water mark, including page cache and wrapper processes"
    if memory.get("status") == "available":
        assert set(memory) == {"scope", "status", "bytes"} and type(memory["bytes"]) is int and memory["bytes"] >= 0
    else:
        assert memory.get("status") == "unavailable" and set(memory) == {"scope", "status"}
    rss = value["child_process_rss_peak"]
    assert set(rss) == {"scope", "bytes"} and type(rss["bytes"]) is int and rss["bytes"] >= 0
    assert rss["scope"] == "Linux RUSAGE_CHILDREN lifetime maximum child-process RSS; excludes cgroup page cache"
    return value


def write_payload(path: Path, patch: bytes) -> Path:
    """A submission archive as the platform hands it to the verifier: payload/solution.patch."""
    import io
    import tarfile
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo("payload/solution.patch")
        member.size = len(patch)
        archive.addfile(member, io.BytesIO(patch))
    return path
