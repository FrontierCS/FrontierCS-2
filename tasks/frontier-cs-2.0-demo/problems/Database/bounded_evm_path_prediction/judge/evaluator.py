"""bounded_evm_path_prediction judge: evaluate(tree) for the shared FrontierCS2 verifier.

Scoring. The judged tree (pristine/ + the submitted core/vm/predictor.go), the reference
(upstream_predictor.go) and the always-taken calibration predictor (baseline_naive.go) are
built into SeerEVM and replay the bundled Ethereum snapshot: 200 blocks for train, 600 for
final. From the printed `T && B` records of the scored epoch (block 200 in train, 400 in
final): Q = 100 * (0.8 * mean B + 0.2 * mean T); with Q0 the calibration's quality,
raw_score = score_unbounded = 100 * max(0, (Q - Q0) / (100 - Q0))^2 (0 if Q0 >= 100).

Metrics (one row per 200-block epoch x {complex, normal} category; 2 rows in train, 6 in final):
  branch_accuracy            sum of printed branch hundredths, higher is better; passes at
                             >= reference + 1 with equal coverage.
  transaction_path_accuracy  sum of printed transaction-path hundredths, higher is better;
                             passes at >= reference with equal coverage.
  report_coverage            observation count; passes when equal to the reference's (> 0).
beats_reference = 1 iff every metric passes on every row. Train also returns raw_quality,
branch_accuracy, transaction_path_accuracy and observations of the scored epoch.

Every build and replay (candidate, reference and calibration alike) runs in the shared test
sandbox: a fresh instance that reads only the system and its own copy of the sources or
the snapshot, gets a clean environment and leaves nothing running.

The shared verifier has already applied the patch with the shared edit scope; check_tree
keeps the source policy (UTF-8, package vm, import allowlist).
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import isolate  # the shared launcher, beside edit_scope in the Verifier base

TASK_DIR = Path(__file__).resolve().parent
PRISTINE = TASK_DIR / "pristine"
PREDICTOR = "core/vm/predictor.go"
DATASET = TASK_DIR / "judge_data" / "ethereumdata_1465_small"
REFERENCE_SOURCE = TASK_DIR / "upstream_predictor.go"
DEV_BLOCKS = 200
FINAL_BLOCKS = 400
BUILD_TIMEOUT = 300
RUN_TIMEOUT = 420
GO_ENV = {"GOTOOLCHAIN": "local", "GOCACHE": "{home}/go-cache", "GOPATH": "{home}/go-path",
          "GOFLAGS": "-mod=vendor"}
SAFE_IMPORTS = {"math", "math/bits", "sort", "sync", "sync/atomic"}
# Preserve the historical finite 0.xx/1.xx domain for transaction satisfaction.
# The upstream normal denominator subtracts all high-gas receipts from contract
# calls, including creations that were never counted as calls; it is not a
# probability. Branch accuracy still has matching numerator/denominator domains.
PAIR_RE = re.compile(r"^([01]\.\d{2})\s*&&\s*(0\.\d{2}|1\.00)$")
HEADER_RE = re.compile(r"^===== Block height (\d+) =====$")
WORKLOADS = ("complex", "normal")
BEAT_METRICS = ("branch_accuracy", "transaction_path_accuracy", "report_coverage")
# Beat spec v3: the paper reports prediction accuracy per 200-block interval
# (Section 5.2, Figure 10). Development grades its single epoch; final grades
# every epoch of its replay, i.e. the freshly replayed public prefix (blocks
# 1-200) and the held-out epoch (blocks 201-400), each category separately.
# The continuous score is unchanged and still uses only the scored epoch.
# Final replays 600 blocks so a second held-out interval (blocks 401-600) is
# graded as well; FINAL_BLOCKS stays the continuous-score epoch.
FINAL_REPLAY_BLOCKS = 600
DEV_EPOCHS = (DEV_BLOCKS,)
FINAL_EPOCHS = (DEV_BLOCKS, FINAL_BLOCKS, FINAL_REPLAY_BLOCKS)


class ReferenceFailure(RuntimeError):
    """The paired reference or calibration replay did not produce a valid report.

    An evaluator error, never a candidate result: a failed reference cannot set
    a bar and the candidate is not scored against it.
    """


def _graded_epochs(blocks: int) -> tuple[int, ...]:
    return FINAL_EPOCHS if blocks == FINAL_REPLAY_BLOCKS else DEV_EPOCHS


def _workload_name(epoch: int, category: str) -> str:
    return f"epoch{epoch}-{category}"
_REFERENCE_CACHE: dict[int, tuple[float, dict[str, Any]]] = {}
_BASELINE_CACHE: dict[int, tuple[float, dict[str, Any]]] = {}


def _invalid(reason: str) -> tuple[float, float, str, dict]:
    return 0.0, 0.0, reason, {
        "beats_reference": 0,
        "reference_score": 0.0,
        "margin": 0.0,
        "beat_spec_workload_passes": _failed_rows(),
    }


def _role_blocks() -> int:
    return FINAL_REPLAY_BLOCKS if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else DEV_BLOCKS


def _failed_rows() -> list[dict[str, Any]]:
    return [{"workload": _workload_name(epoch, name), "metrics": dict.fromkeys(BEAT_METRICS, False)}
            for epoch in _graded_epochs(_role_blocks()) for name in WORKLOADS]


def _beat_rows(candidate: dict[str, Any], reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare each graded epoch and transaction class using exact printed ticks."""
    candidate_epochs = candidate["epoch_measurements"]
    reference_epochs = reference["epoch_measurements"]
    if set(candidate_epochs) != set(reference_epochs):
        raise ValueError("prediction report lacks a required epoch")
    rows = []
    for epoch in sorted(reference_epochs, key=int):
        candidate_rows = candidate_epochs[epoch]
        reference_rows = reference_epochs[epoch]
        if set(candidate_rows) != set(WORKLOADS) or set(reference_rows) != set(WORKLOADS):
            raise ValueError("prediction report lacks a required transaction category")
        for name in WORKLOADS:
            own, ref = candidate_rows[name], reference_rows[name]
            coverage = own["observations"] == ref["observations"] > 0
            rows.append({"workload": _workload_name(int(epoch), name), "metrics": {
                "branch_accuracy": coverage and own["branch_ticks"] >= ref["branch_ticks"] + 1,
                "transaction_path_accuracy": coverage and own["transaction_ticks"] >= ref["transaction_ticks"],
                "report_coverage": coverage,
            }})
    return rows


def _category_ticks(categories: list[list[tuple[float, float]]]) -> dict[str, dict[str, int]]:
    return {
        name: {"observations": len(records),
               "transaction_ticks": sum(round(tx * 100) for tx, _ in records),
               "branch_ticks": sum(round(branch * 100) for _, branch in records)}
        for name, records in zip(WORKLOADS, categories)
    }


def _validate_source(source: str) -> str | None:
    if not re.match(r"\s*package\s+vm\b", source):
        return "predictor must remain in package vm"
    if not set(_imports(source)) <= SAFE_IMPORTS:
        return "predictor imports a package outside the safe allowlist"
    return None


_GO_TOKEN = re.compile(r"""
    (?P<space>[ \t\r\n]+) | (?P<line>//[^\n]*) | (?P<block>/\*.*?\*/)
  | (?P<string>"(?:[^"\\\n]|\\.)*") | (?P<raw>`[^`]*`) | (?P<rune>'(?:[^'\\\n]|\\.)+')
  | (?P<ident>[^\W\d]\w*) | (?P<other>.)""", re.S | re.X)


def _imports(source: str) -> list[str]:
    """The import paths of a Go file: the string literals of its import declarations, which
    Go only accepts right after the package clause. A later `import` keyword (a compile error)
    is returned as the path "import" so the allowlist rejects it."""
    tokens = [(m.lastgroup, m.group()) for m in _GO_TOKEN.finditer(source)
              if m.lastgroup not in ("space", "line", "block")]
    i = 2 if tokens[:1] == [("ident", "package")] else 0
    paths = []
    while i < len(tokens) and tokens[i] in (("other", ";"), ("ident", "import")):
        if tokens[i] == ("other", ";"):
            i += 1
            continue
        i += 1
        group = i < len(tokens) and tokens[i] == ("other", "(")
        i += group
        while i < len(tokens):
            kind, text = tokens[i]
            i += 1
            if group and (kind, text) == ("other", ")"):
                break
            if kind in ("string", "raw"):
                paths.append(text[1:-1])
                if not group:
                    break
    return paths + ["import" for token in tokens[i:] if token == ("ident", "import")]


def _adapt_upstream_source(source: str) -> str:
    """Rename integration symbols without changing the upstream algorithm."""
    return (
        source.replace("Perceptron", "Predictor")
        .replace("perceptron", "predictor")
        .replace("HISTORY_LENGTH_MIN", "WARMUP_OBSERVATIONS")
        .replace("historyLength", "observationCount")
    )


def _parse_report(report: Path, blocks: int = DEV_BLOCKS) -> tuple[float, dict[str, Any]]:
    if not report.is_file() or report.stat().st_size > 2_000_000:
        raise ValueError("prediction report is missing or invalid")
    groups: dict[int, list[list[tuple[float, float]]]] = {}
    current_height: int | None = None
    section = 0
    for line in report.read_text("utf-8", errors="strict").splitlines():
        header = HEADER_RE.fullmatch(line.strip())
        if header:
            current_height = int(header.group(1))
            if current_height in groups:
                raise ValueError("prediction report repeats an epoch")
            groups[current_height] = [[], []]
            section = 0
            continue
        match = PAIR_RE.fullmatch(line.strip())
        if match:
            tx_accuracy, branch_accuracy = map(float, match.groups())
            if section == 0 and tx_accuracy > 1.0:
                raise ValueError("prediction report contains an invalid accuracy record")
            if current_height is None:
                current_height = blocks
                groups[current_height] = [[], []]
            groups[current_height][section].append((tx_accuracy, branch_accuracy))
        elif not line.strip() and current_height is not None and groups[current_height][0]:
            section = 1
        elif "&&" in line:
            raise ValueError("prediction report contains an invalid accuracy record")
    final = blocks == FINAL_REPLAY_BLOCKS
    # The continuous score keeps its v2 epoch: 200 for development and 400
    # (blocks 201-400) for final, even though final now replays 600 blocks.
    scored = FINAL_BLOCKS if final else blocks
    if any(epoch not in groups for epoch in _graded_epochs(blocks)):
        raise ValueError("prediction report lacks a required epoch")
    pairs = groups[scored][0] + groups[scored][1]
    scored_start = DEV_BLOCKS + 1 if final else 1
    if len(pairs) < 20:
        raise ValueError("prediction report contains too few observations")
    categories = groups[scored]
    if not all(categories):
        raise ValueError("prediction report lacks a required transaction category")
    if any(not all(groups[epoch]) for epoch in _graded_epochs(blocks)):
        raise ValueError("prediction report lacks a required transaction category")
    mean_tx = sum(x for x, _ in pairs) / len(pairs)
    mean_branch = sum(x for _, x in pairs) / len(pairs)
    # Branch accuracy is the paper experiment's primary measure. Preserve the
    # upstream transaction satisfaction ratio as the smaller quality term,
    # including normal-category values above one, without clamping.
    quality = 100.0 * (0.8 * mean_branch + 0.2 * mean_tx)
    return quality, {
        "branch_accuracy": mean_branch,
        "transaction_path_accuracy": mean_tx,
        "observations": len(pairs),
        "scored_block_start": scored_start,
        "scored_block_end": scored,
        # Private exact hundredth sums retain the upstream printer's resolution.
        # Only aggregate candidate metrics are allowed through train feedback.
        "category_measurements": _category_ticks(categories),
        # Beat spec v3: exact ticks for every graded epoch (string keys keep
        # the evidence JSON stable). The prefix epoch must be complete too.
        "epoch_measurements": {
            str(epoch): _category_ticks(groups[epoch]) for epoch in _graded_epochs(blocks)
        },
    }


def _sandboxed(argv: list[str], **options: Any) -> isolate.Result:
    """One build or replay instance; a timeout is the candidate's."""
    result = isolate.run(argv, env=GO_ENV, **options)
    if result.timed_out:
        raise ValueError("SeerEVM build or replay timed out")
    return result


def _run_real(source: str, blocks: int) -> tuple[float, dict[str, Any]]:
    problem = _validate_source(source)
    if problem:
        raise ValueError(problem)
    if not (DATASET / "copychain").is_dir():
        raise ValueError("benchmark dataset is unavailable")
    root = Path(tempfile.mkdtemp(prefix="seer-real-"))
    app = root / "app"
    try:
        shutil.copytree(PRISTINE, app)
        (app / PREDICTOR).write_text(source, "utf-8")
        shutil.copytree(TASK_DIR / "judge_data" / "vendor", app / "vendor")
        build = _sandboxed(["go", "build", "-buildvcs=false", "-o", "{out}/seerevm", "."],
                           tree=app, outputs=["seerevm"], output_bytes=1 << 30,
                           timeout_seconds=BUILD_TIMEOUT)
        binary = build.outputs.get("seerevm")
        if not build.ok or not binary:
            raise ValueError("predictor does not compile in SeerEVM")
        shutil.rmtree(app)
        (root / "bin").mkdir()
        (root / "bin" / "seerevm").write_bytes(binary)
        (root / "bin" / "seerevm").chmod(0o555)
        # The replay runs in experiments/ beside its own writable copy of the snapshot (the
        # binary opens ../ethereumdata_1465_small) and writes prediction_height.txt there,
        # linked to the instance's output.
        (root / "replay" / "experiments").mkdir(parents=True)
        shutil.copytree(DATASET, root / "replay" / DATASET.name, symlinks=True)
        run = _sandboxed(["sh", "-c", 'ln -s "$1" prediction_height.txt && shift && exec "$@" >/dev/null 2>&1',
                          "sh", "{out}/prediction_height.txt", str(root / "bin" / "seerevm"),
                          "--indicator", "1", "--start", "14650000", "--blockNum", str(blocks)],
                         tree=root / "replay", cwd="{work}/experiments",
                         read_paths=[root / "bin"], outputs=["prediction_height.txt"],
                         timeout_seconds=RUN_TIMEOUT)
        shutil.rmtree(root / "replay")
        if run.returncode != 0:
            raise ValueError("SeerEVM replay failed")
        report = root / "prediction_height.txt"
        if run.outputs.get("prediction_height.txt") is not None:
            report.write_bytes(run.outputs["prediction_height.txt"])
        return _parse_report(report, blocks)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _reference(blocks: int) -> tuple[float, dict[str, Any]]:
    if blocks not in _REFERENCE_CACHE:
        try:
            source = _adapt_upstream_source(REFERENCE_SOURCE.read_text("utf-8"))
            _REFERENCE_CACHE[blocks] = _run_real(source, blocks)
        except Exception as error:
            raise ReferenceFailure(f"reference replay failed: {type(error).__name__}") from error
    return _REFERENCE_CACHE[blocks]


def _baseline(blocks: int) -> tuple[float, dict[str, Any]]:
    if blocks not in _BASELINE_CACHE:
        try:
            source = (TASK_DIR / "baseline_naive.go").read_text("utf-8")
            _BASELINE_CACHE[blocks] = _run_real(source, blocks)
        except Exception as error:
            raise ReferenceFailure(f"calibration replay failed: {type(error).__name__}") from error
    return _BASELINE_CACHE[blocks]


def _bounded_score(quality: float, baseline_quality: float) -> float:
    if baseline_quality >= 100.0:
        return 0.0
    normalized = max(0.0, (quality - baseline_quality) / (100.0 - baseline_quality))
    return max(0.0, min(100.0, 100.0 * normalized * normalized))


def prepare() -> dict:
    blocks = _role_blocks()
    quality, _ = _reference(blocks)
    baseline_quality, _ = _baseline(blocks)
    score = _bounded_score(quality, baseline_quality)
    return {"ready": True, "blocks": blocks, "reference_valid": math.isfinite(score)}


def check_tree(tree: str, pristine: str, changes: list) -> tuple[str, list[str]] | None:
    """The source policy task.toml cannot express: valid UTF-8, package vm and the import
    allowlist (the 100,000-byte cap is task.toml's max_file_bytes, enforced by the shared edit
    scope)."""
    path = Path(tree) / PREDICTOR
    try:
        source = path.read_text("utf-8")
    except UnicodeDecodeError:
        return "content_policy", [PREDICTOR]
    if _validate_source(source) is not None:
        return "content_policy", [PREDICTOR]
    return None


def evaluate(tree: str) -> tuple[float, float, str, dict]:
    try:
        source = (Path(tree) / PREDICTOR).read_text("utf-8")
        blocks = _role_blocks()
        reference_source = _adapt_upstream_source(REFERENCE_SOURCE.read_text("utf-8"))
        if source == reference_source:
            candidate_quality, public = _reference(blocks)
            reference_quality = candidate_quality
            reference_public = public
        else:
            candidate_quality, public = _run_real(source, blocks)
            reference_quality, reference_public = _reference(blocks)
        baseline_quality, _ = _baseline(blocks)
        candidate = _bounded_score(candidate_quality, baseline_quality)
        reference = _bounded_score(reference_quality, baseline_quality)
        rows = _beat_rows(public, reference_public)
        metrics = {
            "beats_reference": int(all(all(row["metrics"].values()) for row in rows)),
            "beat_spec_workload_passes": rows,
            "reference_score": reference,
            "margin": candidate - reference,
            "raw_quality": candidate_quality,
            **public,
        }
        return candidate, candidate, f"scored {blocks}-block Ethereum replay", metrics
    except (ReferenceFailure, isolate.IsolateError):
        raise
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) else "submission could not be evaluated"
        return _invalid(message[:120])


if __name__ == "__main__":
    try:
        result = evaluate(sys.argv[1])
        print(json.dumps({"score": result[0], "score_unbounded": result[1], "message": result[2], "metrics": result[3]}, sort_keys=True))
    except Exception:
        result = _invalid("invalid invocation")
        print(json.dumps({"score": result[0], "score_unbounded": result[1], "message": result[2], "metrics": result[3]}))
