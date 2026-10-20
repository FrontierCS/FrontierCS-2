"""Native evaluator for localization-guided regex repair reconstruction."""
from __future__ import annotations

import atexit
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

from submission_patch import apply_submission_patch
import yaml

TASK_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((TASK_DIR / "config.yaml").read_text(encoding="utf-8"))
DATA = TASK_DIR / "workloads.json"
ASSETS = TASK_DIR / "judge_assets"
METRIC_NAMES = ("vulnerability_elimination_rate", "repair_rate", "well_formed_rate", "rli", "nls", "jaccard", "precision", "recall")
AUTHENTIC_ENTRYPOINT = "scripts/inference.py"
_runtime_state: dict[str, object] = {"server": None}
_reference_cache: dict[str, dict[str, dict[str, float]]] = {}

def _fail(reason: str):
    return 0.0, 0.0, reason[:240], {}


def _endpoint() -> str:
    server = _runtime_state["server"]
    if server is None:
        cmd = [
            "/app/llama-server", "-m", "/models/Qwen3-8B-Q4_K_M.gguf",
            "--device", "CUDA0", "--n-gpu-layers", "99",
            "-lv", "5",
            "--host", "127.0.0.1", "--port", "18081", "-t", "4", "-c", "8192",
        ]
        log = tempfile.TemporaryFile(mode="w+t")
        env = dict(os.environ, LD_LIBRARY_PATH="/app:/usr/local/cuda/lib64")
        server = subprocess.Popen(cmd, cwd="/app", env=env, stdout=subprocess.DEVNULL, stderr=log, text=True)
        _runtime_state["server"] = server
        _runtime_state["log"] = log
        atexit.register(server.terminate)
    url = "http://127.0.0.1:18081"
    for _ in range(180):
        if server.poll() is not None:
            log = _runtime_state["log"]
            log.seek(0)
            detail = log.read()[-400:]
            raise RuntimeError("fixed model server exited during startup: " + detail)
        try:
            urllib.request.urlopen(url + "/health", timeout=1).read()
            if "gpu_backend" not in _runtime_state:
                log = _runtime_state["log"]
                startup = os.pread(log.fileno(), min(os.fstat(log.fileno()).st_size, 1048576), 0).decode("utf-8", "replace")
                offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", startup)
                if offload is None or int(offload[1]) == 0 or offload[1] != offload[2]:
                    raise RuntimeError("fixed model did not confirm complete GPU layer offload")
                _runtime_state["gpu_backend"] = {"device": "CUDA0", "offloaded_layers": int(offload[1]), "total_layers": int(offload[2])}
            return url
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("fixed model server did not become ready")


class PolicyError(Exception):
    """The submitted prompt policy failed; a candidate failure, never infra."""


CONSTRUCTOR_TIMEOUT_SECONDS = 600


SANDBOX_ISOLATION_UNAVAILABLE = 70
SANDBOX_BAD_ARGUMENTS = 64


def _policy_input(item_input: dict) -> dict:
    """Exactly the inputs the policy may use: the original regex and the
    first-step RegexScalpel localization. Every other field is cleared."""
    result = item_input.get("result")
    if result is not None:
        repairs = result.get("repairs") or []
        first = repairs[0] if repairs else None
        result = {
            "regex": item_input["pattern"], "final_regex": item_input["pattern"],
            "repairs": [] if first is None else [{
                **{key: None for key in first}, "subpatterns": first.get("subpatterns"),
            }],
        }
    return {"pattern": item_input["pattern"], "error": item_input.get("error"), "result": result}


def _render_chats(repo, task: str, lines: list[str]) -> list:
    """Run the retained constructor over every input in an isolated process.

    policy_sandbox.py gives it a view without the judge, the verifier I/O or
    any scratch state, and its PID namespace ends every process it started.
    """
    bundle = Path(os.path.realpath(tempfile.mkdtemp(prefix="lrr-policy-", dir="/tmp")))
    try:
        shutil.copytree(repo, bundle / "tree", symlinks=True)
        shutil.copyfile(TASK_DIR / "constructor_worker.py", bundle / "worker.py")
        (bundle / "request.json").write_text(json.dumps({"task": task, "lines": lines}), encoding="utf-8")
        for path in [bundle, *bundle.rglob("*")]:
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)
        proc = subprocess.Popen(
            # The platform copies TASK_DIR under /workspace, which the sandbox hides.
            [sys.executable, "-I", str(TASK_DIR / "policy_sandbox.py"), "/opt/frontier/task", str(bundle)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
        try:
            out, err = proc.communicate(timeout=CONSTRUCTOR_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            out, err = "", "prompt constructor exceeded its time limit"
        finally:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            proc.wait()
    finally:
        shutil.rmtree(bundle, ignore_errors=True)
    if proc.returncode in (SANDBOX_ISOLATION_UNAVAILABLE, SANDBOX_BAD_ARGUMENTS):
        raise RuntimeError("policy sandbox unavailable: " + (err or "").strip()[-200:])
    if proc.returncode != 0 or not out:
        tail = (err or "").strip().splitlines()[-1:] or ["no diagnostic"]
        raise PolicyError("prompt constructor failed: " + tail[0][:160])
    try:
        rows = json.loads(out)["rows"]
    except (ValueError, KeyError, TypeError) as exc:
        raise PolicyError("prompt constructor returned malformed output") from exc
    if not isinstance(rows, list) or len(rows) != len(lines):
        raise PolicyError("prompt constructor returned the wrong number of chats")
    return rows


def _model_answer(endpoint: str, chat: list[dict[str, str]]) -> str:
    payload = {
        "model": "Qwen3-8B-Q4_K_M", "messages": chat, "temperature": 0,
        "seed": 42, "max_tokens": 512, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(endpoint + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        # Worst case on the L4 is an uncached 8,192-token prefill plus 512
        # decoded tokens, about 20-30 s; 600 s keeps a wide margin. A timeout
        # stays an execution failure (evaluator_error), never a candidate score.
        response = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as exc:
        # HTTP 400 is the server rejecting the submitted chat itself (for example
        # a rendered prompt longer than the fixed context). The prompt is fully
        # policy-controlled, so this is a failed repair for that input, not an
        # execution failure. Every other status remains an evaluator error.
        if exc.code == 400:
            exc.close()
            return ""
        raise
    with response:
        choice = json.load(response)["choices"][0]
        if choice.get("finish_reason") != "stop":
            return ""
        text = choice["message"]["content"]
    blocks = re.findall(r"```(?:[^\n]*\n)?(.*?)```", text, re.S)
    if len(blocks) != 1:
        return ""
    answer = blocks[0].strip("\r\n")
    return "" if "\n" in answer or "\r" in answer else answer


def _parse_hunter_output(pattern: str, output: str) -> dict:
    """Use the original parser; malformed attack records remain unknown."""
    lines = output.splitlines()
    if (len(lines) < 3 or lines[1] != "regex: " + pattern
            or not lines[2].startswith("mode: java/s;")
            or (len(lines) - 3) % 6):
        return {"pattern": pattern, "result": None, "error": "invalid detector framing"}
    parsed = subprocess.run(
        [sys.executable, str(ASSETS / "parse_redos_hunter.py")],
        input=json.dumps({"pattern": pattern, "result": output, "error": None}) + "\n",
        capture_output=True, text=True, timeout=15,
    )
    if parsed.returncode:
        return {"pattern": pattern, "result": None, "error": "detector parser failed"}
    row = json.loads(parsed.stdout)
    if row.get("pattern") != pattern or not isinstance(row.get("result"), list):
        return {"pattern": pattern, "result": None, "error": row.get("error") or "invalid parser result"}
    return row


def _run_hunter(pattern: str) -> dict:
    if not pattern:
        return {"pattern": None, "result": None, "error": "empty repair"}
    try:
        p = subprocess.run(
            ["java", "-Xmx1024m", "-Dfile.encoding=UTF-8", "-jar", str(ASSETS / "ReDoSHunter-1.0.0.jar"), pattern],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {"pattern": pattern, "result": None, "error": "timeout"}
    if p.returncode or "PARSE ERROR" in p.stderr or "mode: java/s;" not in p.stdout:
        return {"pattern": pattern, "result": None, "error": "detector error"}
    return _parse_hunter_output(pattern, p.stdout)


def _semantic_evidence_valid(row: dict) -> bool:
    texts, results = row.get("texts"), row.get("results")
    return (row.get("error") is None and isinstance(texts, list) and len(texts) == 100
            and isinstance(results, list) and len(results) == len(texts)
            and all(type(value) is bool for value in results))


def _aggregate_scores(row: dict, verdicts: list[dict], semantic_valid: list[bool],
                      directional: list[float] | None = None) -> dict[str, float]:
    """Full-denominator paper metrics, with explicit fail-closed adaptation."""
    n = len(verdicts)
    if not n or len(semantic_valid) != n:
        raise ValueError("incomplete scoring evidence")
    required = ("compilability", "length_ratio", "normalized-edit-distance", "jaccard", "precision", "recall")
    if any(len(row[key]) != n for key in required):
        raise ValueError("scorer output length mismatch")
    valid = [bool(row["compilability"][i] and semantic_valid[i]) for i in range(n)]
    semantic = {key: [float(row[key][i]) if valid[i] else 0.0 for i in range(n)]
                for key in ("jaccard", "precision", "recall")}
    if any(not math.isfinite(x) or not 0 <= x <= 1 for values in semantic.values() for x in values):
        raise ValueError("invalid semantic score")
    eliminated = [bool(v.get("error") is None and isinstance(v.get("result"), list)
                       and not any(a["attack_success"] for a in v["result"])) for v in verdicts]
    repair = [bool(valid[i] and semantic["jaccard"][i] > 0
                   and verdicts[i].get("error") is None
                   and isinstance(verdicts[i].get("result"), list)
                   and not any(a["attack_success"] for a in verdicts[i]["result"])) for i in range(n)]
    # Continuous score term. The artifact's Jaccard estimate is the mean of two
    # directional acceptance rates (original samples accepted by the repair, and
    # repair samples accepted by the original), so any superset of the original
    # language scores at least 0.5. A credited repair instead earns the smaller
    # of the two rates: a safe rewrite that widens or narrows the language
    # earns proportionally less.
    if directional is None or len(directional) != n:
        raise ValueError("directional semantic evidence missing")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in directional):
        raise ValueError("invalid directional semantic score")
    fidelity = [directional[i] if repair[i] else 0.0 for i in range(n)]
    return {
        "fidelity": sum(fidelity) / n,
        "vulnerability_elimination_rate": sum(eliminated) / n,
        "repair_rate": sum(repair) / n,
        "well_formed_rate": sum(bool(x) for x in row["compilability"]) / n,
        "rli": sum(float(x) - 1 for x in row["length_ratio"]) / n,
        "nls": sum(1 - float(x) if row["compilability"][i] else 0
                   for i, x in enumerate(row["normalized-edit-distance"])) / n,
        **{key: sum(values) / n for key, values in semantic.items()},
    }


def _run_script(name: str, args: list[str], *, stdin: str | None = None, stdout: Path | None = None, cwd: Path) -> None:
    out = stdout.open("w", encoding="utf-8") if stdout else subprocess.DEVNULL
    try:
        p = subprocess.run([sys.executable, str(ASSETS / name), *args], input=stdin, text=True, stdout=out, stderr=subprocess.PIPE, cwd=cwd, timeout=900)
    finally:
        if stdout:
            out.close()
    if p.returncode:
        raise RuntimeError(f"upstream scorer {name} failed: {(p.stderr or '')[-500:]}")


def _score_patterns(originals: list[str], repairs: list[str]) -> dict[str, float]:
    n = len(originals)
    with tempfile.TemporaryDirectory(prefix="lrr-score-") as td:
        root = Path(td); dataset = "suite"; method = "candidate"
        source = root / "patterns" / f"{dataset}.txt"
        target = root / "repaired-patterns" / f"{method}-{dataset}.txt"
        hunter = root / "results" / f"redos_hunter-{method}-{dataset}.jsonl"
        source.parent.mkdir(); target.parent.mkdir(); hunter.parent.mkdir()
        source.write_text("".join(x + "\n" for x in originals), encoding="utf-8")
        target.write_text("".join(x + "\n" for x in repairs), encoding="utf-8")
        verdicts = [_run_hunter(x) for x in repairs]
        hunter.write_text("".join(json.dumps(x) + "\n" for x in verdicts), encoding="utf-8")
        ss = root / "source-samples.jsonl"; ts = root / "target-samples.jsonl"
        _run_script("run_xeger.py", [], stdin=source.read_text(), stdout=ss, cwd=root)
        _run_script("run_xeger.py", [], stdin=target.read_text(), stdout=ts, cwd=root)
        on_source = root / "full-matching" / "reference-test-cases" / f"xeger-{method}-{dataset}.jsonl"
        on_target = root / "full-matching" / "reference-patterns" / f"xeger-{method}-{dataset}.jsonl"
        on_source.parent.mkdir(parents=True); on_target.parent.mkdir(parents=True)
        _run_script("get_mathing_results.py", [str(target), str(ss)], stdout=on_source, cwd=root)
        _run_script("get_mathing_results.py", [str(source), str(ts)], stdout=on_target, cwd=root)
        scores = root / "scores.json"
        _run_script("update_scores.py", [method, str(scores), "--data-dir", str(root), "--dataset-name", dataset], cwd=root)
        row = json.loads(scores.read_text())[method]
        left = [json.loads(x) for x in on_source.read_text().splitlines()]
        right = [json.loads(x) for x in on_target.read_text().splitlines()]
        if len(left) != n or len(right) != n:
            raise ValueError("semantic evidence length mismatch")
        semantic_valid = [_semantic_evidence_valid(a) and _semantic_evidence_valid(b)
                          for a, b in zip(left, right, strict=True)]
        # left: the repair applied to the original's samples; right: the
        # original applied to the repair's samples.
        directional = [min(sum(a["results"]) / len(a["results"]), sum(b["results"]) / len(b["results"]))
                       if ok else 0.0 for a, b, ok in zip(left, right, semantic_valid, strict=True)]
    return _aggregate_scores(row, verdicts, semantic_valid, directional)


def _run_policy(repo, workloads: list[dict], task: str = "five_shot_localization") -> dict[str, dict[str, float]]:
    endpoint = _endpoint(); result = {}
    for workload in workloads:
        originals: list[str] = []; repairs: list[str] = []
        lines = [json.dumps(_policy_input(item["input"])) if task.endswith("localization") else item["input"]["pattern"]
                 for item in workload["items"]]
        rows = _render_chats(repo, task, lines)
        for item, chat_and_pattern in zip(workload["items"], rows, strict=True):
            if chat_and_pattern is None:
                raise PolicyError("constructor rejected frozen input")
            chat, pattern = chat_and_pattern
            if pattern != item["input"]["pattern"]:
                raise PolicyError("constructor changed the original pattern")
            try:
                answer = _model_answer(endpoint, chat)
            except Exception as exc:
                raise RuntimeError(
                    f"fixed model request failed for workload {workload['id']} "
                    f"item {item['id']}"
                ) from exc
            originals.append(pattern); repairs.append(answer)
            _runtime_state.setdefault("observations", []).append({
                "workload": workload["id"], "id": item["id"], "method": task,
                "pattern": pattern, "answer": answer,
            })
        result[workload["id"]] = _score_patterns(originals, repairs)
    return result


def _reference(workloads: list[dict]) -> dict[str, dict[str, float]]:
    role = "final" if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else "development"
    if role not in _reference_cache:
        applied = apply_submission_patch(str(TASK_DIR / "reference.patch"), TASK_DIR, CONFIG)
        if not applied.ok:
            raise RuntimeError("private reference patch is invalid")
        try:
            _reference_cache[role] = _run_policy(Path(applied.workdir), workloads)
        except PolicyError as exc:
            raise RuntimeError("private reference policy failed") from exc
        finally:
            shutil.rmtree(Path(applied.workdir).parent, ignore_errors=True)
    return _reference_cache[role]


def _clears(name: str, candidate: float, reference: float) -> bool:
    if not math.isfinite(candidate): return False
    if name == "rli": return candidate < reference
    return candidate > reference or (candidate == reference == 1.0)


def prepare() -> dict:
    """Warm the paired reference and measure both required scientific controls."""
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    workloads = json.loads(DATA.read_text(encoding="utf-8"))["final" if final else "development"]
    reference = _reference(workloads)
    applied = apply_submission_patch(str(TASK_DIR / "reference.patch"), TASK_DIR, CONFIG)
    if not applied.ok:
        raise RuntimeError("private reference patch is invalid")
    try:
        ordinary = _run_policy(Path(applied.workdir), workloads, "five_shot")
    finally:
        shutil.rmtree(Path(applied.workdir).parent, ignore_errors=True)
    no_repair = {}
    for workload in workloads:
        originals = [item["input"]["pattern"] for item in workload["items"]]
        no_repair[workload["id"]] = _score_patterns(originals, originals)
    return {
        "status": "ready", "workloads": len(workloads),
        "reference_repair_rate": {k: v["repair_rate"] for k, v in reference.items()},
        "ordinary_five_shot_repair_rate": {k: v["repair_rate"] for k, v in ordinary.items()},
        "no_repair_rate": {k: v["repair_rate"] for k, v in no_repair.items()},
    }


def evaluate(solution_path: str):
    started = time.monotonic()
    if not Path("/app/scripts/inference.py").is_file():
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "entrypoint_probe.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=True,
        )
    final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
    data = json.loads(DATA.read_text(encoding="utf-8"))
    workloads = data["final"] if final else data["development"]
    applied = apply_submission_patch(solution_path, TASK_DIR, CONFIG)
    if not applied.ok:
        return _fail(applied.message)
    try:
        candidate = _run_policy(Path(applied.workdir), workloads)
    except PolicyError as exc:
        return _fail(str(exc))
    finally:
        shutil.rmtree(Path(applied.workdir).parent, ignore_errors=True)
    reference = _reference(workloads)
    rows = []
    for workload in workloads:
        wid = workload["id"]
        rows.append({"workload": wid, "metrics": {name: bool(_clears(name, candidate[wid][name], reference[wid][name])) for name in METRIC_NAMES}})
    beats = int(all(v for row in rows for v in row["metrics"].values()))
    counts = {name: sum(len(w["items"]) for w in workloads) for name in METRIC_NAMES}
    weighted = {name: sum(candidate[w["id"]][name] * len(w["items"]) for w in workloads) / counts[name] for name in METRIC_NAMES}
    total = sum(len(w["items"]) for w in workloads)
    fidelity = sum(candidate[w["id"]]["fidelity"] * len(w["items"]) for w in workloads) / total
    ref_fidelity = sum(reference[w["id"]]["fidelity"] * len(w["items"]) for w in workloads) / total
    score = 100.0 * fidelity
    metrics = {
        "runtime_backend": _runtime_state.get("gpu_backend", {}),
        "runtime_seconds": time.monotonic() - started,
        "beats_reference": beats, "beat_spec_workload_passes": rows,
        "reference_score": 100.0 * ref_fidelity, "margin": score - 100.0 * ref_fidelity,
        "train_vulnerability_elimination_rate": weighted["vulnerability_elimination_rate"],
        "train_repair_rate": weighted["repair_rate"], "train_well_formed_rate": weighted["well_formed_rate"],
        "train_mean_rli": weighted["rli"], "train_mean_nls": weighted["nls"],
        "train_mean_jaccard": weighted["jaccard"], "train_mean_precision": weighted["precision"],
        "train_mean_recall": weighted["recall"],
    }
    return score, score, "evaluated through constructor, fixed model, detector, and semantic scorers", metrics


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(TASK_DIR / "reference.patch")
    print(json.dumps(evaluate(path), indent=2))
