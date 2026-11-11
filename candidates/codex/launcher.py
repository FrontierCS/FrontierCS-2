#!/usr/bin/env python3
"""Thin launcher for the generic Frontier Codex Candidate overlay."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from urllib.parse import urlsplit


TASK_README = Path("/frontier/task/README.md")
WORKSPACE = Path("/workspace")
CODEX = Path("/opt/frontier/candidate/codex/bin/codex")
CODEX_TOOLS = Path("/opt/frontier/candidate/codex/codex-path")
SKILLS = Path("/opt/frontier/candidate/skills")
SUBMIT_BIN = Path("/opt/frontier/candidate/bin")
FINAL_RECEIPT = WORKSPACE / ".frontier" / "final-submission.json"
# Byte-exact instructions that pinned Codex 0.162.1 sends for a model slug that
# is absent from its bundled catalog. Only read when a Candidate opts into an
# explicit context window (see model_catalog below).
FALLBACK_INSTRUCTIONS = Path("/opt/frontier/candidate/codex-fallback-base-instructions.md")
FALLBACK_INSTRUCTIONS_SHA256 = "3b08633fa672906666659d764864dfda1d7af5b5111ea5817c8f46e5de4e1a8d"
MODEL_CATALOG_NAME = "frontier-model-catalog.json"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
QWEN_MODEL = "qwen3.8-max-vl-thinking"
QWEN_REASONING_EFFORTS = ("none", "high")
CONTEXT_WINDOW_RANGE = (16_384, 2**31 - 1)
RUNTIME_DIRS = {
    "CODEX_HOME": WORKSPACE / ".frontier-codex-home",
    "HOME": WORKSPACE / ".frontier-home",
    "TMPDIR": WORKSPACE / ".frontier-tmp",
}

PROMPT = """\
Solve the evaluation task in /frontier/task/README.md.

Before doing any task work, read that file completely and read the installed
`frontier-submit` skill at
`/opt/frontier/candidate/skills/frontier-submit/SKILL.md`. The task README is authoritative for the exact train
and final tar layouts. Work only in /workspace. Use the `frontier-submit`
command described by the skill to submit a tar to verifier-train, inspect its
score and remaining call count, and iterate while useful. Do not call the
Driver API, inspect its token, or write a replacement submission client.

If the task explicitly supports Agent GPU execution, read the installed
`frontier-gpu` skill and use its create/push/exec/pull/delete workflow within
the task's GPU budget, unless the README explicitly requires a legacy task-local
GPU client; in that case use that client for task operations. Do not mix clients.
Only the selected tool may manage the Driver-owned runner.
Use `frontier-gpu quota` to check remaining SSH sessions before debugging.
The command reports whether verifier traffic has a separate quota or shares a legacy one.

When the answer is ready, create the final tar in the task-defined format, run
`frontier-submit final` exactly once as your last platform action, then exit.
Web search is disabled; use only local task files and tools.
"""

class LaunchError(RuntimeError):
    pass


def _regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise LaunchError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(mode):
        raise LaunchError(f"{label} must be a regular file: {path}")


def _endpoint(name: str, expected_port: int, expected_path: str) -> str:
    value = os.environ.get(name, "").strip()
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LaunchError(f"{name} must be one plain HTTP run-local endpoint")
    if parsed.port != expected_port or parsed.path.rstrip("/") != expected_path:
        raise LaunchError(f"{name} must use port {expected_port} and path {expected_path or '/'}")
    try:
        ipaddress.ip_address(parsed.hostname or "")
    except ValueError as error:
        raise LaunchError(f"{name} host must be an IP address") from error
    return value.rstrip("/")


def _reset_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    path.mkdir(mode=0o700)


def reset_codex_home(child: dict[str, str]) -> None:
    codex_home = RUNTIME_DIRS["CODEX_HOME"]
    _reset_directory(codex_home)
    skills = codex_home / "skills"
    skills.mkdir(mode=0o700)
    (skills / "frontier-submit").symlink_to(
        SKILLS / "frontier-submit", target_is_directory=True
    )
    (skills / "frontier-gpu").symlink_to(
        SKILLS / "frontier-gpu", target_is_directory=True
    )
    child["CODEX_HOME"] = str(codex_home)


def direct_model_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment
                or any(character.isspace() for character in value)):
            raise ValueError("invalid endpoint")
    except ValueError:
        raise LaunchError("FRONTIER_MODEL_DIRECT_BASE_URL must be an HTTPS endpoint without credentials, query or fragment") from None
    return value.rstrip("/")


def prepare_environment() -> dict[str, str]:
    for path, label in ((CODEX, "Codex"), (TASK_README, "task README"),
                        (SKILLS / "frontier-submit" / "SKILL.md", "submission skill"),
                        (SKILLS / "frontier-gpu" / "SKILL.md", "GPU skill"),
                        (SUBMIT_BIN / "frontier-gpu", "GPU tool"),
                        (SUBMIT_BIN / "frontier-gpu-connect-proxy", "GPU CONNECT proxy"),
                        (SUBMIT_BIN / "frontier-submit", "submission tool")):
        _regular(path, label)
    if not WORKSPACE.is_dir():
        raise LaunchError("/workspace is unavailable")

    driver_url = _endpoint("FRONTIER_DRIVER_API_URL", 8080, "")
    direct = os.environ.get("FRONTIER_MODEL_DIRECT_BASE_URL", "").strip()
    if direct:
        direct = direct_model_url(direct)
    else:
        model_url = _endpoint("FRONTIER_MODEL_API_BASE_URL", 8082, "/v1")
        if urlsplit(driver_url).hostname != urlsplit(model_url).hostname:
            raise LaunchError("Driver and model proxy must share the run-local IP")
    for name in (("CODEX_API_KEY",) if direct else ("CODEX_API_KEY", "FRONTIER_MODEL_PROXY_TOKEN")):
        if not os.environ.get(name, "").strip():
            raise LaunchError(f"{name} is missing")

    child = os.environ.copy()
    # CODEX_API_KEY is a legacy alias for the run-local Driver token. Codex
    # treats this well-known variable as provider authority even when the
    # custom provider disables OpenAI auth, so never pass it to the child.
    if direct:
        # Explicit direct mode keeps its operator-supplied provider key.
        # Production's default proxy mode below still strips the legacy alias.
        child["FRONTIER_MODEL_DIRECT_BASE_URL"] = direct
        child.pop("FRONTIER_MODEL_PROXY_TOKEN", None)
    else:
        child.pop("CODEX_API_KEY", None)
    for name, path in RUNTIME_DIRS.items():
        if name == "CODEX_HOME":
            continue
        _reset_directory(path)
        child[name] = str(path)
    reset_codex_home(child)
    child["PATH"] = f"{SUBMIT_BIN}:{CODEX_TOOLS}:{child.get('PATH', '/usr/bin:/bin')}"
    child["FRONTIER_DRIVER_API_URL"] = driver_url
    return child


def context_window() -> int | None:
    value = os.environ.get("FRONTIER_CODEX_CONTEXT_WINDOW", "").strip()
    if not value:
        return None
    low, high = CONTEXT_WINDOW_RANGE
    if not (value.isascii() and value.isdigit() and len(value) <= 10 and low <= int(value) <= high):
        raise LaunchError("FRONTIER_CODEX_CONTEXT_WINDOW is invalid")
    return int(value)


def reasoning_effort(model: str) -> str:
    effort = os.environ.get("FRONTIER_CODEX_REASONING_EFFORT", "").strip()
    if model == QWEN_MODEL:
        effort = effort or "high"
        if effort not in QWEN_REASONING_EFFORTS:
            raise LaunchError("Qwen reasoning effort must be none or high")
    elif effort and effort not in REASONING_EFFORTS:
        raise LaunchError("FRONTIER_CODEX_REASONING_EFFORT is invalid")
    return effort


def model_catalog(codex_home: Path) -> Path | None:
    """Write a one-row native model catalog when a context window is explicit.

    Pinned Codex gives an unknown model slug fallback metadata with a 272,000
    token window; `model_context_window` and `model_auto_compact_token_limit`
    cannot raise it. A catalog row is the native way to declare another
    window. Every other field reproduces the fallback request (tools and
    instructions), except that reasoning summaries are not requested, so the
    request carries `reasoning: {"effort": ...}` only.
    """
    window = context_window()
    if window is None:
        return None
    model = os.environ.get("FRONTIER_CODEX_MODEL", "").strip()
    if not model:
        raise LaunchError("FRONTIER_CODEX_MODEL is missing")
    reasoning_effort(model)
    _regular(FALLBACK_INSTRUCTIONS, "Codex fallback instructions")
    raw = FALLBACK_INSTRUCTIONS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FALLBACK_INSTRUCTIONS_SHA256:
        raise LaunchError("Codex fallback instructions do not match the pinned text")
    row = {
        "slug": model, "display_name": model, "description": "",
        "default_reasoning_level": "high" if model == QWEN_MODEL else "medium",
        "supported_reasoning_levels": [
            {"effort": effort, "description": effort}
            for effort in (QWEN_REASONING_EFFORTS if model == QWEN_MODEL
                           else REASONING_EFFORTS[2:])
        ],
        "shell_type": "unified_exec", "visibility": "none", "supported_in_api": True,
        "priority": 99, "additional_speed_tiers": [], "service_tiers": [],
        "availability_nux": None, "upgrade": None,
        "base_instructions": raw.decode("utf-8"),
        "supports_reasoning_summaries": True, "default_reasoning_summary": "none",
        "support_verbosity": False, "default_verbosity": None,
        "apply_patch_tool_type": None, "web_search_tool_type": "text",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_parallel_tool_calls": True,
        "context_window": window, "max_context_window": window,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [], "input_modalities": ["text", "image"],
        "supports_search_tool": False,
    }
    path = codex_home / MODEL_CATALOG_NAME
    path.write_text(json.dumps({"models": [row]}), encoding="utf-8")
    path.chmod(0o600)
    return path


def command(model_url: str, *, direct: bool = False, catalog: Path | None = None) -> list[str]:
    model = os.environ.get("FRONTIER_CODEX_MODEL", "").strip()
    if not model:
        raise LaunchError("FRONTIER_CODEX_MODEL is missing")
    effort = reasoning_effort(model)
    return [
        str(CODEX), "--model", model, "--config", "approval_policy=\"never\"",
        "--config", "web_search=\"disabled\"",
        "--config", "allow_login_shell=false",
        "--config", "model_verbosity=\"medium\"",
        *(["--config", "model_reasoning_effort=" + json.dumps(effort)] if effort else []),
        *(["--config", "model_catalog_json=" + json.dumps(str(catalog))] if catalog else []),
        "--config", "model_provider=\"frontier\"",
        "--config", 'model_providers.frontier.name=' + json.dumps("Frontier Direct" if direct else "Frontier LiteLLM"),
        "--config", "model_providers.frontier.base_url=" + json.dumps(model_url),
        "--config", "model_providers.frontier.wire_api=\"responses\"",
        "--config", "model_providers.frontier.requires_openai_auth=false",
        # Keep a high native retry allowance inside the pinned official CLI so
        # transient provider failures preserve the conversation and Run clock.
        "--config", "model_providers.frontier.request_max_retries=100",
        "--config", "model_providers.frontier.stream_max_retries=100",
        "--config", ('model_providers.frontier.env_key="CODEX_API_KEY"' if direct else
                     'model_providers.frontier.env_http_headers.X-Frontier-Agent-Token="FRONTIER_MODEL_PROXY_TOKEN"'),
        "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
        "--skip-git-repo-check", "--sandbox", "danger-full-access",
        "--cd", str(WORKSPACE), "-",
    ]


def validate_final_receipt() -> None:
    _regular(FINAL_RECEIPT, "final submission receipt")
    value = json.loads(FINAL_RECEIPT.read_text(encoding="utf-8"))
    if value.get("schema_version") != "frontier.candidate-submit-result/v1" or value.get("operation") != "final" or value.get("state") != "accepted":
        raise LaunchError("final submission receipt is invalid")


def cleanup() -> None:
    for path in RUNTIME_DIRS.values():
        if path.exists() or path.is_symlink():
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()


def main() -> int:
    try:
        child = prepare_environment()
        direct = child.get("FRONTIER_MODEL_DIRECT_BASE_URL", "")
        model_url = direct or child["FRONTIER_MODEL_API_BASE_URL"]
        catalog = model_catalog(RUNTIME_DIRS["CODEX_HOME"])
        completed = subprocess.run(
            command(model_url, direct=bool(direct), catalog=catalog),
            input=PROMPT, text=True, env=child, check=False
        )
        if completed.returncode != 0:
            raise LaunchError(f"Codex exited with status {completed.returncode}")
        validate_final_receipt()
        return 0
    except (LaunchError, OSError, json.JSONDecodeError) as error:
        print(f"frontier-codex-launch: ERROR: {error}", file=sys.stderr)
        return 2
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
