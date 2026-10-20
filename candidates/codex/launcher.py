#!/usr/bin/env python3
"""Thin launcher for the generic Frontier Codex Candidate overlay."""

from __future__ import annotations

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


def command(model_url: str, *, direct: bool = False) -> list[str]:
    model = os.environ.get("FRONTIER_CODEX_MODEL", "").strip()
    if not model:
        raise LaunchError("FRONTIER_CODEX_MODEL is missing")
    effort = os.environ.get("FRONTIER_CODEX_REASONING_EFFORT", "").strip()
    if effort and effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        raise LaunchError("FRONTIER_CODEX_REASONING_EFFORT is invalid")
    return [
        str(CODEX), "--model", model, "--config", "approval_policy=\"never\"",
        "--config", "web_search=\"disabled\"",
        "--config", "allow_login_shell=false",
        "--config", "model_verbosity=\"medium\"",
        *(["--config", "model_reasoning_effort=" + json.dumps(effort)] if effort else []),
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
        completed = subprocess.run(
            command(model_url, direct=bool(direct)), input=PROMPT, text=True, env=child, check=False
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
