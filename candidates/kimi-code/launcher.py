#!/usr/bin/env python3
"""Thin launcher for the generic Frontier Kimi Code Candidate overlay."""

from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from urllib.parse import urlsplit


TASK_README = Path("/frontier/task/README.md")
WORKSPACE = Path("/workspace")
NODE = Path("/opt/frontier/candidate/kimi-code/node")
KIMI_MAIN = Path("/opt/frontier/candidate/kimi-code/app/dist/main.mjs")
KIMI_LIB = Path("/opt/frontier/candidate/kimi-code/lib")
KIMI_LOADER = KIMI_LIB / "ld-linux-x86-64.so.2"
SKILLS = Path("/opt/frontier/candidate/skills")
SUBMIT_BIN = Path("/opt/frontier/candidate/bin")
FINAL_RECEIPT = WORKSPACE / ".frontier/final-submission.json"
TERMINATION_LOG = Path("/dev/termination-log")
DIAGNOSTIC_TAIL_BYTES = 64 * 1024
DIAGNOSTIC_MAX_BYTES = 3072
DIAGNOSTIC_CLASSES = frozenset({
    "Error", "TypeError", "SyntaxError", "RangeError", "ReferenceError",
    "URIError", "EvalError", "AbortError", "KimiError", "ChatProviderError",
    "APIConnectionError", "APITimeoutError", "APIStatusError",
    "APIContextOverflowError", "APIRequestTooLargeError",
    "APIProviderRateLimitError", "APIProviderQuotaExhaustedError",
    "APIEmptyResponseError",
})
DIAGNOSTIC_CODES = frozenset({
    "context.overflow", "loop.max_steps_exceeded", "provider.api_error",
    "provider.filtered", "provider.rate_limit", "provider.auth_error",
    "provider.connection_error", "records.write_failed", "compaction.failed",
    "compaction.unable", "session.closed", "internal",
})
RUNTIME_DIRS = {
    "KIMI_CODE_HOME": WORKSPACE / ".frontier-kimi-code-home",
    "HOME": WORKSPACE / ".frontier-home",
    "TMPDIR": WORKSPACE / ".frontier-tmp",
}
MODEL_ALIAS = re.compile(r"^[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?$")

PROMPT = """\
Solve the evaluation task in /frontier/task/README.md.

Before doing any task work, read that file completely and use the installed
`frontier-submit` skill. The task README is authoritative for the exact train
and final tar layouts. Work only in /workspace. Use `frontier-submit train` to
obtain task-approved feedback and the authoritative remaining call count, and
iterate only while useful. After one representative local check, submit train
promptly instead of reconstructing the private Verifier workload. Do not call
the Driver API, inspect its token, or write a replacement submission client.

If the task explicitly supports Agent GPU execution, read the installed
`frontier-gpu` skill and use its create/push/exec/pull/delete workflow within
the task's GPU budget, unless the README explicitly requires a legacy task-local
GPU client; in that case use that client for task operations. Do not mix clients.
Only the selected tool may manage the Driver-owned runner.
Use `frontier-gpu quota` to check remaining SSH sessions before debugging.
The command reports whether verifier traffic has a separate quota or shares a legacy one.

When the answer is ready, create the final tar in the task-defined format, run
`frontier-submit final` exactly once as your last platform action. Immediately
after it reports acceptance, do not call TodoList, Bash, or any other tool. Your
next and only response must be a brief completion message with no tool call so
the process exits successfully.
Web search and URL fetching are disabled; use only local task files and tools.
"""


class LaunchError(RuntimeError):
    pass


def _regular(path: Path, label: str, *, executable: bool = False) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise LaunchError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(mode) or (executable and mode & 0o111 == 0):
        raise LaunchError(f"{label} must be a regular executable file: {path}")


def _endpoint(name: str, expected_port: int, expected_path: str) -> str:
    value = os.environ.get(name, "").strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port != expected_port
        or parsed.path.rstrip("/") != expected_path
    ):
        raise LaunchError(f"{name} must be one plain run-local HTTP endpoint")
    try:
        ipaddress.ip_address(parsed.hostname or "")
    except ValueError as error:
        raise LaunchError(f"{name} host must be an IP address") from error
    return value.rstrip("/")


def _reset_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    path.mkdir(mode=0o700)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def context_tokens() -> int:
    value = os.environ.get("FRONTIER_KIMI_MAX_CONTEXT_TOKENS", "262144")
    if not re.fullmatch(r"[1-9][0-9]{0,9}", value) or int(value) > 2_147_483_647:
        raise LaunchError("FRONTIER_KIMI_MAX_CONTEXT_TOKENS must be a positive 32-bit integer")
    return int(value)


def thinking_efforts() -> tuple[list[str], str] | None:
    supported = os.environ.get("FRONTIER_KIMI_SUPPORT_EFFORTS")
    default = os.environ.get("FRONTIER_KIMI_DEFAULT_EFFORT")
    if supported is None and default is None:
        return None
    message = ("FRONTIER_KIMI_SUPPORT_EFFORTS must be a nonempty JSON array of unique "
               "minimal/low/medium/high/xhigh/max values and FRONTIER_KIMI_DEFAULT_EFFORT "
               "must be one of those values")
    if supported is None or default is None or len(supported) > 256:
        raise LaunchError(message)
    try:
        values = json.loads(supported)
    except ValueError as error:
        raise LaunchError(message) from error
    allowed = {"minimal", "low", "medium", "high", "xhigh", "max"}
    if (not isinstance(values, list) or not values
            or any(not isinstance(value, str) or value not in allowed for value in values)
            or len(set(values)) != len(values) or default not in values):
        raise LaunchError(message)
    return values, default


def render_config(model_url: str, model_alias: str, proxy_token: str,
                  *, max_context_tokens: int = 262144,
                  efforts: tuple[list[str], str] | None = None) -> str:
    return "\n".join(
        [
            'default_model = "frontier"',
            'default_permission_mode = "auto"',
            "builtin_product_skills = false",
            "telemetry = false",
            "",
            "[providers.frontier]",
            'type = "openai"',
            f"base_url = {_toml_string(model_url)}",
            f"api_key = {_toml_string(proxy_token)}",
            "",
            "[models.frontier]",
            'provider = "frontier"',
            f"model = {_toml_string(model_alias)}",
            f"max_context_size = {max_context_tokens}",
            'capabilities = ["thinking", "tool_use"]',
            *([] if efforts is None else [
                "support_efforts = " + json.dumps(efforts[0]),
                "default_effort = " + _toml_string(efforts[1]),
            ]),
            "",
            "[loop_control]",
            "max_steps_per_turn = 0",
            "",
            "[background]",
            "keep_alive_on_exit = false",
            "",
            "[tools]",
            'disabled = ["WebSearch", "FetchURL"]',
            "",
        ]
    )


def prepare_environment() -> tuple[dict[str, str], str]:
    for path, label, executable in (
        (NODE, "Node runtime", True),
        (KIMI_LOADER, "Node runtime loader", True),
        (KIMI_MAIN, "Kimi Code CLI", False),
        (TASK_README, "task README", False),
        (SKILLS / "frontier-submit" / "SKILL.md", "submission skill", False),
        (SKILLS / "frontier-gpu" / "SKILL.md", "GPU skill", False),
        (SUBMIT_BIN / "frontier-gpu", "GPU tool", True),
        (SUBMIT_BIN / "frontier-gpu-connect-proxy", "GPU CONNECT proxy", True),
        (SUBMIT_BIN / "frontier-submit", "submission tool", True),
    ):
        _regular(path, label, executable=executable)
    if not WORKSPACE.is_dir() or not KIMI_LIB.is_dir():
        raise LaunchError("workspace or Kimi Code runtime library is unavailable")

    driver_url = _endpoint("FRONTIER_DRIVER_API_URL", 8080, "")
    model_url = _endpoint("FRONTIER_MODEL_API_BASE_URL", 8082, "/v1")
    if urlsplit(driver_url).hostname != urlsplit(model_url).hostname:
        raise LaunchError("Driver and model proxy must share the run-local IP")
    model_alias = os.environ.get("FRONTIER_CODEX_MODEL", "").strip()
    if not MODEL_ALIAS.fullmatch(model_alias):
        raise LaunchError("FRONTIER_CODEX_MODEL is missing or invalid")
    proxy_token = os.environ.get("FRONTIER_MODEL_PROXY_TOKEN", "").strip()
    if not proxy_token or len(proxy_token) > 16 * 1024:
        raise LaunchError("FRONTIER_MODEL_PROXY_TOKEN is missing or invalid")

    max_context_tokens = context_tokens()
    efforts = thinking_efforts()
    child = os.environ.copy()
    # Native model overlays can override the immutable alias, context and sampling.
    for name in tuple(child):
        if name.startswith("KIMI_MODEL_"):
            child.pop(name)
    for name in (
        "CODEX_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "KIMI_API_KEY",
        "KIMI_BASE_URL",
        "KIMI_LOOP_MAX_STEPS_PER_TURN",
        "LD_LIBRARY_PATH",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ):
        child.pop(name, None)
    for name, path in RUNTIME_DIRS.items():
        _reset_directory(path)
        child[name] = str(path)
    config_path = RUNTIME_DIRS["KIMI_CODE_HOME"] / "config.toml"
    descriptor = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        destination.write(render_config(model_url, model_alias, proxy_token,
                                        max_context_tokens=max_context_tokens,
                                        efforts=efforts))
        destination.flush()
        os.fsync(destination.fileno())
    child["PATH"] = f"{SUBMIT_BIN}:{child.get('PATH', '/usr/bin:/bin')}"
    child["FRONTIER_DRIVER_API_URL"] = driver_url
    return child, model_alias


def command(model_alias: str) -> list[str]:
    del model_alias
    return [
        str(KIMI_LOADER),
        "--library-path",
        str(KIMI_LIB),
        str(NODE),
        str(KIMI_MAIN),
        "--model",
        "frontier",
        "--prompt",
        PROMPT,
        "--output-format",
        "text",
        "--skills-dir",
        str(SKILLS),
    ]


def _accepted_final_receipt() -> bool:
    try:
        metadata = FINAL_RECEIPT.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
            or not 0 < metadata.st_size <= 16 * 1024
        ):
            return False
        document = json.loads(FINAL_RECEIPT.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(document, dict)
        and set(document)
        == {"schema_version", "operation", "state", "payload_id", "verification_id"}
        and document.get("schema_version") == "frontier.candidate-submit-result/v1"
        and document.get("operation") == "final"
        and document.get("state") == "accepted"
        and isinstance(document.get("payload_id"), str)
        and document["payload_id"].startswith("payload_")
        and isinstance(document.get("verification_id"), str)
        and document["verification_id"].startswith("verification_")
    )


def _diagnostic_from_tail(tail: bytes, truncated: bool, returncode: int) -> dict:
    diagnostic = {
        "schema_version": "frontier.kimi-code-failure/v1",
        "child_returncode": returncode,
        "capture": "no_terminal_record",
        "tail_truncated": truncated,
        "exception_class": "other_or_absent",
        "error_code": "other_or_absent",
        "frames": [],
    }
    # Only the native top-level prompt failure record is eligible. Neither
    # arbitrary error text nor the logger's generic redaction is an export rule.
    lines = tail.decode("utf-8", errors="replace").splitlines()
    header = re.compile(
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z '
        r'ERROR startup failed  operation="run prompt"$'
    )
    positions = [index for index, line in enumerate(lines) if header.fullmatch(line)]
    if not positions:
        return diagnostic
    diagnostic["capture"] = "terminal_record"
    record = []
    for line in lines[positions[-1] + 1:]:
        if not line.startswith("  "):
            break
        record.append(line)
    if record:
        error = re.match(r"^  ([A-Za-z]+): (?:([a-z_.]+): )?", record[0])
        if error:
            if error[1] in DIAGNOSTIC_CLASSES:
                diagnostic["exception_class"] = error[1]
            if error[2] in DIAGNOSTIC_CODES:
                diagnostic["error_code"] = error[2]
    frame = re.compile(
        r"^    at (?:[^\r\n()]{1,200} \()?"
        r"(?:file://)?" + re.escape(str(KIMI_MAIN))
        + r":([1-9][0-9]{0,5}):([1-9][0-9]{0,5})\)?$"
    )
    for line in record[1:]:
        match = frame.fullmatch(line)
        if match:
            diagnostic["frames"].append({"line": int(match[1]), "column": int(match[2])})
            if len(diagnostic["frames"]) == 6:
                break
    return diagnostic


def _failure_diagnostic(returncode: int) -> dict:
    diagnostic = _diagnostic_from_tail(b"", False, returncode)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors = []
    try:
        home = os.open(RUNTIME_DIRS["KIMI_CODE_HOME"], directory_flags)
        descriptors.append(home)
        logs = os.open("logs", directory_flags, dir_fd=home)
        descriptors.append(logs)
        source = os.open("kimi-code.log", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=logs)
        descriptors.append(source)
        metadata = os.fstat(source)
        if not stat.S_ISREG(metadata.st_mode):
            diagnostic["capture"] = "read_failed"
            return diagnostic
        offset = max(0, metadata.st_size - DIAGNOSTIC_TAIL_BYTES)
        os.lseek(source, offset, os.SEEK_SET)
        tail = os.read(source, DIAGNOSTIC_TAIL_BYTES)
        if offset:
            tail = tail.partition(b"\n")[2]
        return _diagnostic_from_tail(tail, bool(offset), returncode)
    except FileNotFoundError:
        diagnostic["capture"] = "log_missing"
        return diagnostic
    except OSError:
        diagnostic["capture"] = "read_failed"
        return diagnostic
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _retain_failure_diagnostic(returncode: int) -> None:
    # The kubelet owns this fixed file; missing/unwritable storage must not
    # change the native process outcome or start a second platform operation.
    try:
        data = json.dumps(_failure_diagnostic(returncode), separators=(",", ":")).encode("ascii")
        if len(data) > DIAGNOSTIC_MAX_BYTES:
            return
        descriptor = os.open(TERMINATION_LOG, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.ftruncate(descriptor, 0)
                os.write(descriptor, data)
        finally:
            os.close(descriptor)
    except Exception:
        pass


def main() -> int:
    try:
        child, model_alias = prepare_environment()
        os.chdir(WORKSPACE)
        completed = subprocess.run(command(model_alias), env=child, check=False)
    except (LaunchError, OSError) as error:
        print(f"frontier-kimi-code-launch: ERROR: {error}", file=sys.stderr)
        return 2
    if completed.returncode == 0 or _accepted_final_receipt():
        return 0
    _retain_failure_diagnostic(completed.returncode)
    return completed.returncode if 0 < completed.returncode < 256 else 2


if __name__ == "__main__":
    raise SystemExit(main())
