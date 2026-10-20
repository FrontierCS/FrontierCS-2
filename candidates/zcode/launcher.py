#!/usr/bin/env python3
"""Launch the official ZCode CLI against the run-local model proxy."""

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

WORKSPACE = Path("/workspace")
RUNTIME = Path("/opt/frontier/candidate/zcode")
SKILLS = Path("/opt/frontier/candidate/skills")
SUBMIT_BIN = Path("/opt/frontier/candidate/bin")
MODEL_ALIAS = re.compile(r"^[a-z](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
PROMPT = """Solve the evaluation task in /frontier/task/README.md.
Read the complete task README and /opt/frontier/candidate/skills/frontier-submit/SKILL.md
before working. The README defines the required train and final archive layouts.
Work in /workspace. Use frontier-submit train for asynchronous task-approved
feedback: it returns an accepted verification ID immediately, and frontier-submit
status reads progress once. You can continue coding while verification runs.
Use store-final to save your selected usable answer, updating it when you choose
a better answer. Saved answers are not final submissions and storage does not
verify correctness. Follow the task's scientific budgets and instructions.
If the task supports Agent GPU execution, read the installed frontier-gpu skill
and use its Driver-owned workflow, unless the README requires a task-local GPU
client. Never mix these clients. Use quota to inspect remaining SSH sessions.
Never call the Driver API directly, read its token, or replace the submission client.
When ready, submit the selected final archive with frontier-submit final exactly
once. After acceptance, do not call any tool; return a brief completion message.
Web search and URL fetching are disabled; use local task files and tools.
"""


class LaunchError(RuntimeError):
    pass


def endpoint(name: str, port: int, path: str) -> str:
    value = os.environ.get(name, "").strip()
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.username or parsed.password or parsed.query
            or parsed.fragment or parsed.port != port or parsed.path.rstrip("/") != path):
        raise LaunchError(f"{name} must be a run-local HTTP endpoint")
    try:
        ipaddress.ip_address(parsed.hostname or "")
    except ValueError:
        raise LaunchError(f"{name} host must be an IP address") from None
    return value.rstrip("/")


def provider_config(base_url: str, model: str, token: str, effort: str) -> dict:
    return {"schemaVersion": 1, "config": {
        "providerConfigRules": {"providerRules": [{
            "providerId": "frontier", "providerName": "Frontier LiteLLM",
            "config": {"group": "standard-personal",
                       "access": {"type": "api-key", "apiKey": token},
                       "api": {"type": "openai-chat-completions", "baseUrl": base_url},
                       "personalModelIds": [model], "modelOrder": [model]}}]},
        "modelConfigRules": {"providerModelRules": [{
            "providerId": "frontier", "modelId": model,
            "config": {"enabled": True,
                       "properties": {"contextWindow": 1048576, "supportsToolCall": True,
                                      "supportsJsonSchemaOutput": True},
                       "optionSpecs": {
                           "reasoningLevel": {"values": ["low", "high", "max"],
                                              "map": '{"reasoning_effort": reasoningLevel}'},
                           "maxOutputTokens": {"max": 262144,
                                               "map": '{"max_tokens": maxOutputTokens}'}}}}],
                             "manualProviderModelRules": []},
        "defaultModelSelection": {"providerId": "frontier", "modelId": model,
                                  "options": {"reasoningLevel": effort}}}}


def write_json(path: Path, value: dict) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream)
        stream.write("\n")


def prepare_environment() -> dict[str, str]:
    for path in (RUNTIME / "node", RUNTIME / "lib/ld-linux-x86-64.so.2",
                 RUNTIME / "app/zcode.cjs", RUNTIME / "fetch-sanitize.cjs",
                 RUNTIME / "app/provider/zcode-builtin.json",
                 Path("/frontier/task/README.md"), SKILLS / "frontier-submit/SKILL.md",
                 SKILLS / "frontier-gpu/SKILL.md", SUBMIT_BIN / "frontier-submit",
                 SUBMIT_BIN / "frontier-gpu", SUBMIT_BIN / "frontier-gpu-connect-proxy"):
        if not stat.S_ISREG(path.lstat().st_mode):
            raise LaunchError(f"required regular file is unavailable: {path}")
    if not WORKSPACE.is_dir():
        raise LaunchError("workspace is unavailable")
    driver = endpoint("FRONTIER_DRIVER_API_URL", 8080, "")
    model_url = endpoint("FRONTIER_MODEL_API_BASE_URL", 8082, "/v1")
    if urlsplit(driver).hostname != urlsplit(model_url).hostname:
        raise LaunchError("model proxy and Driver must share the run-local IP")
    model = os.environ.get("FRONTIER_CODEX_MODEL", "").strip()
    token = os.environ.get("FRONTIER_MODEL_PROXY_TOKEN", "").strip()
    effort = os.environ.get("FRONTIER_CODEX_REASONING_EFFORT", "max").strip()
    if not MODEL_ALIAS.fullmatch(model) or not token or len(token) > 16384:
        raise LaunchError("model alias or run proxy token is invalid")
    if effort not in {"low", "high", "max"}:
        raise LaunchError("GLM reasoning effort must be low, high or max")
    child = os.environ.copy()
    for name in tuple(child):
        if (name.startswith(("ZCODE_", "ANTHROPIC_", "KIMI_")) or name.endswith("_API_KEY")
                or name in {"OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY",
                            "META_API_KEY", "LD_LIBRARY_PATH",
                            "FRONTIER_MODEL_DIRECT_BASE_URL", "MODAL_PROXY_TOKEN_ID",
                            "MODAL_PROXY_TOKEN_SECRET"}):
            child.pop(name)
    home = WORKSPACE / ".frontier-zcode-home"
    temp = WORKSPACE / ".frontier-zcode-tmp"
    for path in (home, temp):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
        path.mkdir(mode=0o700)
    config_dir = home / ".zcode/cli"
    config_dir.mkdir(parents=True, mode=0o700)
    write_json(config_dir / "config.json", {
        "permission": {"mode": "yolo"}, "plugins": {"enabled": False},
        "features": {"memory": False, "mcp": False}, "memory": {"use": False},
        "skills": {"roots": [str(SKILLS)]},
        "modelStream": {"idleTimeoutMs": 3600000}, "network": {"timeout": 3600000}})
    provider_path = home / "provider_config.json"
    write_json(provider_path, provider_config(model_url, model, token, effort))
    child.update({"HOME": str(home), "TMPDIR": str(temp),
                  "PATH": f"{SUBMIT_BIN}:{child.get('PATH', '/usr/bin:/bin')}",
                  "ZCODE_PERSONAL_PROVIDER_CONFIG_FILE": str(provider_path),
                  "ZCODE_BUILTIN_PROVIDER_CONFIG_FILE": str(RUNTIME / "app/provider/zcode-builtin.json"),
                  "ZCODE_DATA_BASE_DIR": str(home)})
    return child


def command() -> list[str]:
    return [str(RUNTIME / "node"), "--require", str(RUNTIME / "fetch-sanitize.cjs"),
            str(RUNTIME / "app/zcode.cjs"), "--mode", "yolo",
            "--cwd", str(WORKSPACE), "--output-format", "stream-json",
            "--disallowed-tools", "WebSearch,WebFetch", "-p", PROMPT]


def accepted_final() -> bool:
    path = WORKSPACE / ".frontier/final-submission.json"
    try:
        metadata = path.lstat()
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077
                or not 0 < metadata.st_size <= 16384):
            return False
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return False
    return (isinstance(value, dict)
            and set(value) == {"schema_version", "operation", "state", "payload_id", "verification_id"}
            and value.get("schema_version") == "frontier.candidate-submit-result/v1"
            and value.get("operation") == "final" and value.get("state") == "accepted"
            and isinstance(value.get("payload_id"), str) and value["payload_id"].startswith("payload_")
            and isinstance(value.get("verification_id"), str)
            and value["verification_id"].startswith("verification_"))


def main() -> int:
    try:
        child = prepare_environment()
        os.chdir(WORKSPACE)
        result = subprocess.run(command(), env=child, check=False)
        if accepted_final():
            return 0
        if result.returncode == 0:
            raise LaunchError("official harness exited without an accepted final receipt")
        return result.returncode if 0 < result.returncode < 256 else 2
    except (LaunchError, OSError, ValueError) as error:
        print(f"frontier-zcode-launch: ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
