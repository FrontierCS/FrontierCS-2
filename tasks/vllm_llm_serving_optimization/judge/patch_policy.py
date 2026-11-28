"""Original Python-only patch policy; static acceptance is not a task score.

Extracted from Frontier-CS 1bccbadc6bead714d25c546cf954a37bb0791fd6.
Keep this policy separate from unresolved native scoring and runtime choices.
"""
from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_PATCH_BYTES = 1_500_000
MAX_CHANGED_FILES = 80

STRONGLY_ALLOWED_PATTERNS = (
    "vllm/v1/core/**",
    "vllm/v1/core/sched/**",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/config/scheduler.py",
    "vllm/config/cache.py",
)

CONDITIONALLY_ALLOWED_PATTERNS = (
    "vllm/v1/worker/**",
    "vllm/v1/engine/**",
    "vllm/v1/executor/**",
    "vllm/v1/request.py",
    "vllm/v1/outputs.py",
    "vllm/v1/serial_utils.py",
    "vllm/entrypoints/openai/protocol.py",
    "vllm/entrypoints/openai/serving_engine.py",
    "vllm/entrypoints/openai/serving_chat.py",
    "vllm/entrypoints/openai/serving_completion.py",
    "vllm/sampling_params.py",
)

DENIED_PATTERNS = (
    "csrc/**",
    "cmake/**",
    "CMakeLists.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "requirements/**",
    "requirements*.txt",
    "tests/**",
    "benchmarks/**",
    "benchmark/**",
    "docs/**",
    "examples/**",
    "tools/**",
    ".buildkite/**",
    ".github/**",
    "docker/**",
    "Dockerfile*",
    ".dockerignore",
    "vllm/model_executor/models/**",
    "vllm/model_executor/model_loader/**",
    "vllm/transformers_utils/**",
    "vllm/lora/**",
    "vllm/distributed/**",
    "vllm/entrypoints/llm.py",
    "vllm/entrypoints/api_server.py",
    "vllm/entrypoints/cli/**",
    "vllm/version.py",
    "vllm/_version.py",
)

# Source-line tokens that are forbidden in added code: benchmark/dataset names
# (anti hard-coding) and secret/judge environment access (anti exfiltration and
# anti benchmark-detection).
HARD_CODE_TOKENS = (
    "swebench",
    "swe-bench",
    "swe_bench",
    "sweb.eval",
    "minisweagent",
    "mini-swe",
    "princeton-nlp",
    "SWE-bench_Verified",
)

SECRET_TOKENS = (
    "MODAL_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB",
    "FRONTIER_",
    "HARBOR_",
    "JUDGE_URL",
    "RUN_OUTPUT_DIR",
    "scheduler_timestamps",
)

HARD_CODE_TOKEN_RE = re.compile(
    "|".join(re.escape(token) for token in HARD_CODE_TOKENS),
    re.IGNORECASE,
)
SECRET_TOKEN_RE = re.compile("|".join(re.escape(token) for token in SECRET_TOKENS))

# Patches must stay Python-only because the Modal build uses VLLM_USE_PRECOMPILED
# (prebuilt CUDA kernels, Python layer rebuilt). New .py/.pyi files are allowed.
ALLOWED_SOURCE_EXTENSIONS = (".py", ".pyi")


@dataclass(frozen=True)
class PatchFile:
    old_path: str
    new_path: str
    added_lines: tuple[str, ...]
    removed_lines: tuple[str, ...]

    @property
    def path(self) -> str:
        return self.new_path if self.new_path != "/dev/null" else self.old_path


def _match(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _is_allowed_source_path(path: str) -> bool:
    return _match(path, STRONGLY_ALLOWED_PATTERNS) or _match(path, CONDITIONALLY_ALLOWED_PATTERNS)


def _parse_patch(text: str) -> list[PatchFile]:
    files: list[PatchFile] = []
    current_old = ""
    current_new = ""
    added: list[str] = []
    removed: list[str] = []
    in_file = False

    for line in text.splitlines():
        if line.startswith("diff --git "):
            if in_file:
                files.append(PatchFile(current_old, current_new, tuple(added), tuple(removed)))
            in_file = True
            current_old = ""
            current_new = ""
            added = []
            removed = []
            continue
        if not in_file:
            continue
        if line.startswith("--- "):
            current_old = line[4:].strip()
            if current_old.startswith("a/"):
                current_old = current_old[2:]
            continue
        if line.startswith("+++ "):
            current_new = line[4:].strip()
            if current_new.startswith("b/"):
                current_new = current_new[2:]
            continue
        if line.startswith("+") and not line.startswith("+++ "):
            added.append(line[1:])
            continue
        if line.startswith("-") and not line.startswith("--- "):
            removed.append(line[1:])

    if in_file:
        files.append(PatchFile(current_old, current_new, tuple(added), tuple(removed)))
    return files


def _validate_patch_path(path: str, metrics: dict[str, Any]) -> tuple[bool, str]:
    if not path or path == "/dev/null":
        return True, ""
    if path.startswith("/") or ".." in Path(path).parts:
        return False, f"unsafe patch path: {path}"
    if _match(path, DENIED_PATTERNS):
        return False, f"changed file is outside task boundary: {path}"
    if not _is_allowed_source_path(path):
        return False, f"changed file is not allowlisted: {path}"
    if not path.endswith(ALLOWED_SOURCE_EXTENSIONS):
        return False, f"only Python source changes are allowed (VLLM_USE_PRECOMPILED build): {path}"
    return True, ""


def validate_patch(patch_path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not patch_path.exists():
        return False, "solution patch does not exist", {}
    size = patch_path.stat().st_size
    if size > MAX_PATCH_BYTES:
        return False, f"patch is too large ({size} bytes > {MAX_PATCH_BYTES})", {}
    text = patch_path.read_text(encoding="utf-8", errors="replace")
    patch_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    files = _parse_patch(text)
    metrics: dict[str, Any] = {
        "patch_bytes": size,
        "patch_sha256": patch_hash,
        "changed_files": len(files),
    }
    # Binary hunks (`git diff --binary`) carry a base85 payload the +line token
    # scanners below never see, so they could smuggle secret/benchmark access or
    # non-Python content past the policy while `git apply` honours them. The build
    # is Python-only (VLLM_USE_PRECOMPILED), so reject any binary patch outright
    # (audit finding: binary-hunk policy bypass).
    if "GIT binary patch" in text or re.search(r"^Binary files .* differ$", text, re.MULTILINE):
        return False, "binary patch hunks are not allowed (Python-only build; use a text diff)", metrics
    if len(files) > MAX_CHANGED_FILES:
        return False, f"too many changed files ({len(files)} > {MAX_CHANGED_FILES})", metrics

    for patch_file in files:
        path = patch_file.path
        if patch_file.new_path == "/dev/null":
            return False, f"deleting source files is outside task boundary: {patch_file.old_path}", metrics
        if patch_file.old_path != "/dev/null" and patch_file.old_path != patch_file.new_path:
            ok, error = _validate_patch_path(patch_file.old_path, metrics)
            if not ok:
                return False, f"rename/copy source is outside task boundary: {error}", metrics

        ok, error = _validate_patch_path(path, metrics)
        if not ok:
            return False, error, metrics
        if not path or path == "/dev/null":
            return False, "could not determine changed path from patch", metrics

        added_text = "\n".join(patch_file.added_lines)
        secret_match = SECRET_TOKEN_RE.search(added_text)
        if secret_match:
            return False, f"{path}: judge/secret environment access is forbidden ({secret_match.group(0)})", metrics
        hard_code_match = HARD_CODE_TOKEN_RE.search(added_text)
        if hard_code_match:
            return False, f"{path}: benchmark-specific token is forbidden ({hard_code_match.group(0)})", metrics

    metrics["valid_patch"] = 1
    return True, "patch accepted by static policy", metrics


def patch_is_empty(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("changed_files", 0)) == 0


