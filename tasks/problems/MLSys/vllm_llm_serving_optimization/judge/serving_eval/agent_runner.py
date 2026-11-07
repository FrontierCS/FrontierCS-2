"""Original SWE episode with an already-prepared local testbed/client.

Only resource construction/ownership changes; Driver lifecycle is not here.
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Any
from .settings import EpisodeSettings as EvalSettings

SUBMIT_SENTINEL = "VLLM_SERVING_OPT_SUBMIT"


SYSTEM_PROMPT = (
    "You are a software engineering agent fixing a bug in a code repository.\n"
    "Your working directory is the repository root.\n"
    "At each step, reply with exactly ONE shell command inside a single fenced\n"
    "```bash\n...\n``` block. Do not include any other text.\n"
    "Inspect files, make edits, and run any checks you need.\n"
    f"When the fix is complete, run: echo {SUBMIT_SENTINEL}\n"
    "and nothing else, to submit your changes."
)


INSTANCE_TEMPLATE = (
    "Resolve the following issue in the repository.\n\n"
    "<issue>\n{problem_statement}\n</issue>\n\n"
    "Begin by exploring the repository. Respond with one ```bash``` command."
)


BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)


@dataclass
class InstanceResult:
    instance_id: str
    latency_seconds: float
    n_calls: int
    exit_status: str
    patch: str = ""
    error: str = ""
    per_call_seconds: list[float] = field(default_factory=list)
    started_monotonic: float = 0.0
    ended_monotonic: float = 0.0


def _parse_action(text: str) -> str | None:
    match = BASH_BLOCK_RE.search(text or "")
    if not match:
        return None
    return match.group(1).strip()


def run_instance(
    instance: dict[str, Any],
    *,
    client,
    prepared_testbed,
    settings: EvalSettings,
) -> InstanceResult:
    instance_id = instance["instance_id"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTANCE_TEMPLATE.format(problem_statement=instance["problem_statement"])},
    ]
    per_call: list[float] = []
    exit_status = "incomplete"
    sandbox = prepared_testbed
    started = time.perf_counter()
    for _ in range(max(1, settings.step_limit)):
        call_start = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=settings.model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_completion_tokens,
                # Stable per-conversation job id (every turn of this instance
                # shares it). Forwarded via vllm_xargs -> sampling_params.
                # extra_args["job_id"], which vanilla vLLM ignores but a
                # job-aware scheduler can use for job-level FCFS. Does not
                # change generated tokens.
                extra_body={"vllm_xargs": {"job_id": str(instance_id)}},
            )
        except Exception as exc:  # noqa: BLE001
            # An API error (e.g. the growing conversation exceeding the model's
            # context window) ends the loop, but any edits the agent already
            # made to the sandbox are real work — capture the patch-so-far
            # instead of discarding it.
            exit_status = "api_error"
            partial_patch = ""
            try:
                partial_patch = sandbox.read_patch() if sandbox is not None else ""
            except Exception:  # noqa: BLE001
                partial_patch = ""
            return InstanceResult(
                instance_id=instance_id,
                latency_seconds=(ended := time.perf_counter()) - started, started_monotonic=started, ended_monotonic=ended,
                n_calls=len(per_call),
                exit_status="limit_with_patch" if partial_patch.strip() else exit_status,
                patch=partial_patch,
                error=type(exc).__name__,
                per_call_seconds=per_call,
            )
        per_call.append(time.perf_counter() - call_start)
        content = completion.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})

        action = _parse_action(content)
        if action is None:
            messages.append(
                {
                    "role": "user",
                    "content": "Reply with exactly one ```bash``` command block.",
                }
            )
            continue
        if SUBMIT_SENTINEL in action:
            exit_status = "submitted"
            break

        code, output = sandbox.run(action, timeout=60)
        output = output[-4000:]
        messages.append(
            {"role": "user", "content": f"(exit={code})\n{output}"}
        )
    patch = sandbox.read_patch() if sandbox is not None else ""
    if exit_status == "incomplete" and patch.strip():
        exit_status = "limit_with_patch"
    return InstanceResult(
        instance_id=instance_id,
        latency_seconds=(ended := time.perf_counter()) - started, started_monotonic=started, ended_monotonic=ended,
        n_calls=len(per_call),
        exit_status=exit_status,
        patch=patch,
        per_call_seconds=per_call,
    )
