"""Model access for the trusted episode runner: key resolution, retries, per-episode caps.

The agent model and the policy model are the same task alias. In the platform the
key comes from the additional_keys inventory (``FRONTIER_LITELLM_INVENTORY``); for
local runs ``PROGENT_BASE_URL`` and ``PROGENT_KEY_FILE`` name an OpenAI-compatible
endpoint. Endpoint failures that survive the retries, and an exhausted key budget,
are infrastructure errors: they abort the evaluation instead of scoring a candidate.
"""

import json
import os
import random
import threading
import time
from pathlib import Path

import openai
from openai.types.chat import ChatCompletion

MODEL_ALIAS = "progent-agent-gpt-4o-mini"
MAX_COMPLETION_TOKENS = 4096
# Per-episode caps: about 3x the reference p99 measured in the pilot (55 calls, 215k tokens).
EPISODE_TOKEN_CAP = 650_000
EPISODE_POLICY_CALL_CAP = 120
RETRIES = 6
# 429 is load, not failure: the Driver's per-Run model session cap or a provider limit.
# One call waits it out for up to this long before it counts as an endpoint failure.
RATE_LIMIT_WAIT_SECONDS = 600.0
RATE_LIMIT_MAX_DELAY = 30.0


class InfraError(RuntimeError):
    """Model endpoint unavailable or key budget exhausted: an evaluator error."""


def _inventory_endpoint(key_name):
    inventory = json.loads(Path(os.environ["FRONTIER_LITELLM_INVENTORY"]).read_text())
    entry = next((k for k in inventory["keys"] if k["name"] == key_name), None)
    if entry is None or not entry.get("exposed"):
        raise InfraError(f"model key '{key_name}' is not available in this environment")
    if MODEL_ALIAS not in entry["model_aliases"]:
        raise InfraError(f"model key '{key_name}' does not allow {MODEL_ALIAS}")
    auth = inventory["caller_auth"]
    headers = {auth["header"]: os.environ[auth["token_env"]]}
    return inventory["base_url"], Path(entry["value_path"]).read_text(), headers


def endpoint(key_name):
    if os.environ.get("FRONTIER_LITELLM_INVENTORY"):
        return _inventory_endpoint(key_name)
    base_url = os.environ.get("PROGENT_BASE_URL")
    key_file = os.environ.get("PROGENT_KEY_FILE")
    if not base_url or not key_file:
        raise InfraError("no model endpoint: set FRONTIER_LITELLM_INVENTORY or PROGENT_BASE_URL/PROGENT_KEY_FILE")
    return base_url, Path(key_file).read_text().strip(), {}


def _budget_exhausted(error):
    text = str(error).lower()
    return "budget" in text and ("exceed" in text or "exhaust" in text)


class Gateway:
    """One OpenAI-compatible client shared by every episode in a runner process."""

    def __init__(self, key_name, model=MODEL_ALIAS):
        base_url, key, headers = endpoint(key_name)
        # Local runs may reach the same model under its provider id (e.g. openai/gpt-4o-mini).
        self.model = model if os.environ.get("FRONTIER_LITELLM_INVENTORY") else os.environ.get("PROGENT_MODEL", model)
        self.client = openai.OpenAI(base_url=base_url, api_key=key, default_headers=headers,
                                    max_retries=0, timeout=180.0)
        self.lock = threading.Lock()
        self.totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def complete(self, **kwargs):
        kwargs.setdefault("max_tokens", MAX_COMPLETION_TOKENS)
        delay = 2.0
        attempt = 0
        limited_since = None
        limited_delay = 1.0
        while True:
            try:
                response = self.client.chat.completions.create(model=self.model, **kwargs)
            except openai.RateLimitError as error:
                if _budget_exhausted(error):
                    raise InfraError("model key budget exhausted") from error
                now = time.monotonic()
                limited_since = now if limited_since is None else limited_since
                if now - limited_since >= RATE_LIMIT_WAIT_SECONDS:
                    raise InfraError(f"model endpoint still rate limited after {RATE_LIMIT_WAIT_SECONDS:.0f}s") from error
                time.sleep(limited_delay + random.random())
                limited_delay = min(limited_delay * 2, RATE_LIMIT_MAX_DELAY)
                continue
            except openai.BadRequestError as error:
                if _budget_exhausted(error):
                    raise InfraError("model key budget exhausted") from error
                raise  # request-specific (e.g. context length): the caller's outcome
            except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
                if _budget_exhausted(error):
                    raise InfraError("model key budget exhausted") from error
                raise InfraError(f"model key rejected: {type(error).__name__}") from error
            except (openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError, openai.APIStatusError) as error:
                if _budget_exhausted(error):
                    raise InfraError("model key budget exhausted") from error
                attempt += 1
                if attempt == RETRIES:
                    raise InfraError(f"model endpoint failed after {RETRIES} attempts: {type(error).__name__}") from error
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60.0)
                continue
            usage = response.usage
            with self.lock:
                self.totals["calls"] += 1
                if usage is not None:
                    self.totals["prompt_tokens"] += usage.prompt_tokens or 0
                    self.totals["completion_tokens"] += usage.completion_tokens or 0
            return response


class EpisodeMeter:
    """Counts one episode's model use (agent + policy) and enforces the caps."""

    def __init__(self, gateway):
        self.gateway = gateway
        self.tokens = 0
        self.agent_calls = 0
        self.policy_calls = 0
        self.capped = False

    def _account(self, response):
        if response.usage is not None:
            self.tokens += (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)

    def over_tokens(self):
        return self.tokens >= EPISODE_TOKEN_CAP

    def policy(self, system, user, temperature):
        """Policy-model call on behalf of the sandboxed worker. Returns text or None (refused)."""
        if self.over_tokens() or self.policy_calls >= EPISODE_POLICY_CALL_CAP:
            self.capped = True
            return None
        self.policy_calls += 1
        try:
            response = self.gateway.complete(messages=[{"role": "system", "content": system},
                                                       {"role": "user", "content": user}],
                                             temperature=temperature)
        except openai.BadRequestError as error:
            raise PolicyModelRejected(type(error).__name__) from error
        self._account(response)
        return response.choices[0].message.content or ""

    def agent_client(self):
        return _AgentClient(self)


class PolicyModelRejected(RuntimeError):
    pass


class _Completions:
    def __init__(self, meter):
        self.meter = meter

    def create(self, **kwargs):
        meter = self.meter
        kwargs.pop("model", None)
        if meter.over_tokens():
            meter.capped = True
            return _final_message("Episode model budget reached; stopping.")
        meter.agent_calls += 1
        response = meter.gateway.complete(**kwargs)
        meter._account(response)
        return response


class _Chat:
    def __init__(self, meter):
        self.completions = _Completions(meter)


class _AgentClient:
    """Duck-typed ``openai.OpenAI`` handed to AgentDojo's OpenAILLM."""

    def __init__(self, meter):
        self.chat = _Chat(meter)


def _final_message(text):
    return ChatCompletion.model_validate({
        "id": "episode-cap", "object": "chat.completion", "created": int(time.time()), "model": MODEL_ALIAS,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })
