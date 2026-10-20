"""LiteLLM callback registration for FrontierCS2 compatibility fixes."""

from __future__ import annotations

from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from .responses_tool_order import (
    is_kimi_responses_request,
    repair_responses_input_order,
    repair_tool_result_order,
)


class KimiResponsesToolOrderHook(CustomLogger):
    """Repair only Kimi requests lowered from the Responses API."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del user_api_key_dict, cache, call_type, kwargs
        if not str(data.get("model") or "").endswith("kimi-k2.7-code"):
            return data

        original_input = data.get("input")
        repaired_input = repair_responses_input_order(original_input)
        if repaired_input is not original_input:
            data["input"] = repaired_input

        if is_kimi_responses_request(data.get("model"), data.get("metadata")):
            original_messages = data.get("messages")
            repaired_messages = repair_tool_result_order(original_messages)
            if repaired_messages is not original_messages:
                data["messages"] = repaired_messages
        return data


kimi_responses_tool_order_hook = KimiResponsesToolOrderHook()
