"""Repair the narrow Responses-to-Chat tool ordering emitted by Codex.

Codex can replay one assistant turn as separate Responses items in this order:
function calls, assistant text, then function outputs. LiteLLM 1.95.0 lowers
those items to separate Chat Completions messages. OpenAI-compatible providers
reject the intervening assistant text because tool results must immediately
follow the assistant message containing ``tool_calls``.

This module only joins that assistant text back into the preceding assistant
tool-call message when tool output messages immediately follow. It does not
change call IDs, tool results, user messages, or unrelated conversations.
"""

from __future__ import annotations

import copy
from typing import Any


def is_kimi_responses_request(model: Any, metadata: Any) -> bool:
    """Select only the known Kimi route that needs Responses lowering."""

    route = metadata.get("user_api_key_request_route") if isinstance(metadata, dict) else None
    return str(model or "").endswith("kimi-k2.7-code") and route == "/v1/responses"


def _responses_type(item: Any) -> str:
    return str(item.get("type") or "") if isinstance(item, dict) else ""


def _text_content_is_empty(content: Any) -> bool:
    if content in (None, "", []):
        return True
    if not isinstance(content, list):
        return False
    return all(
        isinstance(part, dict)
        and part.get("type") in ("input_text", "output_text", "text")
        and not str(part.get("text") or "")
        for part in content
    )


def _is_text_assistant_item(item: Any) -> bool:
    if not isinstance(item, dict) or item.get("role") != "assistant":
        return False
    if _responses_type(item) not in ("", "message"):
        return False
    content = item.get("content")
    if isinstance(content, str):
        return bool(content)
    return isinstance(content, list) and bool(content) and all(
        isinstance(part, dict) and part.get("type") in ("output_text", "text")
        for part in content
    )


def repair_responses_input_order(items: Any) -> Any:
    """Move intervening assistant text before its Responses function calls.

    LiteLLM's proxy pre-call hook receives the original Responses ``input``.
    Putting the text item first lets LiteLLM's existing transformer merge the
    following function calls into that same assistant Chat message.
    """

    if not isinstance(items, list):
        return items

    repaired: list[Any] = []
    index = 0
    while index < len(items):
        if _responses_type(items[index]) != "function_call":
            repaired.append(copy.deepcopy(items[index]))
            index += 1
            continue

        call_end = index
        call_ids: list[str] = []
        while call_end < len(items) and _responses_type(items[call_end]) == "function_call":
            call_id = str(items[call_end].get("call_id") or items[call_end].get("id") or "")
            if not call_id:
                break
            call_ids.append(call_id)
            call_end += 1

        if (
            not call_ids
            or call_end >= len(items)
            or not _is_text_assistant_item(items[call_end])
        ):
            repaired.append(copy.deepcopy(items[index]))
            index += 1
            continue

        output_cursor = call_end + 1
        output_ids: set[str] = set()
        while output_cursor < len(items) and _responses_type(items[output_cursor]) == "function_call_output":
            output_id = str(items[output_cursor].get("call_id") or "")
            if output_id:
                output_ids.add(output_id)
            output_cursor += 1

        if not set(call_ids).issubset(output_ids):
            repaired.append(copy.deepcopy(items[index]))
            index += 1
            continue

        assistant_item = items[call_end]
        if not _text_content_is_empty(assistant_item.get("content")):
            repaired.append(copy.deepcopy(assistant_item))
        repaired.extend(copy.deepcopy(items[index:call_end]))
        index = call_end + 1

    return repaired


def _role(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    return str(message.get("role") or "")


def _has_tool_calls(message: Any) -> bool:
    return isinstance(message, dict) and bool(message.get("tool_calls"))


def _merge_text(left: Any, right: Any) -> Any:
    if right in (None, "", []):
        return left
    if left in (None, "", []):
        return copy.deepcopy(right)
    if isinstance(left, str) and isinstance(right, str):
        return f"{left}\n{right}"
    if isinstance(left, list) and isinstance(right, list):
        return [*copy.deepcopy(left), *copy.deepcopy(right)]
    return None


def _mergeable_text_assistant(message: Any) -> bool:
    if _role(message) != "assistant" or _has_tool_calls(message):
        return False
    if not isinstance(message, dict):
        return False
    if message.get("name") is not None:
        return False
    return set(message).issubset({"role", "content", "reasoning_content"})


def repair_tool_result_order(messages: Any) -> Any:
    """Return a repaired copy of the exact broken Chat message sequence."""

    if not isinstance(messages, list):
        return messages

    repaired: list[Any] = []
    index = 0
    while index < len(messages):
        current = messages[index]
        if _role(current) != "assistant" or not _has_tool_calls(current):
            repaired.append(copy.deepcopy(current))
            index += 1
            continue

        cursor = index + 1
        text_messages: list[dict[str, Any]] = []
        while cursor < len(messages) and _mergeable_text_assistant(messages[cursor]):
            text_messages.append(messages[cursor])
            cursor += 1

        # Do not collapse ordinary adjacent assistant messages. The repair is
        # allowed only when the next message is the tool result whose required
        # adjacency is currently broken.
        if not text_messages or cursor >= len(messages) or _role(messages[cursor]) != "tool":
            repaired.append(copy.deepcopy(current))
            index += 1
            continue

        merged = copy.deepcopy(current)
        can_merge = True
        for text_message in text_messages:
            source_content = text_message.get("content")
            if not _text_content_is_empty(source_content):
                content = _merge_text(merged.get("content"), source_content)
                if content is None:
                    can_merge = False
                    break
                merged["content"] = content
            source_reasoning = text_message.get("reasoning_content")
            if source_reasoning not in (None, "", []):
                reasoning = _merge_text(merged.get("reasoning_content"), source_reasoning)
                if reasoning is None:
                    can_merge = False
                    break
                merged["reasoning_content"] = reasoning

        if not can_merge:
            repaired.append(copy.deepcopy(current))
            index += 1
            continue

        repaired.append(merged)
        index = cursor

    return repaired
