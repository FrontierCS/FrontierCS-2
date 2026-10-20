"""Trusted stand-in for ``secagent`` inside the episode runner.

AgentDojo (as modified by the Progent authors) calls ``secagent`` at five points:
tool wrapping at suite import, the suite's always-allowed/blocked defaults, policy
generation on the user query, policy update after tool results, and the per-call
check inside each wrapped tool. This module keeps the trusted halves (tool
descriptors, argument binding, tool execution) and forwards every policy decision
to the current episode's sandboxed worker via ``bind_episode``.
"""

import inspect
import json
import sys
import types
import builtins
from functools import wraps
from typing import Any, get_args, get_origin, get_type_hints

from docstring_parser import parse
from pydantic import Field, create_model

_available_tools = []      # descriptors in registration order (upstream Tool dicts)
_base_calls = []           # suite defaults replayed into every fresh worker
_episode = None            # object with generate/update/check, set per episode
_bypass = False            # attack setup runs the ground-truth pipeline without policy


class ValidationError(Exception):
    pass


def bind_episode(episode):
    global _episode
    _episode = episode


def tool_descriptors():
    return list(_available_tools)


def base_calls():
    return list(_base_calls)


# ---- upstream signatures used by AgentDojo -------------------------------------

def _check_type_hints(type_hint):
    def is_builtin_type(t):
        return t in vars(builtins).values() or t in vars(types).values()

    def recursive(t):
        if is_builtin_type(t):
            return True
        origin = get_origin(t)
        if origin is not None:
            return all(recursive(arg) for arg in get_args(t))
        return False

    return recursive(type_hint)


def secure_tool_wrapper(tool: Any) -> Any:
    # Descriptor construction copied from upstream secagent/tool.py so the policy
    # code sees byte-identical TOOLS text for reference and candidate.
    function_docs = parse(tool.__doc__)
    args_docs = function_docs.params
    sig = inspect.signature(tool)
    fields = {}
    if len(args_docs) == 0:
        for name, param in sig.parameters.items():
            type_hint = get_type_hints(tool).get(name, Any)
            if not _check_type_hints(type_hint):
                continue
            fields[name] = (type_hint, ...) if param.default is inspect.Parameter.empty else (type_hint, param.default)
    else:
        for x in args_docs:
            name = x.arg_name
            type_hint = get_type_hints(tool).get(name, Any)
            if not _check_type_hints(type_hint):
                continue
            default = sig.parameters.get(name).default
            if default is inspect.Parameter.empty:
                fields[name] = (type_hint, Field(..., description=x.description))
            else:
                fields[name] = (type_hint, Field(default=default, description=x.description))
    args_model = create_model(tool.__name__, **fields)
    properties = args_model.model_json_schema()["properties"]
    _available_tools.append({"name": tool.__name__, "description": function_docs.description,
                             "args": properties})
    public = set(properties)

    @wraps(tool)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        if not _bypass:
            arguments = {k: v for k, v in bound.arguments.items() if k in public}
            check_tool_call(tool.__name__, json.loads(json.dumps(arguments, default=str)))
        return tool(*args, **kwargs)
    return wrapper


def apply_secure_tool_wrapper(input: Any) -> Any:
    if isinstance(input, list):
        return [secure_tool_wrapper(f) for f in input]
    return secure_tool_wrapper(input)


def update_always_allowed_tools(tools, allow_all_no_arg_tools: bool = False) -> None:
    _base_calls.append({"fn": "always_allowed", "tools": list(tools), "no_arg": bool(allow_all_no_arg_tools)})


def update_always_blocked_tools(tools) -> None:
    _base_calls.append({"fn": "always_blocked", "tools": list(tools)})


def get_current_config():
    return {"bypass": _bypass}


def update_security_policy(policy) -> None:
    # Only used by attacks/base_attacks.py around the ground-truth injection probe:
    # update_security_policy(None) ... update_security_policy(previous).
    global _bypass
    _bypass = policy is None


def reset_security_policy(include_human_policy=False) -> None:
    pass  # every episode gets a fresh worker


def generate_security_policy(query: str, manual_check=False) -> None:
    if _episode is not None and not _bypass:
        _episode.generate(query)


def generate_update_security_policy(tool_call_param, tool_call_result, manual_check=True) -> None:
    if _episode is not None and not _bypass:
        _episode.update(tool_call_param, tool_call_result)


def check_tool_call(tool_name, kwargs) -> None:
    if _episode is None:
        raise ValidationError(f"The tool '{tool_name}' is not allowed.")
    error = _episode.check(tool_name, kwargs)
    if error is not None:
        raise ValidationError(error)


def get_available_tools():
    return tool_descriptors()


def update_available_tools(tools) -> None:
    _available_tools[:] = list(tools)
