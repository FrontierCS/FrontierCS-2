"""Untrusted policy process: one fresh isolate instance per episode (policy_boundary.py).

It imports the submitted ``secagent.tool`` and serves the four Progent hooks the
episode runner forwards (generate, update, check, plus the suite's base
allowances). Model access exists only through ``api_request``, which this worker
rebinds to a request/response exchange with the trusted parent; the parent owns
the key, the caps and the accounting.
"""

import json
import os
import resource
import sys

# The statement's 768 MiB of address space, before any candidate code runs.
resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
# Candidate output never reaches the transport or the judge.
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
# Unused by the hooks (and never importable by the policy); importing it would cost
# about 1.5 s of the policy's CPU seconds per episode, so secagent.tool sees it unavailable.
sys.modules["langchain"] = None
# The working directory is the instance's copy of the candidate tree; this file and the
# protocol module are its read-only inputs beside each other.
_tree = os.getcwd()
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), _tree, os.path.join(_tree, "agentdojo", "src")]
from policy_protocol import MAX_ERROR, MAX_REQUEST, encode, parse_parent_message

# Progent-LLM configuration (paper code path): generate + update, update errors ignored.
os.environ.update(SECAGENT_GENERATE="True", SECAGENT_UPDATE="True",
                  SECAGENT_IGNORE_UPDATE_ERROR="True", SECAGENT_POLICY_MODEL="task-policy-model")

_out = os.fdopen(1, "wb", buffering=0, closefd=False)
sys.stdout = sys.stderr  # candidate prints go to /dev/null, never to the transport
_in = sys.stdin.buffer


def _send(message):
    _out.write(encode(message))


def _read():
    raw = _in.readline(MAX_REQUEST + 1)
    if not raw:
        raise SystemExit(0)
    return parse_parent_message(raw)


class ModelUnavailable(RuntimeError):
    pass


def api_request(sys_prompt, user_prompt, temperature=0.0):
    _send({"kind": "llm", "system": str(sys_prompt), "user": str(user_prompt),
           "temperature": float(min(max(temperature, 0.0), 2.0))})
    reply = _read()
    if reply.get("op") != "llm_result":
        raise RuntimeError("protocol violation: expected model result")
    if reply.get("error"):
        raise ModelUnavailable(reply["error"])
    return reply["content"]


import secagent.tool as policy  # noqa: E402  (after the environment is fixed)

policy.api_request = api_request


def _error(exc):
    return (type(exc).__name__ + ": " + str(exc))[:MAX_ERROR]


def main():
    while True:
        message = _read()
        op, payload = message["op"], message.get("payload")
        done = {"kind": "done", "op": op}
        try:
            if op == "configure":
                policy.update_available_tools(payload)
            elif op == "base":
                for item in payload:
                    if item["fn"] == "always_allowed":
                        policy.update_always_allowed_tools(item["tools"], allow_all_no_arg_tools=item["no_arg"])
                    elif item["fn"] == "always_blocked":
                        policy.update_always_blocked_tools(item["tools"])
                    else:
                        raise ValueError("unknown base function")
            elif op == "generate":
                policy.generate_security_policy(payload["query"], manual_check=False)
            elif op == "update":
                policy.generate_update_security_policy(payload["calls"], payload["results"], manual_check=False)
            elif op == "check":
                try:
                    policy.check_tool_call(payload["name"], payload["args"])
                    done["decision"] = "allow"
                except Exception as exc:  # any rejection, including candidate bugs, denies
                    done["decision"] = "deny"
                    done["error"] = str(exc)[:MAX_ERROR]  # upstream message, shown to the agent
            else:
                raise ValueError("unknown operation")
        except Exception as exc:
            done["error"] = _error(exc)
        _send(done)


if __name__ == "__main__":
    main()
