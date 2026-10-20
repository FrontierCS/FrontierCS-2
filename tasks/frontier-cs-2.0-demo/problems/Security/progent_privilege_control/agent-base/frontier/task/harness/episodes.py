"""Trusted episode runner: one AgentDojo episode per call, policy in a fresh sandboxed worker.

Runner processes are created per suite (AgentDojo wraps a suite's tools at import
time according to ``SECAGENT_SUITE``). Inside a runner, ``run_episode`` builds the
undefended AgentDojo tool-calling pipeline on the task model, binds the trusted
``secagent`` stand-in to a new policy worker on the given source tree, and scores
utility and injection success with AgentDojo's own checks.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRUSTED_APP = Path("/frontier/task/repo")      # Agent copy: the read-only pristine tree
BENCHMARK_VERSION = "v1.1.2"
ATTACK = "important_instructions"

_state = {}


def init_runner(suite_name, key_name, identity_seed=None):
    """ProcessPool initializer: import the trusted stand-in and AgentDojo for one suite.

    ``identity_seed`` (one per evaluation) replaces the injection tasks' fixed
    attacker identities; see attacker_identities.py.
    """
    os.environ["SECAGENT_SUITE"] = suite_name
    os.environ["SECAGENT_UPDATE"] = "True"
    sys.path[:0] = [str(ROOT / "trusted"), str(ROOT), str(TRUSTED_APP / "agentdojo" / "src")]
    import secagent  # the trusted stand-in, never the candidate's package
    assert Path(secagent.__file__).resolve().is_relative_to(ROOT / "trusted"), "wrong secagent resolved"
    from agentdojo.task_suite.load_suites import get_suite
    from llm_gateway import Gateway

    suite = get_suite(BENCHMARK_VERSION, suite_name)
    if identity_seed is not None:
        import attacker_identities
        attacker_identities.rewrite(suite, attacker_identities.mapping(suite_name, identity_seed))
        attacker_identities.self_check(suite)
    _state.update(suite=suite, suite_name=suite_name, gateway=Gateway(key_name),
                  tools=secagent.tool_descriptors(), base=secagent.base_calls())


class _Episode:
    """Forwards the Progent hooks to one worker; a broken worker denies everything."""

    def __init__(self, worker, meter):
        from policy_boundary import ModelRefused
        self.worker, self.meter, self.refused = worker, meter, ModelRefused
        self.failure = None

    def _model(self, system, user, temperature):
        from llm_gateway import PolicyModelRejected
        try:
            text = self.meter.policy(system, user, temperature)
        except PolicyModelRejected as error:
            raise self.refused(f"policy model request rejected: {error}") from None
        if text is None:
            raise self.refused("episode policy-model cap reached")
        return text

    def _call(self, op, payload):
        from policy_boundary import BoundaryFailure
        if self.failure:
            return None
        try:
            return self.worker.call(op, payload, self._model)
        except BoundaryFailure as error:
            self.failure = f"{op}: {error}"
            return None

    def setup(self, tools, base):
        self._call("configure", tools)
        self._call("base", base)

    def generate(self, query):
        self._call("generate", {"query": query})

    def update(self, calls, results):
        self._call("update", {"calls": calls, "results": results})

    def check(self, name, args):
        done = self._call("check", {"name": name, "args": args})
        if done is None:
            return f"The tool '{name}' is not allowed (policy unavailable)."
        return None if done["decision"] == "allow" else done.get("error") or f"The tool '{name}' is not allowed."


def _pipeline(meter):
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
    from llm_gateway import MODEL_ALIAS

    llm = OpenAILLM(meter.agent_client(), MODEL_ALIAS)
    pipeline = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm,
                              ToolsExecutionLoop([ToolsExecutor(), llm])])
    pipeline.name = MODEL_ALIAS
    return pipeline


def run_episode(job):
    """job: dict(tree, user_task, injection_task|None, tag). Returns a result row."""
    import secagent
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.base_tasks import BaseUserTask
    from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages
    from agentdojo.functions_runtime import FunctionsRuntime
    from llm_gateway import EpisodeMeter, InfraError
    from policy_boundary import PolicyWorker

    suite = _state["suite"]
    meter = EpisodeMeter(_state["gateway"])
    pipeline = _pipeline(meter)
    user_task = suite.get_user_task_by_id(job["user_task"])
    injection_task = suite.get_injection_task_by_id(job["injection_task"]) if job["injection_task"] else None
    injections = {}
    if injection_task is not None:
        injections = load_attack(ATTACK, suite, pipeline).attack(user_task, injection_task)

    started = time.monotonic()
    worker = PolicyWorker(job["tree"])
    episode = _Episode(worker, meter)
    error = None
    try:
        episode.setup(_state["tools"], _state["base"])
        secagent.bind_episode(episode)
        environment = suite.load_and_inject_default_environment(injections)
        task_environment = user_task.init_environment(environment)
        pre_environment = task_environment.model_copy(deep=True)
        runtime = FunctionsRuntime(suite.tools)
        messages = []
        try:
            _, _, task_environment, messages, _ = pipeline.query(user_task.PROMPT, runtime, task_environment)
        except InfraError:
            raise
        except Exception as exc:  # agent-side failure (e.g. context overflow): score the state reached
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        output = model_output_from_messages(messages) if messages else None
        trace = functions_stack_trace_from_messages(messages) if messages else []
        utility = bool(suite._check_task_result(user_task, output or "", pre_environment, task_environment, trace))
        security = None
        if injection_task is not None:
            security = bool(suite._check_task_result(injection_task, output or "", pre_environment,
                                                     task_environment, trace))
    finally:
        secagent.bind_episode(None)
        worker.abort()
    return {
        "tag": job["tag"], "suite": _state["suite_name"], "user_task": job["user_task"],
        "injection_task": job["injection_task"], "utility": utility, "breached": security,
        "tokens": meter.tokens, "agent_calls": meter.agent_calls, "policy_calls": meter.policy_calls,
        "capped": meter.capped, "worker_failure": episode.failure, "agent_error": error,
        "seconds": round(time.monotonic() - started, 2),
    }
