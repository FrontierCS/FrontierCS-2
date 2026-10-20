"""Bounded private observations of the unchanged RAGEN execution path."""
from __future__ import annotations

import functools
import json
import math
import os
from pathlib import Path
import re
import stat

SCHEMA = "ragen.private-execution/v1"
MAX_RECORD_BYTES = 8192
MAX_RUN_BYTES = 4 * MAX_RECORD_BYTES
RECORDS = ("trainer", "rollout", "actor", "critic")
DIAGNOSTIC_SCHEMA = "ragen.private-diagnostic/v1"
DIAGNOSTIC_EVENTS = 16
DIAGNOSTIC_BYTES = 4096
DIAGNOSTIC_COUNT = 1000000
FAILURE_LOCATIONS = {
    "response_adapter": "CTX_ENV_INPUTS", "environment_step": "ENV_STEP",
    "formulate": "CTX_FORMULATE", "structure": "CTX_VALIDATE",
    "unknown": "UNKNOWN",
}
EXCEPTION_CLASSES = (ValueError, TypeError, RuntimeError, KeyError, IndexError,
                     AttributeError, AssertionError, NotImplementedError,
                     MemoryError, OSError, OverflowError, ZeroDivisionError)
_DIAGNOSTICS = {}
LOSS_KEYS = {"actor": ("actor/pg_loss", "actor/entropy_loss", "actor/kl_loss"),
             "critic": ("critic/vf_loss",)}
EXPECTED_CONFIG = {
    "training_steps": 1, "val_before_train": False, "test_freq": -1,
    "critic_warmup": 0, "adv_estimator": "gae", "group_count": 6,
    "group_size": 4, "filter_ratio": 0.25, "max_turns": 3, "response_length": 96,
}
CONFIG_PATHS = {
    "training_steps": "trainer.total_training_steps",
    "val_before_train": "trainer.val_before_train", "test_freq": "trainer.test_freq",
    "critic_warmup": "trainer.critic_warmup", "adv_estimator": "algorithm.adv_estimator",
    "group_count": "es_manager.train.env_groups", "group_size": "es_manager.train.group_size",
    "filter_ratio": "actor_rollout_ref.rollout.rollout_filter_ratio",
    "max_turns": "agent_proxy.max_turn", "response_length": "actor_rollout_ref.rollout.response_length",
}


STRUCTURE_REASONS = frozenset(['binary_masks', 'empty_response', 'environment_identity', 'executed_action', 'finite_tensors', 'group_identity', 'initial_state_group', 'left_padding', 'loss_padding', 'original_reward', 'padding_tokens', 'position_ids', 'required_fields', 'row_count', 'shifted_responses', 'tensor_shapes', 'terminal_reward', 'trainable_tokens', 'training_reward'])


class CandidateStructureError(RuntimeError):
    def __init__(self, reason):
        if reason not in STRUCTURE_REASONS:
            raise ValueError("unknown candidate structure reason")
        self.reason = reason
        super().__init__(reason)


class ExecutionEvidenceError(RuntimeError):
    def __init__(self, evidence):
        super().__init__("numerical execution evidence is missing or invalid")
        self.private_execution_evidence = evidence


def validate_trajectory_layout(rows, env_ids, expected_env_ids, pad_token_id):
    """Check the public tensor layout without imposing a prompt representation.

    Lists come from Tensor.tolist() at the existing validation boundary. These
    checks validate cooperative execution; they do not isolate submitted code.
    """
    inputs = rows["input_ids"]
    count = len(expected_env_ids)
    if not inputs or len(inputs) != count or not isinstance(inputs[0], list):
        raise CandidateStructureError("row_count")
    width = len(inputs[0])
    if width < 2:
        raise CandidateStructureError("empty_response")
    for name, values in rows.items():
        expected_width = width if name in {
            "input_ids", "attention_mask", "position_ids"} else width - 1
        if (not isinstance(values, list) or len(values) != count
                or any(not isinstance(row, list) or len(row) != expected_width
                       for row in values)):
            raise CandidateStructureError("tensor_shapes")
    if env_ids != expected_env_ids:
        raise CandidateStructureError("environment_identity")
    for attention, positions in zip(rows["attention_mask"], rows["position_ids"]):
        cumulative = []
        total = 0
        for visible in attention:
            total += visible
            cumulative.append(total)
        # RAGEN uses one-based positions; pinned veRL also supports zero-based.
        zero_based = [max(value - 1, 0) for value in cumulative]
        if positions != cumulative and positions != zero_based:
            raise CandidateStructureError("position_ids")
    for ids, attention, loss, rewards, original in zip(
            inputs, rows["attention_mask"], rows["loss_mask"],
            rows["rm_scores"], rows["original_rm_scores"]):
        if any(value not in (0, 1) for value in attention + loss):
            raise CandidateStructureError("binary_masks")
        if any(mask == 0 and token != pad_token_id
               for token, mask in zip(ids, attention)):
            raise CandidateStructureError("padding_tokens")
        # Upstream loss_mask indexes source-token positions (input_ids[:, :-1]).
        if any(mask and not visible for mask, visible in zip(loss, attention[:-1])):
            raise CandidateStructureError("loss_padding")
        if any(value != 0 for value in rewards[:-1] + original[:-1]):
            raise CandidateStructureError("terminal_reward")


def _diagnostic_body(record):
    return json.dumps(record, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("ascii")


def _validate_event(event):
    if type(event) is not dict:
        raise ValueError("diagnostic_schema")
    kind = event.get("event")
    if kind == "response":
        counts = {"rows", "nonempty_rows", "total_length", "min_length", "max_length"}
        if (set(event) != counts | {"event", "representation"}
                or event["representation"] not in {"text_characters", "tensor_slots", "unknown"}):
            raise ValueError("diagnostic_schema")
        if event["representation"] == "unknown":
            valid = all(event[k] is None for k in counts)
        else:
            valid = all(type(event[k]) is int and 0 <= event[k] <= DIAGNOSTIC_COUNT for k in counts)
            if valid:
                valid = (event["nonempty_rows"] <= event["rows"]
                         and event["min_length"] <= event["max_length"]
                         and event["rows"] * event["min_length"] <= event["total_length"]
                         <= event["rows"] * event["max_length"]
                         and event["total_length"] >= event["nonempty_rows"]
                         and (event["nonempty_rows"] == 0) == (event["total_length"] == 0)
                         and (event["rows"] > 0 or event["max_length"] == 0))
                if valid and event["representation"] == "tensor_slots":
                    valid = (event["min_length"] == event["max_length"]
                             and event["nonempty_rows"] == (event["rows"] if event["max_length"] else 0))
        if not valid:
            raise ValueError("diagnostic_schema")
    elif kind == "actions":
        if (set(event) != {"event", "requested", "mapped", "executed"}
                or any(type(event[k]) is not int or not 0 <= event[k] <= DIAGNOSTIC_COUNT
                       for k in ("requested", "mapped", "executed"))
                or not event["executed"] <= event["mapped"] <= event["requested"]):
            raise ValueError("diagnostic_schema")
    elif kind in {"formulate", "structure"}:
        allowed = {"entered", "completed"} if kind == "formulate" else {"entered", "passed", "failed"}
        if set(event) != {"event", "status"} or event["status"] not in allowed:
            raise ValueError("diagnostic_schema")
    elif kind == "candidate_rejection":
        if set(event) != {"event", "reason"} or event["reason"] not in STRUCTURE_REASONS:
            raise ValueError("diagnostic_schema")
    elif kind == "failure":
        if (set(event) != {"event", "stage", "exception_class", "location"}
                or event["stage"] not in FAILURE_LOCATIONS
                or event["location"] != FAILURE_LOCATIONS[event["stage"]]
                or event["exception_class"] not in {c.__name__ for c in EXCEPTION_CLASSES} | {"unknown"}):
            raise ValueError("diagnostic_schema")
    else:
        raise ValueError("diagnostic_schema")


def validate_diagnostics(record):
    if (type(record) is not dict or set(record) != {"schema", "events", "dropped_events"}
            or record["schema"] != DIAGNOSTIC_SCHEMA or type(record["events"]) is not list
            or len(record["events"]) > DIAGNOSTIC_EVENTS
            or type(record["dropped_events"]) is not int
            or not 0 <= record["dropped_events"] <= DIAGNOSTIC_COUNT):
        raise ValueError("diagnostic_schema")
    for event in record["events"]:
        _validate_event(event)
    if len(_diagnostic_body(record)) > DIAGNOSTIC_BYTES:
        raise ValueError("diagnostic_cap")


class DiagnosticRecord:
    def __init__(self):
        self.record = {"schema": DIAGNOSTIC_SCHEMA, "events": [], "dropped_events": 0}
        self.failure_seen = False

    def add(self, **event):
        _validate_event(event)
        record = self.record
        if record["dropped_events"] or len(record["events"]) == DIAGNOSTIC_EVENTS:
            record["dropped_events"] += 1
        else:
            record["events"].append(event)
        # Retain an event prefix deterministically; never truncate serialized JSON.
        while len(_diagnostic_body(record)) > DIAGNOSTIC_BYTES and record["events"]:
            record["events"].pop()
            record["dropped_events"] += 1
        validate_diagnostics(record)

    def failure(self, stage, error):
        if not self.failure_seen:
            self.add(event="failure", stage=stage, location=FAILURE_LOCATIONS[stage],
                     exception_class=next((c.__name__ for c in EXCEPTION_CLASSES
                                           if type(error) is c), "unknown"))
            self.failure_seen = True


def _diagnostics(directory):
    return _DIAGNOSTICS.setdefault(str(directory), DiagnosticRecord())


def _observe_responses(outputs):
    counts = dict(rows=None, nonempty_rows=None, total_length=None, min_length=None, max_length=None)
    event = dict(event="response", representation="unknown", **counts)
    batch = getattr(outputs, "batch", None)
    if batch is not None and "responses" in batch:
        shape = getattr(batch["responses"], "shape", ())
        if len(shape) != 2:
            return event
        rows, width = map(int, shape)
        # Width includes padding: these counts do not establish generated text.
        return dict(event="response", representation="tensor_slots", rows=rows,
                    nonempty_rows=rows if width else 0, total_length=rows * width,
                    min_length=width if rows else 0, max_length=width if rows else 0)
    non_tensor = getattr(outputs, "non_tensor_batch", None)
    if not isinstance(non_tensor, dict):
        return event
    text = non_tensor.get("response_texts")
    if text is None:
        return event
    lengths = []
    for value in text:
        if type(value) is not str or len(lengths) >= DIAGNOSTIC_COUNT:
            return event
        lengths.append(len(value))
    return dict(event="response", representation="text_characters", rows=len(lengths),
                nonempty_rows=sum(n > 0 for n in lengths), total_length=sum(lengths),
                min_length=min(lengths, default=0), max_length=max(lengths, default=0))


def observe_context(cls, directory):
    original_inputs, original_formulate = cls.get_env_inputs, cls.formulate_rollouts

    @functools.wraps(original_inputs)
    def inputs(self, outputs, *args, **kwargs):
        diagnostic = _diagnostics(directory)
        diagnostic.add(**_observe_responses(outputs))
        try:
            return original_inputs(self, outputs, *args, **kwargs)
        except BaseException as error:
            diagnostic.failure("response_adapter", error)
            raise

    @functools.wraps(original_formulate)
    def formulate(self, *args, **kwargs):
        diagnostic = _diagnostics(directory)
        diagnostic.add(event="formulate", status="entered")
        try:
            result = original_formulate(self, *args, **kwargs)
        except BaseException as error:
            diagnostic.failure("formulate", error)
            raise
        diagnostic.add(event="formulate", status="completed")
        return result

    cls.get_env_inputs, cls.formulate_rollouts = inputs, formulate


def observe_validation(function, directory):
    @functools.wraps(function)
    def checked(*args, **kwargs):
        diagnostic = _diagnostics(directory)
        diagnostic.add(event="structure", status="entered")
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            diagnostic.add(event="structure", status="failed")
            if type(error) is CandidateStructureError:
                diagnostic.add(event="candidate_rejection", reason=error.reason)
            diagnostic.failure("structure", error)
            raise
        diagnostic.add(event="structure", status="passed")
        return result
    return checked


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _record_fd(directory, kind, flags):
    parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(parent)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("private_directory_mode")
        return os.open(kind + ".json", flags | os.O_NOFOLLOW, 0o600, dir_fd=parent)
    finally:
        os.close(parent)


def write_record(directory, kind, record):
    if kind not in RECORDS:
        raise ValueError("unknown private observation kind")
    body = json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    if len(body) > MAX_RECORD_BYTES:
        raise ValueError("private observation exceeds its byte limit")
    # A second update/rollout must not overwrite the one-iteration receipt.
    fd = _record_fd(directory, kind, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)


def _config(config):
    result = {}
    for key, path in CONFIG_PATHS.items():
        value = config
        for part in path.split("."):
            value = getattr(value, part)
        if type(value) not in (str, int, float, bool) or isinstance(value, str) and len(value) > 32:
            raise ValueError("invalid private runtime configuration scalar")
        result[key] = value
    return result


def observe_trainer(cls, directory):
    original = cls.fit

    @functools.wraps(original)
    def fit(self, *args, **kwargs):
        diagnostic = _diagnostics(directory)
        record = {"kind": "trainer", "uid": os.getuid(), "entered": True,
                  "completed": False, "config": _config(self.config),
                  "use_critic": bool(self.use_critic), "global_step": None}
        try:
            result = original(self, *args, **kwargs)
            record["completed"] = True
            record["global_step"] = int(self.global_steps)
            return result
        except BaseException as error:
            diagnostic.failure("unknown", error)
            raise
        finally:
            validate_diagnostics(diagnostic.record)
            fd = _record_fd(directory, "diagnostics", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(fd, "wb") as stream:
                stream.write(_diagnostic_body(diagnostic.record))
            write_record(directory, "trainer", record)

    cls.fit = fit


def observe_environment(cls, directory):
    original_map, original_states = cls._extract_map_valid_actions, cls.get_rollout_states
    original_step = cls.step

    @functools.wraps(original_step)
    def step(self, *args, **kwargs):
        try:
            result = original_step(self, *args, **kwargs)
        except BaseException as error:
            if self.mode == "train":
                _diagnostics(directory).failure("environment_step", error)
            raise
        if self.mode == "train":
            counters = getattr(self, "_frontier_action_observation", {})
            _diagnostics(directory).add(event="actions",
                requested=counters.get("requested_actions", 0),
                mapped=counters.get("mapped_actions", 0),
                executed=sum(entry["status"].num_actions for entry in self.envs))
        return result

    @functools.wraps(original_map)
    def mapped(self, entry, actions):
        result = original_map(self, entry, actions)
        if self.mode == "train":
            counters = getattr(self, "_frontier_action_observation", None)
            if counters is None:
                counters = dict(input_turns=0, requested_actions=0, mapped_actions=0,
                                empty_action_turns=0, format_penalized_turns=0)
                self._frontier_action_observation = counters
            counters["input_turns"] += 1
            counters["requested_actions"] += len(actions)
            counters["mapped_actions"] += len(result)
            counters["empty_action_turns"] += int(not actions)
            counters["format_penalized_turns"] += int(len(actions) != len(result) or not result)
        return result

    @functools.wraps(original_states)
    def states(self, *args, **kwargs):
        result = original_states(self, *args, **kwargs)
        if self.mode == "train":
            statuses = [entry["status"] for entry in self.envs]
            counters = getattr(self, "_frontier_action_observation", {})
            record = {"kind": "rollout", "uid": os.getuid(), "trajectories": len(result),
                      "executed_actions": sum(s.num_actions for s in statuses),
                      "terminated": sum(int(s.terminated) for s in statuses),
                      "truncated": sum(int(s.truncated) for s in statuses),
                      "success": sum(int(s.terminated and not s.truncated) for s in statuses),
                      **counters}
            write_record(directory, "rollout", record)
        return result

    cls._extract_map_valid_actions, cls.get_rollout_states = mapped, states
    cls.step = step


def _restore(obj, name, had_attribute, previous):
    if had_attribute:
        setattr(obj, name, previous)
    else:
        delattr(obj, name)


def observe_worker(cls, role, directory, torch):
    if role not in LOSS_KEYS:
        raise ValueError("unknown numerical worker role")
    update_name = "update_policy" if role == "actor" else "update_critic"
    original_update = getattr(cls, update_name)

    @functools.wraps(original_update)
    def update(self, data, *args, **kwargs):
        optimizer = getattr(self, role + "_optimizer")
        original_step, original_routine = optimizer.step, self._optimizer_step
        step_owned, routine_owned = "step" in vars(optimizer), "_optimizer_step" in vars(self)
        previous_step, previous_routine = vars(optimizer).get("step"), vars(self).get("_optimizer_step")
        groups = data.non_tensor_batch["group_ids"].tolist()
        record = {"kind": role, "uid": os.getuid(),
                  "rank": int(torch.distributed.get_rank()), "world_size": int(torch.distributed.get_world_size()),
                  "update_routine_entered": 1, "update_routine_completed": 0,
                  "batch_rows": int(data.batch.batch_size[0]), "batch_group_ids": [int(x) for x in groups],
                  "ppo_epochs": int(self.config.ppo_epochs), "mini_batch_size": int(self.config.ppo_mini_batch_size),
                  "micro_batch_size": int(self.config.ppo_micro_batch_size_per_gpu),
                  "use_kl_loss": bool(getattr(self.config, "use_kl_loss", False)),
                  "learning_rates": [number(group["lr"]) for group in optimizer.param_groups],
                  "optimizer_routines_entered": 0, "optimizer_routines_completed": 0,
                  "optimizer_steps_entered": 0, "optimizer_steps_applied": 0, "optimizer_steps_skipped": 0,
                  "gradient_norms": [], "losses": {}, "error_type": None}

        def step(*step_args, **step_kwargs):
            record["optimizer_steps_entered"] += 1
            result = original_step(*step_args, **step_kwargs)
            # This measures return from the original optimizer.step, not a parameter delta.
            record["optimizer_steps_applied"] += 1
            return result

        def routine(*routine_args, **routine_kwargs):
            record["optimizer_routines_entered"] += 1
            before = record["optimizer_steps_applied"]
            result = original_routine(*routine_args, **routine_kwargs)
            record["optimizer_routines_completed"] += 1
            record["optimizer_steps_skipped"] += int(record["optimizer_steps_applied"] == before)
            record["gradient_norms"].append(number(result))
            return result

        optimizer.step, self._optimizer_step = step, routine
        try:
            result = original_update(self, data, *args, **kwargs)
            record["update_routine_completed"] = 1
            for key in LOSS_KEYS[role]:
                if key in result:
                    values = result[key] if isinstance(result[key], (list, tuple)) else [result[key]]
                    if len(values) > 64:
                        raise ValueError("too many private loss observations")
                    record["losses"][key] = [number(value) for value in values]
            return result
        except BaseException as error:
            record["error_type"] = type(error).__name__[:64]
            raise
        finally:
            _restore(optimizer, "step", step_owned, previous_step)
            _restore(self, "_optimizer_step", routine_owned, previous_routine)
            write_record(directory, role, record)

    setattr(cls, update_name, update)


def install(workdir, private_module, directory):
    directory = Path(directory)
    directory.mkdir(mode=0o700)
    hooks = (
        ("ragen/trainer/agent_trainer.py", "observe_trainer", "RayAgentTrainer"),
        ("ragen/llm_agent/es_manager.py", "observe_environment", "EnvStateManager"),
        ("ragen/llm_agent/ctx_manager.py", "observe_context", "ContextManager"),
        ("ragen/llm_agent/agent_proxy.py", "observe_proxy", "LLMAgentProxy"),
        ("ragen/workers/actor/dp_actor.py", "observe_worker", "DataParallelPPOActor, 'actor'"),
        ("ragen/workers/critic/dp_critic.py", "observe_worker", "DataParallelPPOCritic, 'critic'"),
    )
    for relative, function, arguments in hooks:
        extra = ", torch" if function == "observe_worker" else ""
        suffix = ("\n\n# Private verifier observation; original functions remain intact.\n"
                  "import importlib.util as _frontier_private_import, sys as _frontier_private_sys\n"
                  "if '_frontier_private_observation' not in _frontier_private_sys.modules:\n"
                  f"    _frontier_private_spec = _frontier_private_import.spec_from_file_location('_frontier_private_observation', {str(private_module)!r})\n"
                  "    _frontier_private_module = _frontier_private_import.module_from_spec(_frontier_private_spec)\n"
                  "    _frontier_private_sys.modules['_frontier_private_observation'] = _frontier_private_module\n"
                  "    _frontier_private_spec.loader.exec_module(_frontier_private_module)\n"
                  "_frontier_private_module = _frontier_private_sys.modules['_frontier_private_observation']\n"
                  f"_frontier_private_module.{function}"
                  f"({arguments}, {str(directory)!r}{extra})\n")
        with (Path(workdir) / relative).open("a", encoding="utf-8") as stream:
            stream.write(suffix)


def _integer(value, low=0, high=100000):
    return type(value) is int and low <= value <= high


def _numbers(values, low=None, max_count=64):
    return (isinstance(values, list) and 1 <= len(values) <= max_count
            and all(type(v) in (int, float) and math.isfinite(v) and (low is None or v >= low) for v in values))


def _number_shape(values, maximum=64):
    return (isinstance(values, list) and len(values) <= maximum
            and all(v is None or type(v) in (int, float) and math.isfinite(v) for v in values))


def _validate(kind, row):
    """Closed schemas prevent malformed observations becoming private log dumps."""
    if not isinstance(row, dict) or row.get("kind") != kind or not _integer(row.get("uid"), high=2**32 - 1):
        raise ValueError("schema")
    if kind == "trainer":
        if set(row) != {"kind", "uid", "entered", "completed", "config", "use_critic", "global_step"}:
            raise ValueError("schema")
        if (any(type(row[k]) is not bool for k in ("entered", "completed", "use_critic"))
                or row["global_step"] is not None and not _integer(row["global_step"])):
            raise ValueError("schema")
        if not isinstance(row["config"], dict) or set(row["config"]) != set(EXPECTED_CONFIG):
            raise ValueError("configuration")
        if any(type(row["config"][k]) is not type(v) or row["config"][k] != v for k, v in EXPECTED_CONFIG.items()):
            raise ValueError("configuration")
        if row["entered"] is not True or row["completed"] is not True or row["use_critic"] is not True or type(row["global_step"]) is not int or row["global_step"] != 1:
            raise ValueError("incomplete")
    elif kind == "rollout":
        keys = {"trajectories", "executed_actions", "terminated", "truncated", "success", "input_turns",
                "requested_actions", "mapped_actions", "empty_action_turns", "format_penalized_turns"}
        if set(row) != keys | {"kind", "uid"} or not all(_integer(row[k]) for k in keys):
            raise ValueError("schema")
        if (row["trajectories"] != 24 or not 1 <= row["input_turns"] <= 72
                or not 0 <= row["success"] <= row["terminated"] <= 24
                or row["truncated"] != row["terminated"] - row["success"]
                or row["executed_actions"] > min(240, row["mapped_actions"])
                or row["mapped_actions"] > row["requested_actions"]
                or row["empty_action_turns"] > row["format_penalized_turns"]
                or row["format_penalized_turns"] > row["input_turns"]):
            raise ValueError("counts")
    else:
        counts = {"rank", "world_size", "update_routine_entered", "update_routine_completed", "batch_rows",
                  "ppo_epochs", "mini_batch_size", "micro_batch_size", "optimizer_routines_entered",
                  "optimizer_routines_completed", "optimizer_steps_entered", "optimizer_steps_applied", "optimizer_steps_skipped"}
        if set(row) != counts | {"kind", "uid", "batch_group_ids", "use_kl_loss", "learning_rates", "gradient_norms", "losses", "error_type"}:
            raise ValueError("schema")
        if not all(_integer(row[k]) for k in counts) or type(row["use_kl_loss"]) is not bool:
            raise ValueError("schema")
        if row["error_type"] is not None and (not isinstance(row["error_type"], str) or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", row["error_type"]) is None):
            raise ValueError("schema")
        groups = row["batch_group_ids"]
        if (row["rank"] != 0 or row["world_size"] != 1 or row["batch_rows"] != 4
                or row["mini_batch_size"] != 4 or row["micro_batch_size"] != 4 or row["ppo_epochs"] != 1
                or not isinstance(groups, list) or len(groups) != 4
                or not all(_integer(g, high=5) for g in groups) or len(set(groups)) != 1):
            raise ValueError("batch_or_configuration")
        if (not isinstance(row["losses"], dict) or not set(row["losses"]) <= set(LOSS_KEYS[kind])
                or not all(_number_shape(v) for v in row["losses"].values())
                or not _number_shape(row["gradient_norms"]) or not _number_shape(row["learning_rates"], 16)):
            raise ValueError("loss_schema")
        required = set(LOSS_KEYS[kind][:2] if kind == "actor" else LOSS_KEYS[kind])
        if kind == "actor" and row["use_kl_loss"]:
            required.add("actor/kl_loss")
        if not required <= set(row["losses"]) or not all(_numbers(v) for v in row["losses"].values()):
            raise ValueError("missing_or_nonfinite_loss")
        if not _numbers(row["gradient_norms"], low=0) or not _numbers(row["learning_rates"], low=0, max_count=16):
            raise ValueError("missing_or_nonfinite_gradient_or_lr")
        if (row["error_type"] is not None or row["optimizer_steps_skipped"] != 0
                or any(row[k] != 1 for k in ("update_routine_entered", "update_routine_completed",
                       "optimizer_routines_entered", "optimizer_routines_completed", "optimizer_steps_entered", "optimizer_steps_applied"))
                or len(row["gradient_norms"]) != 1):
            raise ValueError("missing_skipped_or_incomplete_update")


def collect(directory):
    report = {"schema": SCHEMA, "valid": False, "issues": [],
              "parameter_change_measured": False, "post_update_learning_measured": False}
    total = 0
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result
    try:
        with os.scandir(directory) as entries:
            for index, entry in enumerate(entries):
                if index >= len(RECORDS) + 1 or entry.name not in {kind + ".json" for kind in (*RECORDS, "diagnostics")}:
                    report["issues"].append("unexpected_record")
                    break
    except OSError:
        report["issues"].append("observation_directory_unavailable")
    for kind in RECORDS:
        row = None
        try:
            fd = _record_fd(directory, kind, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.getuid()
                        or not 0 < info.st_size <= MAX_RECORD_BYTES):
                    raise ValueError("file_type_or_size")
                raw = stream.read(MAX_RECORD_BYTES + 1)
            total += len(raw)
            if len(raw) > MAX_RECORD_BYTES or total > MAX_RUN_BYTES:
                raise ValueError("overflow")
            row = json.loads(raw, object_pairs_hook=pairs,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
            _validate(kind, row)
        except (OSError, ValueError, TypeError, OverflowError, RecursionError) as error:
            # Failed numeric records are still useful; retain only closed-schema data.
            reason = str(error) if type(error) is ValueError else type(error).__name__
            allowed = {"schema", "configuration", "incomplete", "counts", "loss_schema", "batch_or_configuration",
                       "missing_or_nonfinite_loss", "missing_or_nonfinite_gradient_or_lr", "missing_skipped_or_incomplete_update",
                       "file_type_or_size", "private_directory_mode", "overflow", "duplicate_key", "nonfinite_json"}
            if reason not in allowed:
                reason = type(error).__name__
            report["issues"].append(kind + ":" + reason)
            if reason not in {"incomplete", "missing_or_nonfinite_loss", "missing_or_nonfinite_gradient_or_lr", "missing_skipped_or_incomplete_update"}:
                row = None
        report[kind] = row
    report["diagnostics"] = None
    try:
        fd = _record_fd(directory, "diagnostics", os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.getuid()
                    or not 0 < info.st_size <= DIAGNOSTIC_BYTES):
                raise ValueError("diagnostic_cap")
            raw = stream.read(DIAGNOSTIC_BYTES + 1)
        if len(raw) > DIAGNOSTIC_BYTES or total + len(raw) > MAX_RUN_BYTES:
            raise ValueError("diagnostic_cap")
        diagnostic = json.loads(raw, object_pairs_hook=pairs,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        validate_diagnostics(diagnostic)
        report["diagnostics"] = diagnostic
    except FileNotFoundError:
        pass  # An absent observation stays unknown; it proves no stage outcome.
    except (OSError, ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
        report["issues"].append("diagnostics:invalid")
    if report["actor"] and report["critic"] and report["actor"]["batch_group_ids"] != report["critic"]["batch_group_ids"]:
        report["issues"].append("actor_critic_group_mismatch")
    report["valid"] = not report["issues"]
    return report


def observe_proxy(cls, directory):
    """Install trusted action/prompt/assistant checks outside candidate code."""
    import importlib.util
    from ragen.env import REGISTERED_ENV_CONFIGS
    spec = importlib.util.spec_from_file_location(
        "_frontier_action_provenance", Path(__file__).with_name("action_provenance.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_proxy(cls, directory, REGISTERED_ENV_CONFIGS)
