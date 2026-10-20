"""Render every chat for one workload in a separate process.

The submitted policy is Python that the retained upstream constructor imports.
Running it here, not inside the evaluator process, keeps policy code away from
the evaluator's in-memory scoring state. Input: the JSON file named by argv[1], {"task": str, "lines": [str]}; the
working directory is the patched repository. policy_sandbox.py starts it. Output
(stdout): one JSON document {"rows": [[chat, pattern] | null, ...]}.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys
import types


def _load_constructor(repo: Path, task: str):
    # inference.py imports its upstream training stack eagerly. The constructor
    # does not use it, so task-owned import adapters retain the original bytes.
    peft = types.ModuleType("peft"); peft.PeftModel = object
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    transformers.BitsAndBytesConfig = object
    tqdm = types.ModuleType("tqdm"); tqdm.tqdm = lambda x, *a, **k: x
    sys.modules.update(peft=peft, transformers=transformers, tqdm=tqdm)
    sys.path[:0] = [str(repo / "scripts"), str(repo)]
    namespace = runpy.run_path("scripts/inference.py", run_name="lrr_authentic_inference")
    return namespace["get_line_to_chat_and_pattern"](task)


def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # Keep the protocol channel private: policy prints must not corrupt it.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    repo = Path.cwd()
    make_chat = _load_constructor(repo, request["task"])
    rows = []
    for line in request["lines"]:
        value = make_chat(line, None)
        if value is None:
            rows.append(None)
            continue
        chat, pattern = value
        if (not isinstance(pattern, str) or not isinstance(chat, list)
                or not all(isinstance(m, dict) and set(m) == {"role", "content"}
                           and all(isinstance(v, str) for v in m.values()) for m in chat)):
            raise TypeError("constructor returned a malformed chat")
        rows.append([chat, pattern])
    protocol.write(json.dumps({"rows": rows}))
    protocol.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
