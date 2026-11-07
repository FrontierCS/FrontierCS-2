# Privilege Control for Tool-Using LLM Agents

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `secagent/tool.py` and Python files under `secagent/solver_utils/` there; nothing else
may change. Nothing outside `/workspace` is writable.

To run AgentDojo episodes with your policy locally (billed to the `explore` model key), use
`/frontier/task/repo/tools/run_episodes.py --help`; its harness is beside it in
`/frontier/task/repo/tools/harness/`. The Python environment with AgentDojo, Progent and the
OpenAI client is `/opt/progent/venv` (first on `PATH`).
