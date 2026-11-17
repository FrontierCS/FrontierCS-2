# Privilege Control for Tool-Using LLM Agents

The task statement is at `/frontier/task/repo/README.md`. Read it first.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

Edit `secagent/tool.py` and Python files under `secagent/solver_utils/` there; nothing else
may change. Keep everything you write under `/workspace`.

To run AgentDojo episodes with your policy locally, use
`/frontier/task/repo/tools/run_episodes.py --help`; its harness is beside it in
`/frontier/task/repo/tools/harness/`. The Python environment with AgentDojo, Progent and the
OpenAI client is `/opt/progent/venv` (first on `PATH`). The episodes reach the task model, which
your environment's `PROGENT_MODEL` names, through your sandbox's model endpoint, with these
variables, which your environment also sets; a shell that lacks one sets it so:

```sh
export PROGENT_BASE_URL=http://127.0.0.1:11235/v1   # the model endpoint
export PROGENT_KEY_FILE=/opt/progent/no-key          # a placeholder: the endpoint holds the key
```

Your episodes' model use counts against your budget ($15 of the task model for the whole run,
at this deployment's prices): once it runs out, the endpoint refuses further calls and
`run_episodes.py` stops with an error. The endpoint serves at most 7 requests at once, your own
model's included: keep `--concurrency` at 4 or below.

{platform}
