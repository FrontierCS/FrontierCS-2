# Localized Regex Repair

The task statement is at `/frontier/task/repo/README.md`: the problem, the interface, the
workload and the scoring. This file covers where things are and the task's tools.

The read-only source tree is at `/frontier/task/repo`. Copy it into your workspace and make it
a git repository:

```sh
/frontier/task/prepare_workspace.sh          # -> /workspace/repo
```

The editable files are `scripts/scalpel_to_thinking_steps.py`, `prompts/localization.txt`,
`prompts/5-shot-localization-prompt.txt`, `prompts/1-shot-prompt.txt`, `prompts/5-shot-prompt.txt`
and new or changed files below `frontier_solution/`, at most 20 changed files. Nothing outside
`/workspace` is writable. This task has no Agent GPU: the fixed model, ReDoSHunter and the
scorers run only in the Verifier.
