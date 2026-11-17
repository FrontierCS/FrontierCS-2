# The agent's image of a task whose agent leases GPU runners: the task's agent image, which
# `fcs_task` names `agent-base`, with `frontier-gpu` and its skill where FrontierCS's Codex
# candidate installs them, outside /workspace, which the agent's sandbox fills at its start.
FROM agent-base
COPY --chmod=0555 frontier_gpu.py /opt/frontier/candidate/bin/frontier-gpu
# A `COPY` gives the directories it creates its `--chmod` mode, and the image runs as the agent, so
# no `RUN` can fix them later: `a+rX,a-w` makes the skill 0444 and leaves its directories 0755.
COPY --chmod=a+rX,a-w SKILL.md /opt/frontier/candidate/skills/frontier-gpu/SKILL.md
