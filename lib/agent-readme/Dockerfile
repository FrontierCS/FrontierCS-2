# The agent's `/frontier/task/README.md`, alone in an image: the README of the task's agent recipe's
# directory, mounted at `task/`, joined with the platform's text here by `join.py`. The file has the
# time of the task's README, as a copy of that file has.
ARG AGENT_BASE
FROM ${AGENT_BASE} AS join
RUN --mount=type=bind,target=/context \
    python3 -I /context/join.py /context/task /context > /README.md \
    && touch --reference=/context/task/README.md /README.md

FROM scratch
COPY --from=join /README.md /README.md
