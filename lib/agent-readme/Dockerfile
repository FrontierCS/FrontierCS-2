# The agent's `/frontier/task/README.md`, alone in an image: the README of the task's agent recipe's
# directory, mounted at `task/`, joined with the platform's text here by `join.py`. The file has the
# time of the task's README, as a copy of that file has.
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS join
ARG PUBLIC_METRICS
RUN --mount=type=bind,target=/context \
    python3 -I /context/join.py /context/task /context "$PUBLIC_METRICS" > /README.md \
    && touch --reference=/context/task/README.md /README.md

FROM scratch
COPY --from=join /README.md /README.md
