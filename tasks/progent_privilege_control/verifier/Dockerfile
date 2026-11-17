# FrontierCS's shared verifier base, pinned by digest: Ubuntu 24.04 with git, Python and
# zstandard, apt set as in the agent's base, the user 65532 with /frontier its home, the shared
# verifier's launcher, /opt/frontier/verifier, and /opt/frontier on Python's import path.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The task's Python environment, the same pinned closure as the agent's, in a venv of its own,
# on which the shared verifier runs, since the evaluator runs the AgentDojo agent in its process
# (`/opt/frontier/runtime-python` is the venv's Python). Its
# files are pinned by SHA-256, fetched as in the agent's image. `/opt/progent/no-key` stands in for
# the key file the judge's model client reads (`judge/llm_gateway.py:55-62`): the model endpoint
# holds the key, and drops the client's.
COPY agent-base/requirements.txt /opt/progent/requirements.txt
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates git python3 python3-venv \
    && python3 -m venv /opt/progent/venv \
    && /opt/progent/venv/bin/pip install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /opt/progent/requirements.txt \
    && /opt/progent/venv/bin/python -c "import openai, pydantic, jsonschema, docstring_parser, z3, yaml, langchain, zstandard" \
    && printf '#!/bin/sh\nexec /opt/progent/venv/bin/python "$@"\n' > /opt/frontier/runtime-python.task \
    && chmod 0555 /opt/frontier/runtime-python.task \
    && mv /opt/frontier/runtime-python.task /opt/frontier/runtime-python \
    && printf 'unused\n' > /opt/progent/no-key \
    && chmod 0444 /opt/progent/no-key

LABEL org.opencontainers.image.title="FrontierCS2 verifier progent_privilege_control" \
      dev.frontier.problem-id="progent_privilege_control"

# The verifier writes its result and evidence below /frontier/output, which is Pumpkin's
# /pumpkin/out, and its scratch below /frontier, on the container's writable layer.
RUN ln --symbolic /pumpkin/out /frontier/output

# The shared verifier and the launcher the judge runs each policy worker under: the repository's
# `shared/` ones, which the build puts in its context, and which replace the base's.
COPY shared/verifier/verify.py shared/verifier/fcs2_verifier.py \
     shared/verifier/beat_spec_feedback.py shared/lib/edit_scope.py shared/lib/isolate.py \
     /opt/frontier/
# The judge, whole in both modes, as the source has it, with the one line
# that Pumpkin's model endpoint needs changed: it serves 7 requests at once, and the judge keeps
# one episode, with one request open, for each of its `MODEL_SESSIONS`.
COPY --chmod=a+rX,a-w judge /opt/frontier/task
RUN sed --in-place 's/^MODEL_SESSIONS = 8$/MODEL_SESSIONS = 7/' /opt/frontier/task/evaluator.py \
    && grep --quiet '^MODEL_SESSIONS = 7$' /opt/frontier/task/evaluator.py
# The starter that the shared verifier overlays the submission on, the generated files, and each
# mode's selector.
COPY --chmod=a+rX,a-w repo /opt/frontier/task/pristine
COPY --chmod=a+rX,a-w generated/judge/config.yaml /opt/frontier/task/config.yaml
COPY generated/verifier/verifier.json /opt/frontier/verifier.json
COPY datasets/train.json datasets/eval.json /opt/frontier/datasets/
RUN chmod 0444 /opt/frontier/verifier.json /opt/frontier/*.py \
    && chmod 0555 /opt/frontier \
    && chmod --recursive a+rX,a-w /opt/frontier/datasets \
    && chmod a-w /opt/frontier/task /opt/frontier/task/pristine

USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /frontier
