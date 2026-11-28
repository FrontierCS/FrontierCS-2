# One SWE-bench instance's images, on its SWE-bench image TEST_SANDBOX_BASE (`swe.star` gives each
# instance's, by digest, and its source preparation): target `candidate` is the interactive testbed
# that serves the model's tool calls, target `check` the worker that runs the instance's tests.
# TEST_SANDBOX_SOURCE selects the source preparation:
#   declared     the instance image's source as is
#   restored     after the dependency preparation, check out TEST_SANDBOX_RESTORE_COMMIT again
#   restored-ra  restored, and pytest reports with -rA
# Pumpkin starts no container as root, where the source's services start as root and run the
# model's commands as uid 10001: here each container runs whole as uid 10001, which owns /testbed.
# The two targets differ in what that user may reach besides: in `candidate`, as in the source,
# neither the instance's tests nor its Python environment; in `check`, both, as the tests install
# the patched source into that environment.
ARG TEST_SANDBOX_BASE
FROM docker.io/astral/uv:0.8.22@sha256:ae1ba4db410fbe1d9eaf16c51add8e7608b271ebbb69bd58a56e80384fd190aa AS uv
FROM ${TEST_SANDBOX_BASE} AS prepared
ARG TEST_SANDBOX_INSTANCE
ARG TEST_SANDBOX_SOURCE=declared
ARG TEST_SANDBOX_RESTORE_COMMIT=
ARG TEST_SANDBOX_PYTEST_ADDOPTS=
USER root
ENV UV_PYTHON_INSTALL_DIR=/opt/frontier/python-standalone
# The worker's packages, each file pinned by SHA-256 (`requirements.txt` is `uv pip compile` of
# `requirements.in`, the source's list less `cryptography`, which checked the signatures that the
# scheduler's token replaces, `judge/serving_eval/sidecar_env.py`). Only this step runs uv, so it is
# mounted rather than copied. The worker's files are read-only and root's from this step on, so no
# later step writes them again into a layer of its own.
COPY requirements.txt /opt/frontier/worker-requirements.txt
RUN --mount=from=uv,source=/uv,target=/opt/frontier/bin/uv \
    /opt/frontier/bin/uv python install 3.12.11 \
    && /opt/frontier/bin/uv venv --python 3.12.11 /opt/frontier/worker-python \
    && /opt/frontier/bin/uv pip install --python /opt/frontier/worker-python/bin/python \
       --require-hashes --only-binary=:all: --no-deps -r /opt/frontier/worker-requirements.txt \
    && chown -R root:root /opt/frontier/worker-python /opt/frontier/python-standalone \
    && chmod -R a-w /opt/frontier/worker-python /opt/frontier/python-standalone
COPY judge/instance_manifest.py judge/instance_manifest.json /opt/frontier/judge/
COPY judge/serving_eval/__init__.py judge/serving_eval/swe_patch_disposition.py judge/serving_eval/accuracy.py \
     judge/serving_eval/swe_offline_setup.py /opt/frontier/judge/serving_eval/
COPY judge/serving_eval/swe_data/NOTICE judge/serving_eval/swe_data/instances.json \
     judge/serving_eval/swe_data/provenance.json /opt/frontier/judge/serving_eval/swe_data/
RUN PYTHONPATH=/opt/frontier/judge /opt/frontier/worker-python/bin/python -P -m serving_eval.swe_offline_setup --instance-id "$TEST_SANDBOX_INSTANCE" --phase prepare
# Preserve upstream-compatible dependency preparation, then restore the declared source. The
# upstream image used -rA in tracked tox.ini; retain reporting outside source: only `restored-ra`
# gives pytest an option, and an empty PYTEST_ADDOPTS adds none.
RUN test "$TEST_SANDBOX_SOURCE" = declared \
    || git -c core.fileMode=false checkout --force --detach "$TEST_SANDBOX_RESTORE_COMMIT"
ENV PYTEST_ADDOPTS=${TEST_SANDBOX_PYTEST_ADDOPTS}
RUN chmod -R a-w /opt/frontier/swe-wheelhouse
RUN (getent group 10001 >/dev/null || groupadd --gid 10001 frontier-tool) \
    && (getent passwd 10001 >/dev/null || useradd --uid 10001 --gid 10001 --home-dir /home/frontier-tool --shell /bin/bash frontier-tool) \
    && install -d -o 10001 -g 10001 -m 0700 /home/frontier-tool /var/tmp/frontier-tool /var/tmp/frontier-tool/cache \
    && chown -R 10001:10001 /testbed \
    && chown -R root:root /opt/frontier/judge \
    && chmod -R a-w /opt/frontier/judge \
    && find /opt/frontier/judge -type d -exec chmod 0555 {} + \
    && find /opt/frontier/judge -type f -exec chmod 0444 {} + \
    && chmod 0400 /opt/frontier/judge/serving_eval/swe_data/instances.json /opt/frontier/judge/serving_eval/swe_data/provenance.json
# A Pumpkin sandbox's /tmp may be read-only, so the services, the model's commands and the tests
# keep their temporary files below /var/tmp, where the source has them below /tmp.
ENV PYTHONPATH=/opt/frontier/judge PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TMPDIR=/var/tmp \
    PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/frontier/swe-wheelhouse PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /testbed
ENTRYPOINT []
CMD []

FROM prepared AS runtime
# Mutable runtime boundaries are a thin layer after immutable offline preparation.
COPY --chmod=0444 judge/serving_eval/sidecar_env.py judge/serving_eval/swe_correctness_service.py \
     judge/serving_eval/swe_test_worker.py judge/serving_eval/swe_syntax_probe.py \
     judge/serving_eval/swe_tool_exec.py judge/serving_eval/swe_case_failure.py \
     judge/serving_eval/swe_tool_service.py judge/serving_eval/testbed_reset.py \
     /opt/frontier/judge/serving_eval/
RUN /opt/frontier/worker-python/bin/python -P -c "import serving_eval.swe_tool_service, serving_eval.swe_correctness_service, serving_eval.swe_test_worker"

FROM runtime AS offline-check
ARG TEST_SANDBOX_INSTANCE
# The source runs this check with no network; `pumpkin check` takes no `RUN --network`, and pip
# here reads the wheelhouse alone (PIP_NO_INDEX).
RUN PYTHONPATH=/opt/frontier/judge /opt/frontier/worker-python/bin/python -P -m serving_eval.swe_offline_setup --instance-id "$TEST_SANDBOX_INSTANCE" --phase verify

FROM runtime AS candidate
COPY --from=offline-check --chmod=0444 /opt/frontier/swe-wheelhouse/preflight.json /opt/frontier/swe-wheelhouse/preflight.json
USER 10001:10001

FROM candidate AS check
USER root
RUN chown -R 10001:10001 /opt/miniconda3/envs/testbed \
    && chown 10001:10001 /opt/frontier/judge/serving_eval/swe_data/instances.json /opt/frontier/judge/serving_eval/swe_data/provenance.json
USER 10001:10001
