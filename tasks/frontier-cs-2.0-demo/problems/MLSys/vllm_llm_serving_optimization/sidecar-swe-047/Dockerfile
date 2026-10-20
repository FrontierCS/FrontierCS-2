# syntax=docker/dockerfile:1.7
# Rendered by docs/transfers/vllm_llm_serving_optimization/tools/render_release.py.
# One image per original SWE instance serves both the interactive testbeds and
# the check worker of that instance; the Driver pins it by digest.
FROM docker.io/astral/uv:0.8.22@sha256:ae1ba4db410fbe1d9eaf16c51add8e7608b271ebbb69bd58a56e80384fd190aa AS uv
FROM docker.io/swebench/sweb.eval.x86_64.psf_1776_requests-3362@sha256:d1c8c15eeb5e12b7eb3815307d1131d68c72d53419fa2f8b89bc46bc0009a61b AS prepared
USER root
COPY --from=uv /uv /opt/frontier/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/frontier/python-standalone
RUN /opt/frontier/bin/uv python install 3.12.11 \
    && /opt/frontier/bin/uv venv --python 3.12.11 /opt/frontier/worker-python
COPY judge/swe_sidecar_requirements.txt /opt/frontier/worker-requirements.txt
RUN /opt/frontier/bin/uv pip install --python /opt/frontier/worker-python/bin/python \
    -r /opt/frontier/worker-requirements.txt
COPY judge/instance_manifest.py /opt/frontier/judge/instance_manifest.py
COPY judge/instance_manifest.json /opt/frontier/judge/instance_manifest.json
COPY judge/serving_eval/__init__.py /opt/frontier/judge/serving_eval/__init__.py
COPY judge/serving_eval/swe_patch_disposition.py /opt/frontier/judge/serving_eval/swe_patch_disposition.py
COPY judge/serving_eval/accuracy.py /opt/frontier/judge/serving_eval/accuracy.py
COPY judge/serving_eval/swe_offline_setup.py /opt/frontier/judge/serving_eval/swe_offline_setup.py
COPY judge/serving_eval/swe_data/NOTICE /opt/frontier/judge/serving_eval/swe_data/NOTICE
COPY judge/serving_eval/swe_data/instances.json /opt/frontier/judge/serving_eval/swe_data/instances.json
COPY judge/serving_eval/swe_data/provenance.json /opt/frontier/judge/serving_eval/swe_data/provenance.json
RUN PYTHONPATH=/opt/frontier/judge /opt/frontier/worker-python/bin/python -P -m serving_eval.swe_offline_setup --instance-id psf__requests-3362 --phase prepare
RUN chmod -R a-w /opt/frontier/swe-wheelhouse
RUN (getent group 10001 >/dev/null || groupadd --gid 10001 frontier-tool) \
    && (getent passwd 10001 >/dev/null || useradd --uid 10001 --gid 10001 --home-dir /home/frontier-tool --shell /bin/bash frontier-tool) \
    && install -d -o 10001 -g 10001 -m 0700 /home/frontier-tool /tmp/frontier-tool /tmp/frontier-tool/cache \
    && chown -R 10001:10001 /testbed \
    && chown -R root:root /opt/frontier/judge /opt/frontier/worker-python /opt/frontier/python-standalone \
    && chmod -R a-w /opt/frontier/judge /opt/frontier/worker-python /opt/frontier/python-standalone \
    && find /opt/frontier/judge -type d -exec chmod 0555 {} + \
    && find /opt/frontier/judge -type f -exec chmod 0444 {} + \
    && chmod 0400 /opt/frontier/judge/serving_eval/swe_data/instances.json /opt/frontier/judge/serving_eval/swe_data/provenance.json
ENV PYTHONPATH=/opt/frontier/judge PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/frontier/swe-wheelhouse PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /testbed
ENTRYPOINT []
CMD []

FROM prepared AS runtime
# Mutable runtime boundaries are a thin layer after immutable offline preparation.
COPY --chmod=0444 judge/serving_eval/sidecar_env.py /opt/frontier/judge/serving_eval/sidecar_env.py
COPY --chmod=0444 judge/serving_eval/swe_correctness_service.py /opt/frontier/judge/serving_eval/swe_correctness_service.py
COPY --chmod=0444 judge/serving_eval/swe_test_worker.py /opt/frontier/judge/serving_eval/swe_test_worker.py
COPY --chmod=0444 judge/serving_eval/swe_syntax_probe.py /opt/frontier/judge/serving_eval/swe_syntax_probe.py
COPY --chmod=0444 judge/serving_eval/swe_tool_exec.py /opt/frontier/judge/serving_eval/swe_tool_exec.py
COPY --chmod=0444 judge/serving_eval/swe_case_failure.py /opt/frontier/judge/serving_eval/swe_case_failure.py
COPY --chmod=0444 judge/serving_eval/swe_tool_service.py /opt/frontier/judge/serving_eval/swe_tool_service.py
COPY --chmod=0444 judge/serving_eval/testbed_reset.py /opt/frontier/judge/serving_eval/testbed_reset.py
COPY --chmod=0444 judge/serving_eval/tool_auth.py /opt/frontier/judge/serving_eval/tool_auth.py
RUN /opt/frontier/worker-python/bin/python -P -c "import serving_eval.swe_tool_service, serving_eval.swe_correctness_service, serving_eval.swe_test_worker"

FROM runtime AS offline-check
RUN --network=none PYTHONPATH=/opt/frontier/judge /opt/frontier/worker-python/bin/python -P -m serving_eval.swe_offline_setup --instance-id psf__requests-3362 --phase verify

FROM runtime
COPY --from=offline-check --chmod=0444 /opt/frontier/swe-wheelhouse/preflight.json /opt/frontier/swe-wheelhouse/preflight.json
